#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse

import torch
import multiprocessing as mp
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor_npz

torch.set_float32_matmul_precision("high")

# =========================================================
# Config (edit here)
# =========================================================
pred_jsonl_results = os.environ.get('MEDVOL_CTORG_SEGMENTATION_PRED_JSONL_RESULTS', 'outputs/ctorg/predictions.jsonl')
test_json_file = os.environ.get('MEDVOL_CTORG_SEGMENTATION_TEST_JSON_FILE', 'data/ctorg/ct_org_frames_refseg_test.json')

train_file_root_path = os.environ.get('MEDVOL_CTORG_SEGMENTATION_TRAIN_FILE_ROOT_PATH', 'data/ctorg/ct_org_npy')
output_csv_path = os.environ.get('MEDVOL_EVAL_OUTPUT_CSV', 'outputs/ctorg/segmentation_metrics.csv')

# MedSAM2
MODEL_CFG = "configs/sam2.1_hiera_t512.yaml"
CKPT_PATH = os.environ.get('MEDSAM2_CHECKPOINT', 'checkpoints/MedSAM2_2411.pt')

# prompt bbox expansion on each side (in pixels, 512-space)
BBOX_SHIFT = 0

# =======================
# Multiprocessing + GPU map
# =======================
NUM_WORKERS = int(os.environ.get('MEDVOL_EVAL_NUM_WORKERS', '1'))
GPU_IDS = [int(x) for x in os.environ.get('MEDVOL_EVAL_GPU_IDS', '0').split(',')]
WORKERS_PER_GPU = int(os.environ.get('MEDVOL_EVAL_WORKERS_PER_GPU', '1'))

SAM2_INPUT_SIZE = 512

LABEL_NAME_TO_ID = {
    "liver": 1,
    "bladder": 2,
    "lungs": 3,
    "kidneys": 4,
    "bone": 5,
    "brain": 6,
}
ID_TO_LABEL_NAME = {v: k for k, v in LABEL_NAME_TO_ID.items()}

SPLIT_KIDNEY_COMPONENTS = True
IOU_MATCH_THRESHOLD = 0.0
GLOBAL_SEED = 2024

# =========================================================
# IMPORTANT: CSR decode mode
# You debugged: [Pick] best_mode=yxz_C score=1.0000
# means idx flatten order is (H,W,Z) with z fastest:
# idx = y*(W*Z) + x*Z + z
# =========================================================
CSR_DECODE_MODE = "yxz_C"  # or "zyx_C" if you later confirm different datasets

# =========================================================
# Globals for multiprocessing
# =========================================================
_G_PREDS: Optional[List[Dict[str, Any]]] = None
_G_GTS: Optional[List[Dict[str, Any]]] = None
_G_PREDICTOR = None
_G_DEVICE_ID = None  # assigned gpu id (visible index)


# =========================================================
# Parsing model outputs
# =========================================================
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)


@dataclass
class ParsedAnswer:
    ok: bool
    slice_id: Optional[int]
    bbox_list_1000: List[List[int]]
    reason: str = ""


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(str(x).strip())
    except Exception:
        return None


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _fix_box_1000(bb: Any) -> Optional[List[int]]:
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return None
    vals = []
    for v in bb:
        iv = _safe_int(v)
        if iv is None:
            return None
        vals.append(_clamp(iv, 0, 1000))
    x1, y1, x2, y2 = vals
    if x2 <= x1:
        x2 = min(1000, x1 + 1)
    if y2 <= y1:
        y2 = min(1000, y1 + 1)
    return [x1, y1, x2, y2]


def parse_prediction_text(pred_text: str) -> ParsedAnswer:
    if not isinstance(pred_text, str) or pred_text.strip() == "":
        return ParsedAnswer(False, None, [], "empty_predict")

    m = _ANSWER_RE.search(pred_text)
    if not m:
        return ParsedAnswer(False, None, [], "no_answer_tag")

    inner = m.group(1).strip()

    obj = None
    try:
        obj = json.loads(inner)
    except Exception:
        m2 = re.search(r"(\[.*\])", inner, flags=re.DOTALL)
        if m2:
            try:
                obj = json.loads(m2.group(1))
            except Exception:
                obj = None

    if obj is None:
        return ParsedAnswer(False, None, [], "answer_not_json")

    if not (isinstance(obj, list) and len(obj) >= 1 and isinstance(obj[0], dict)):
        return ParsedAnswer(False, None, [], "wrong_schema")

    item = obj[0]
    slice_id = _safe_int(item.get("slice", None))
    if slice_id is None:
        return ParsedAnswer(False, None, [], "missing_slice")

    bbox_list: List[List[int]] = []
    if isinstance(item.get("bbox_2d_list", None), list):
        for bb in item["bbox_2d_list"]:
            fixed = _fix_box_1000(bb)
            if fixed is not None:
                bbox_list.append(fixed)

    if len(bbox_list) == 0 and isinstance(item.get("bbox_2d", None), list):
        fixed = _fix_box_1000(item["bbox_2d"])
        if fixed is not None:
            bbox_list.append(fixed)

    if len(bbox_list) == 0:
        return ParsedAnswer(False, slice_id, [], "no_valid_bbox")

    return ParsedAnswer(True, slice_id, bbox_list, "")


def denorm_1000_to_512(bb_1000: List[int]) -> List[int]:
    x1n, y1n, x2n, y2n = bb_1000
    x1 = int(round(x1n / 1000.0 * 512.0))
    x2 = int(round(x2n / 1000.0 * 512.0))
    y1 = int(round(y1n / 1000.0 * 512.0))
    y2 = int(round(y2n / 1000.0 * 512.0))

    x1 = _clamp(x1, 0, 511)
    x2 = _clamp(x2, 0, 511)
    y1 = _clamp(y1, 0, 511)
    y2 = _clamp(y2, 0, 511)

    if x2 <= x1:
        x2 = min(511, x1 + 1)
    if y2 <= y1:
        y2 = min(511, y1 + 1)

    return [x1, y1, x2, y2]


def expand_bbox_xyxy_inclusive(bb: List[int], shift: int, W: int, H: int) -> List[int]:
    x1, y1, x2, y2 = map(int, bb)
    x1 = max(0, x1 - shift)
    y1 = max(0, y1 - shift)
    x2 = min(W - 1, x2 + shift)
    y2 = min(H - 1, y2 + shift)
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return [x1, y1, x2, y2]


# =========================================================
# GT helpers
# =========================================================
def resolve_mask_npz(case_dir: Path) -> Optional[Path]:
    cands = sorted(case_dir.glob("mask_*.npz"))
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    for c in cands:
        if c.name.startswith("mask_("):
            return c
    return cands[0]


def load_channel_mapping(case_dir: Path) -> Optional[List[int]]:
    p = case_dir / "mask_labels.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    m = obj.get("channel_to_label_id", None)
    if isinstance(m, list):
        try:
            return [int(x) for x in m]
        except Exception:
            return None
    return None


def infer_ZHW_from_csr(mat: sparse.csr_matrix, H: int = 512, W: int = 512) -> Tuple[int, int, int]:
    C, N = mat.shape
    HW = H * W
    if N % HW != 0:
        raise ValueError(f"CSR N not divisible by H*W: shape={mat.shape}, H={H}, W={W}")
    Z = N // HW
    return Z, H, W


def bbox_from_mask2d(mask2d: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask2d > 0)
    if xs.size == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    x1 = _clamp(x1, 0, 511)
    x2 = _clamp(x2, 0, 511)
    y1 = _clamp(y1, 0, 511)
    y2 = _clamp(y2, 0, 511)

    if x2 <= x1:
        x2 = min(511, x1 + 1)
    if y2 <= y1:
        y2 = min(511, y1 + 1)
    return [x1, y1, x2, y2]


def connected_components_2d(mask: np.ndarray) -> List[np.ndarray]:
    H, W = mask.shape
    mask = (mask > 0).astype(np.uint8)
    visited = np.zeros_like(mask, dtype=np.uint8)
    comps: List[np.ndarray] = []

    for y in range(H):
        for x in range(W):
            if mask[y, x] == 0 or visited[y, x] == 1:
                continue
            stack = [(y, x)]
            visited[y, x] = 1
            coords = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                if cy > 0 and mask[cy - 1, cx] and not visited[cy - 1, cx]:
                    visited[cy - 1, cx] = 1
                    stack.append((cy - 1, cx))
                if cy + 1 < H and mask[cy + 1, cx] and not visited[cy + 1, cx]:
                    visited[cy + 1, cx] = 1
                    stack.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and not visited[cy, cx - 1]:
                    visited[cy, cx - 1] = 1
                    stack.append((cy, cx - 1))
                if cx + 1 < W and mask[cy, cx + 1] and not visited[cy, cx + 1]:
                    visited[cy, cx + 1] = 1
                    stack.append((cy, cx + 1))

            comp = np.zeros_like(mask, dtype=np.uint8)
            yy = [c[0] for c in coords]
            xx = [c[1] for c in coords]
            comp[yy, xx] = 1
            comps.append(comp)

    comps.sort(key=lambda m: int(m.sum()), reverse=True)
    return comps


def _decode_csr_indices_to_zyx(
    idx: np.ndarray,
    Z: int,
    H: int,
    W: int,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return z_all, y_all, x_all from flattened indices.
    - mode="yxz_C": idx = y*(W*Z) + x*Z + z
    - mode="zyx_C": idx = z*(H*W) + y*W + x
    """
    idx = idx.astype(np.int64, copy=False)

    if mode == "yxz_C":
        WZ = W * Z
        y_all = idx // WZ
        rem = idx % WZ
        x_all = rem // Z
        z_all = rem % Z
    elif mode == "zyx_C":
        HW = H * W
        z_all = idx // HW
        rem = idx % HW
        y_all = rem // W
        x_all = rem % W
    else:
        raise ValueError(f"Unsupported CSR decode mode: {mode}")

    ok = (z_all >= 0) & (z_all < Z) & (y_all >= 0) & (y_all < H) & (x_all >= 0) & (x_all < W)
    return z_all[ok], y_all[ok], x_all[ok]


def compute_gt_key_slice_and_bbox_list(
    mat: sparse.csr_matrix,
    channel_to_label_id: Optional[List[int]],
    label_id: int,
    selected_slices: List[int],
    *,
    decode_mode: str = CSR_DECODE_MODE,
    H: int = 512,
    W: int = 512,
    label_name: Optional[str] = None,
) -> Tuple[Optional[int], List[List[int]], List[int], np.ndarray]:
    """
    Returns:
      best_z,
      bbox_list_512 (on best_z),
      pixels_list (per bbox),
      gt_union_3d bool (Z,H,W)
    """
    C, N = mat.shape
    Z, H, W = infer_ZHW_from_csr(mat, H=H, W=W)

    # label_id -> channel
    labelid_to_ch: Dict[int, int] = {}
    if channel_to_label_id is not None and len(channel_to_label_id) == C:
        for ch, lid in enumerate(channel_to_label_id):
            labelid_to_ch[int(lid)] = ch
    else:
        for ch in range(C):
            labelid_to_ch[ch + 1] = ch

    ch = labelid_to_ch.get(int(label_id), None)
    if ch is None or ch < 0 or ch >= C:
        return None, [], [], np.zeros((Z, H, W), dtype=bool)

    row = mat.getrow(ch)
    idx = row.indices
    if idx.size == 0:
        return None, [], [], np.zeros((Z, H, W), dtype=bool)

    # decode indices -> gt_union_3d
    z_all, y_all, x_all = _decode_csr_indices_to_zyx(idx, Z=Z, H=H, W=W, mode=decode_mode)

    gt_union_3d = np.zeros((Z, H, W), dtype=np.uint8)
    gt_union_3d[z_all, y_all, x_all] = 1
    gt_union_3d = gt_union_3d.astype(bool)

    # choose best slice among selected_slices by GT pixels
    counts = np.bincount(z_all, minlength=Z)

    selected_slices = [int(z) for z in selected_slices if 0 <= int(z) < Z]
    if len(selected_slices) == 0:
        return None, [], [], gt_union_3d

    best_z = None
    best_pixels = 0
    for z in selected_slices:
        p = int(counts[z])
        if p > best_pixels:
            best_pixels = p
            best_z = int(z)

    if best_z is None or best_pixels == 0:
        return None, [], [], gt_union_3d

    mask2d = gt_union_3d[best_z].astype(np.uint8)

    # bbox list on best slice
    bbox_list: List[List[int]] = []
    pixels_list: List[int] = []

    # FIX: kidney split should depend on label_name/id; ct_org uses kidneys=4 not 2
    is_kidney = False
    if label_name is not None:
        is_kidney = (label_name.lower() in ["kidney", "kidneys"])
    else:
        # fallback: if your mapping differs, keep old behavior off by default
        is_kidney = False

    if is_kidney and SPLIT_KIDNEY_COMPONENTS:
        comps = connected_components_2d(mask2d)[:2]
        for comp in comps:
            bb = bbox_from_mask2d(comp)
            if bb is None:
                continue
            bbox_list.append(bb)
            pixels_list.append(int(comp.sum()))
    else:
        bb = bbox_from_mask2d(mask2d)
        if bb is not None:
            bbox_list.append(bb)
            pixels_list.append(int(mask2d.sum()))

    return best_z, bbox_list, pixels_list, gt_union_3d


# =========================================================
# MedSAM2 runner
# =========================================================
def dice_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / (denom + eps))


def resize_to_512_rgb_u8(img_zyx_01: np.ndarray) -> Tuple[torch.Tensor, int, int]:
    """
    img_zyx_01: (Z,H,W) float32 in [0,1]
    return torch float tensor (Z,3,512,512) normalized to imagenet, + original (H,W)
    """
    Z, H, W = img_zyx_01.shape
    vol_u8 = np.rint(np.clip(img_zyx_01, 0.0, 1.0) * 255.0).astype(np.uint8)

    out = np.zeros((Z, 3, SAM2_INPUT_SIZE, SAM2_INPUT_SIZE), dtype=np.uint8)
    for i in range(Z):
        pil = Image.fromarray(vol_u8[i], mode="L").convert("RGB")
        if (H != SAM2_INPUT_SIZE) or (W != SAM2_INPUT_SIZE):
            pil = pil.resize((SAM2_INPUT_SIZE, SAM2_INPUT_SIZE))
        arr = np.array(pil, dtype=np.uint8).transpose(2, 0, 1)
        out[i] = arr

    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    x = out.astype(np.float32) / 255.0
    x = torch.from_numpy(x).to(device)

    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=device)[:, None, None]
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=device)[:, None, None]
    x = (x - img_mean) / img_std
    return x, H, W


@torch.inference_mode()
def medsam2_segment_union_from_bboxes(
    predictor,
    img_tensor: torch.Tensor,
    video_height: int,
    video_width: int,
    z_mid: int,
    bbox_list_xyxy_inclusive_512: List[List[int]],
) -> np.ndarray:
    Z = img_tensor.shape[0]
    union = np.zeros((Z, video_height, video_width), dtype=np.uint8)

    for bb in bbox_list_xyxy_inclusive_512:
        box = np.array(bb, dtype=np.int32)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = predictor.init_state(img_tensor, video_height, video_width)

            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=int(z_mid),
                obj_id=1,
                box=box,
            )
            mask_prompt = (out_mask_logits[0] > 0.0).squeeze(0).detach().cpu().numpy().astype(np.uint8)

            _, _, masks = predictor.add_new_mask(
                inference_state, frame_idx=int(z_mid), obj_id=1, mask=mask_prompt
            )
            union[z_mid, ((masks[0] > 0.0).detach().cpu().numpy())[0]] = 1

            for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=int(z_mid), reverse=False
            ):
                union[out_frame_idx, (out_mask_logits[0] > 0.0).detach().cpu().numpy()[0]] = 1

            predictor.reset_state(inference_state)
            inference_state = predictor.init_state(img_tensor, video_height, video_width)
            predictor.add_new_mask(inference_state, frame_idx=int(z_mid), obj_id=1, mask=mask_prompt)
            for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=int(z_mid), reverse=True
            ):
                union[out_frame_idx, (out_mask_logits[0] > 0.0).detach().cpu().numpy()[0]] = 1

            predictor.reset_state(inference_state)

    return union.astype(bool)


# =========================================================
# IO
# =========================================================
def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def ensure_list_int(x: Any) -> List[int]:
    if not isinstance(x, list):
        return []
    out = []
    for v in x:
        iv = _safe_int(v)
        if iv is not None:
            out.append(int(iv))
    return out


# =========================================================
# Core eval
# =========================================================
def eval_one_item(predictor, pred_item: Dict[str, Any], gt_item: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()

    meta = gt_item.get("meta", {}) if isinstance(gt_item, dict) else {}
    case_id = str(meta.get("case_id", "")).strip()

    # GT label_name in your json looks like "Liver" -> lower
    label_name_raw = str(meta.get("label_name", "")).strip()
    label_name = label_name_raw.lower()

    selected_slices = ensure_list_int(meta.get("selected_slices", gt_item.get("selected_slices", [])))
    if len(selected_slices) == 0:
        # fallback: parse from image filenames if exist
        imgs = gt_item.get("images", [])
        for p in imgs:
            m = re.search(r"slice_(\d+)\.(png|jpg|jpeg|bmp|webp)", str(p))
            if m:
                selected_slices.append(int(m.group(1)))
        selected_slices = sorted(list(set(selected_slices)))

    label_id = LABEL_NAME_TO_ID.get(label_name, None)

    out = {
        "case_id": case_id,
        "label_name": label_name,
        "label_id": label_id,
        "num_images": int(meta.get("num_images", len(gt_item.get("images", [])))),
        "pred_parse_ok": False,
        "pred_reason": "",
        "pred_slice": None,
        "pred_num_boxes": 0,
        "gt_slice": None,
        "gt_num_boxes": 0,
        "dice_pred": None,
        "dice_upperbound": None,  # kept for compatibility
        "time_sec": None,
        "worker_device": int(_G_DEVICE_ID) if _G_DEVICE_ID is not None else None,
    }

    if not case_id or label_id is None:
        out["pred_reason"] = "missing_case_or_label"
        out["time_sec"] = round(time.time() - t0, 4)
        return out

    case_dir = Path(train_file_root_path) / case_id
    img_path = case_dir / "image.npy"
    mask_path = resolve_mask_npz(case_dir)
    if (not img_path.exists()) or (mask_path is None) or (not mask_path.exists()):
        out["pred_reason"] = "missing_image_or_mask"
        out["time_sec"] = round(time.time() - t0, 4)
        return out

    img = np.load(str(img_path), mmap_mode="r")
    img = np.squeeze(img)
    if img.ndim != 3:
        out["pred_reason"] = f"bad_image_shape={tuple(img.shape)}"
        out["time_sec"] = round(time.time() - t0, 4)
        return out

    # Ensure (Z,H,W)
    if img.shape[0] == 512 and img.shape[1] == 512 and img.shape[2] != 512:
        img = np.transpose(img, (2, 0, 1))

    mat = sparse.load_npz(str(mask_path)).tocsr()
    channel_map = load_channel_mapping(case_dir)

    gt_key_slice, gt_bbox_list_512, gt_pixels_list, gt_union_3d = compute_gt_key_slice_and_bbox_list(
        mat=mat,
        channel_to_label_id=channel_map,
        label_id=int(label_id),
        selected_slices=selected_slices,
        decode_mode=CSR_DECODE_MODE,   # <<< FIX HERE
        H=512,
        W=512,
        label_name=label_name,
    )
    out["gt_slice"] = None if gt_key_slice is None else int(gt_key_slice)
    out["gt_num_boxes"] = int(len(gt_bbox_list_512))

    if gt_key_slice is None or len(gt_bbox_list_512) == 0:
        out["pred_reason"] = "gt_empty_on_selected"
        out["time_sec"] = round(time.time() - t0, 4)
        return out

    pred_text = pred_item.get("predict", "")
    parsed = parse_prediction_text(pred_text)
    out["pred_parse_ok"] = bool(parsed.ok)
    out["pred_reason"] = parsed.reason
    out["pred_slice"] = None if parsed.slice_id is None else int(parsed.slice_id)

    if not parsed.ok:
        out["time_sec"] = round(time.time() - t0, 4)
        return out

    pred_slice = int(parsed.slice_id)
    out["pred_num_boxes"] = int(len(parsed.bbox_list_1000))

    pred_bbox_list_512 = [denorm_1000_to_512(bb) for bb in parsed.bbox_list_1000]
    pred_bbox_list_512 = [expand_bbox_xyxy_inclusive(bb, BBOX_SHIFT, 512, 512) for bb in pred_bbox_list_512]

    img_tensor, _, _ = resize_to_512_rgb_u8(img.astype(np.float32))
    Z = img_tensor.shape[0]
    if pred_slice < 0 or pred_slice >= Z:
        out["pred_reason"] = f"pred_slice_out_of_range pred={pred_slice} Z={Z}"
        out["time_sec"] = round(time.time() - t0, 4)
        return out

    pred_union = medsam2_segment_union_from_bboxes(
        predictor=predictor,
        img_tensor=img_tensor,
        video_height=SAM2_INPUT_SIZE,
        video_width=SAM2_INPUT_SIZE,
        z_mid=pred_slice,
        bbox_list_xyxy_inclusive_512=pred_bbox_list_512,
    )
    out["dice_pred"] = float(dice_binary(pred_union, gt_union_3d))

    out["time_sec"] = round(time.time() - t0, 4)
    return out


# =========================================================
# Multiprocessing: init + worker
# =========================================================
def _infer_worker_rank() -> int:
    name = mp.current_process().name
    m = re.search(r"(\d+)$", name)
    if not m:
        return 0
    return max(0, int(m.group(1)) - 1)


def _init_worker(preds, gts, model_cfg: str, ckpt_path: str, seed: int):
    """
    Key point: do NOT modify CUDA_VISIBLE_DEVICES inside worker.
    Just torch.cuda.set_device(assigned_gpu).
    """
    global _G_PREDS, _G_GTS, _G_PREDICTOR, _G_DEVICE_ID

    _G_PREDS = preds
    _G_GTS = gts

    rank = _infer_worker_rank()

    assert NUM_WORKERS <= len(GPU_IDS) * WORKERS_PER_GPU, (
        f"NUM_WORKERS={NUM_WORKERS} exceeds mapping capacity {len(GPU_IDS)}*{WORKERS_PER_GPU}."
    )
    gpu_slot = (rank // WORKERS_PER_GPU) % len(GPU_IDS)
    assigned_gpu = int(GPU_IDS[gpu_slot])

    if torch.cuda.is_available():
        torch.cuda.set_device(assigned_gpu)
        _G_DEVICE_ID = assigned_gpu
        print(
            f"[WorkerInit] pid={os.getpid()} rank={rank} -> assigned_gpu={assigned_gpu} "
            f"(current_device={torch.cuda.current_device()} visible_count={torch.cuda.device_count()})",
            flush=True,
        )
    else:
        _G_DEVICE_ID = None

    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    _G_PREDICTOR = build_sam2_video_predictor_npz(model_cfg, ckpt_path)


def _worker_eval(idx: int) -> Tuple[int, Dict[str, Any]]:
    global _G_PREDS, _G_GTS, _G_PREDICTOR, _G_DEVICE_ID
    try:
        r = eval_one_item(_G_PREDICTOR, _G_PREDS[idx], _G_GTS[idx])
    except Exception as e:
        gt_item = _G_GTS[idx] if _G_GTS is not None else {}
        meta = gt_item.get("meta", {}) if isinstance(gt_item, dict) else {}
        r = {
            "case_id": meta.get("case_id", ""),
            "label_name": str(meta.get("label_name", "")).lower(),
            "label_id": LABEL_NAME_TO_ID.get(str(meta.get("label_name", "")).lower(), None),
            "num_images": meta.get("num_images", None),
            "pred_parse_ok": False,
            "pred_reason": f"EXCEPTION: {repr(e)}",
            "pred_slice": None,
            "pred_num_boxes": 0,
            "gt_slice": None,
            "gt_num_boxes": 0,
            "dice_pred": None,
            "dice_upperbound": None,
            "time_sec": None,
            "worker_device": int(_G_DEVICE_ID) if _G_DEVICE_ID is not None else None,
        }
    return idx, r


# =========================================================
# Driver
# =========================================================
def main():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"CKPT_PATH not found: {CKPT_PATH}")
    if not os.path.exists(MODEL_CFG):
        print(f"[Warn] MODEL_CFG not found locally: {MODEL_CFG} (if sam2 resolves it internally, ignore)")

    preds = read_jsonl(pred_jsonl_results)
    gts = read_json(test_json_file)

    if not isinstance(gts, list):
        raise ValueError("test_json_file must be a list of items.")

    N = min(len(preds), len(gts))
    print(f"[Info] evaluating N={N}")
    print(f"[Info] CKPT={CKPT_PATH}")
    print(f"[Info] CFG ={MODEL_CFG}")
    print(f"[Info] CSR_DECODE_MODE={CSR_DECODE_MODE}")
    print(f"[Info] BBOX_SHIFT={BBOX_SHIFT}  SPLIT_KIDNEY_COMPONENTS={SPLIT_KIDNEY_COMPONENTS}")
    print(f"[Info] NUM_WORKERS={NUM_WORKERS} GPU_IDS={GPU_IDS} WORKERS_PER_GPU={WORKERS_PER_GPU}")
    print(f"[Info] torch.cuda.is_available()={torch.cuda.is_available()}  parent_visible_cuda_count={torch.cuda.device_count()}")

    mp.set_start_method("spawn", force=True)

    results: List[Optional[Dict[str, Any]]] = [None] * N

    with mp.Pool(
        processes=NUM_WORKERS,
        initializer=_init_worker,
        initargs=(preds, gts, MODEL_CFG, CKPT_PATH, GLOBAL_SEED),
    ) as pool:
        for idx, r in tqdm(pool.imap_unordered(_worker_eval, range(N)), total=N):
            results[idx] = r

    df = pd.DataFrame(results)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"[OK] wrote csv -> {output_csv_path}")

    df_ok = df.dropna(subset=["dice_pred"])
    if len(df_ok) == 0:
        print("[Warn] no valid rows with dice_pred.")
        print(df["pred_reason"].value_counts(dropna=False).head(20))
        return

    print("========== Summary ==========")
    print(f"Valid rows: {len(df_ok)}/{len(df)}")
    print(f"Mean dice_pred      : {float(df_ok['dice_pred'].mean()):.4f}")
    print("Top pred_reason:")
    print(df["pred_reason"].value_counts(dropna=False).head(10))


if __name__ == "__main__":
    main()
