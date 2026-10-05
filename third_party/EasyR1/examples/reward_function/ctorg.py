#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTOrg reward (sparse mask, image (1,512,512,Z) in [0,1])

Reward = w_format*R_format + w_space*R_space + w_time*R_time + w_consistency*R_consistency

Input per sample (reward_inputs item):
{
  "response": str,
  "ground_truth": dict or json-string
}

Required GT keys (for consistency/time):
- image_rel_path: e.g. "volume-0/image.npy"
- mask_rel_path : e.g. "volume-0/mask_(6,512,512,75).npz"
- label_id      : 1..C (recommended)
Optional GT keys:
- gt_bbox_2d_list_512 / bbox_list_512 : list of [x1,y1,x2,y2] in 512 coords (for space reward)
- canon_size_xy : [W,H], default [512,512]

IMPORTANT:
- This reward expects the model output to be STRICT:
  <think>...</think><answer>...</answer>
- <answer> payload format:
  <answer>[{"slice": int, "bbox_2d_list": [[x1,y1,x2,y2], ...]}]</answer>
  where bbox coords are normalized to [0,1000].
"""

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from sam2.build_sam import build_sam2_video_predictor_npz

# =========================================================
# Config (edit here)
# =========================================================

# CTOrg npy root: contains volume-xxx subfolders
CTORG_NPY_ROOT = os.environ.get('CTORG_NPY_ROOT', 'data/ctorg/ct_org_npy')

# MedSAM2 / SAM2
MODEL_CFG = "configs/sam2.1_hiera_t512.yaml"
CKPT_PATH = os.environ.get('MEDSAM2_CHECKPOINT', 'checkpoints/MedSAM2_2411.pt')

# local clip radius around predicted slice: +/- LOCAL_RADIUS
LOCAL_RADIUS = 5
SAM2_INPUT_SIZE = 512

# bbox expansion in pixels for spatial IoU (optional)
BBOX_SHIFT = 0

# weights (sum=1.0)
W_FORMAT = 0.25
W_SPACE = 0.25
W_TIME = 0.25
W_CONSIST = 0.25

AUTOCAST_DTYPE = torch.bfloat16

DEBUG_EVERY = int(os.getenv("DEBUG_REWARD_EVERY", "0") or "0")
EMPTY_CACHE_AT_END_OF_BATCH = bool(int(os.getenv("EMPTY_CACHE_AT_END_OF_BATCH", "0") or "0"))

REWARD_NAME = "ctorg_format_space_time_consistency_sam2"
REWARD_TYPE = "batch"

# =========================================================
# Regex
# =========================================================
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
STRICT_THINK_ANSWER_RE = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL | re.IGNORECASE)

# =========================================================
# Globals
# =========================================================
_PREDICTOR = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_IMG_MEAN = None
_IMG_STD = None

# =========================================================
# Predictor init
# =========================================================
def _lazy_init_predictor():
    global _PREDICTOR, _IMG_MEAN, _IMG_STD
    if _PREDICTOR is not None:
        return
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"CKPT_PATH not found: {CKPT_PATH}")
    _PREDICTOR = build_sam2_video_predictor_npz(MODEL_CFG, CKPT_PATH)

    if _DEVICE == "cuda":
        _IMG_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=_DEVICE)[:, None, None]
        _IMG_STD  = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=_DEVICE)[:, None, None]

# =========================================================
# Basic helpers
# =========================================================
def _safe_json_loads(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x

def _normalize_tag_spacing(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"<\s*think\s*>", "<think>", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*/\s*think\s*>", "</think>", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*answer\s*>", "<answer>", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*/\s*answer\s*>", "</answer>", s, flags=re.IGNORECASE)
    return s

def _extract_answer_payload_first(response: str) -> Optional[str]:
    m = ANSWER_RE.search(response or "")
    if not m:
        return None
    return m.group(1).strip()

def _extract_think_payload_first(response: str) -> Optional[str]:
    m = THINK_RE.search(response or "")
    if not m:
        return None
    return m.group(1).strip()

def _is_strict_single_think_answer(response: str) -> bool:
    s = response or ""
    if not STRICT_THINK_ANSWER_RE.fullmatch(s):
        return False
    return s.lower().find("<think>") < s.lower().find("<answer>")

def _extract_first_json_substring(payload: str) -> Optional[str]:
    if not isinstance(payload, str) or not payload:
        return None
    s = payload.strip()
    starts = [(s.find("{"), "{"), (s.find("["), "[")]
    starts = [(idx, ch) for idx, ch in starts if idx != -1]
    if not starts:
        return None
    idx0, opener = min(starts, key=lambda x: x[0])
    closer = "}" if opener == "{" else "]"
    stack = []
    for i in range(idx0, len(s)):
        ch = s[i]
        if ch == opener:
            stack.append(opener)
        elif ch == closer:
            if stack:
                stack.pop()
                if not stack:
                    return s[idx0:i + 1]
    return None

def _try_json_load(payload: str) -> Optional[Any]:
    try:
        return json.loads(payload)
    except Exception:
        return None

def _parse_list_or_dict(payload: str) -> Optional[List[Any]]:
    data = _try_json_load(payload)
    if data is None:
        jsub = _extract_first_json_substring(payload)
        if jsub is None:
            return None
        data = _try_json_load(jsub)
        if data is None:
            return None

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("entities", None), list):
            return data["entities"]
        return [data]
    return None

def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))

def _qwen_0_1000_to_pix_xyxy_inclusive(b: List[int], W: int = 512, H: int = 512) -> Optional[List[int]]:
    if not (isinstance(b, list) and len(b) == 4):
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in b]
    except Exception:
        return None
    x1 = _clamp_int(x1, 0, 1000)
    y1 = _clamp_int(y1, 0, 1000)
    x2 = _clamp_int(x2, 0, 1000)
    y2 = _clamp_int(y2, 0, 1000)

    def map_x(xn: int) -> int:
        return int(round((xn / 1000.0) * (W - 1)))

    def map_y(yn: int) -> int:
        return int(round((yn / 1000.0) * (H - 1)))

    x1p, x2p = map_x(x1), map_x(x2)
    y1p, y2p = map_y(y1), map_y(y2)

    x1p = _clamp_int(x1p, 0, W - 1)
    x2p = _clamp_int(x2p, 0, W - 1)
    y1p = _clamp_int(y1p, 0, H - 1)
    y2p = _clamp_int(y2p, 0, H - 1)

    if x2p < x1p:
        x1p, x2p = x2p, x1p
    if y2p < y1p:
        y1p, y2p = y2p, y1p

    if x2p <= x1p:
        x2p = min(W - 1, x1p + 1)
    if y2p <= y1p:
        y2p = min(H - 1, y1p + 1)

    return [x1p, y1p, x2p, y2p]

def _normalize_gt_bbox_xyxy_inclusive(b: Any, W: int = 512, H: int = 512) -> Optional[List[int]]:
    if not (isinstance(b, list) and len(b) == 4):
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in b]
    except Exception:
        return None

    x1 = _clamp_int(x1, 0, W - 1)
    x2 = _clamp_int(x2, 0, W - 1)
    y1 = _clamp_int(y1, 0, H - 1)
    y2 = _clamp_int(y2, 0, H - 1)

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)

    return [x1, y1, x2, y2]

def _expand_bbox_xyxy_inclusive(bb: List[int], shift: int, W: int, H: int) -> List[int]:
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
# IoU + Hungarian
# =========================================================
def _iou_xyxy_inclusive(a: Optional[List[int]], b: Optional[List[int]]) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0

def _hungarian_max_iou(pred_boxes: List[List[int]], gt_boxes: List[List[int]]) -> float:
    N = len(pred_boxes)
    M = len(gt_boxes)
    if N == 0 or M == 0:
        return 0.0

    cost = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            cost[i, j] = 1.0 - float(_iou_xyxy_inclusive(pred_boxes[i], gt_boxes[j]))

    try:
        from scipy.optimize import linear_sum_assignment
        ri, cj = linear_sum_assignment(cost)
        s = 0.0
        for i, j in zip(ri.tolist(), cj.tolist()):
            s += float(1.0 - cost[i, j])
        return float(max(0.0, s))
    except Exception:
        used_g = set()
        s = 0.0
        for i in range(N):
            best_j = -1
            best = 0.0
            for j in range(M):
                if j in used_g:
                    continue
                iou = float(_iou_xyxy_inclusive(pred_boxes[i], gt_boxes[j]))
                if iou > best:
                    best = iou
                    best_j = j
            if best_j >= 0:
                used_g.add(best_j)
                s += best
        return float(max(0.0, s))

# =========================================================
# Format reward (KEEP THINK)
# =========================================================
def format_reward(response: str) -> float:
    """
    [0,1], avg of:
      s1 strict think+answer only
      s2 think non-empty
      s3 answer json parse ok
      s4 bbox validity fraction in first item's bbox_2d_list
    """
    s1 = s2 = s3 = s4 = 0.0

    if _is_strict_single_think_answer(response):
        s1 = 1.0

    t = _extract_think_payload_first(response)
    if t is not None and len(t.strip()) > 0:
        s2 = 1.0

    ans = _extract_answer_payload_first(response)
    if ans is None:
        return float((s1 + s2 + s3 + s4) / 4.0)

    items = _parse_list_or_dict(ans)
    if items is None or not isinstance(items, list) or len(items) == 0:
        return float((s1 + s2 + s3 + s4) / 4.0)
    s3 = 1.0

    first = items[0] if isinstance(items[0], dict) else {}
    total = 0
    valid = 0
    bbl = first.get("bbox_2d_list", None) if isinstance(first, dict) else None
    if isinstance(bbl, list):
        for b in bbl:
            total += 1
            if isinstance(b, list) and len(b) == 4:
                try:
                    x1, y1, x2, y2 = [int(round(float(v))) for v in b]
                    if 0 <= x1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y1 <= 1000 and 0 <= y2 <= 1000 and x1 < x2 and y1 < y2:
                        valid += 1
                except Exception:
                    pass
    if total > 0:
        s4 = float(valid) / float(total)

    return float((s1 + s2 + s3 + s4) / 4.0)

# =========================================================
# Parse predicted bboxes (+ optional pred_slice)
# =========================================================
def _parse_pred_bboxes_and_slice(response: str, W: int = 512, H: int = 512) -> Tuple[List[List[int]], Optional[int]]:
    ans = _extract_answer_payload_first(response)
    if ans is None:
        return [], None

    items = _parse_list_or_dict(ans)
    if items is None or not isinstance(items, list) or len(items) == 0:
        return [], None

    first = items[0] if isinstance(items[0], dict) else {}
    pred_slice = None
    try:
        pred_slice = int(first.get("slice"))
    except Exception:
        pred_slice = None

    boxes: List[List[int]] = []
    if isinstance(first, dict):
        bbl = first.get("bbox_2d_list", None)
        if isinstance(bbl, list):
            for b in bbl:
                if isinstance(b, list) and len(b) == 4:
                    bb = _qwen_0_1000_to_pix_xyxy_inclusive(b, W=W, H=H)
                    if bb is not None:
                        boxes.append(bb)
    return boxes, pred_slice

# =========================================================
# Local window helpers
# =========================================================
def _get_local_window(z_mid: int, Z: int, radius: int) -> Tuple[int, int]:
    z0 = max(0, int(z_mid) - int(radius))
    z1 = min(int(Z) - 1, int(z_mid) + int(radius))
    return z0, z1

# =========================================================
# CTOrg sparse GT expand on GPU (CRITICAL FIX)
# mask shape: (C, H*W*Z), flatten idx = (y*W + x)*Z + z
# =========================================================
def _gt_clip_and_areas_for_label_window_ctorg_sparse(
    mask_npz_path: str,
    label_id: int,
    z0: int,
    z1: int,
    Z: int,
    H: int,
    W: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      gt_clip: (T,H,W) bool on device
      areas:   (T,) int64 on device
    """
    from scipy import sparse

    z0 = max(0, int(z0))
    z1 = min(int(Z) - 1, int(z1))
    T = int(z1 - z0 + 1)
    if T <= 0:
        gt = torch.zeros((0, H, W), dtype=torch.bool, device=device)
        areas = torch.zeros((0,), dtype=torch.int64, device=device)
        return gt, areas

    sp = sparse.load_npz(mask_npz_path)
    r, c = sp.shape
    HWZ = int(H * W * Z)

    # expected: (C, HWZ)
    if c != HWZ:
        # 如果你未来遇到 transpose 保存，这里可以加 (HWZ, C) 分支；当前你给的例子是 (C,HWZ)，先直接返回空更安全
        gt = torch.zeros((T, H, W), dtype=torch.bool, device=device)
        areas = torch.zeros((T,), dtype=torch.int64, device=device)
        return gt, areas

    C = int(r)
    ch = int(label_id) - 1
    if ch < 0 or ch >= C:
        gt = torch.zeros((T, H, W), dtype=torch.bool, device=device)
        areas = torch.zeros((T,), dtype=torch.int64, device=device)
        return gt, areas

    sp = sp.tocsr()
    row = sp.getrow(ch)
    idx = row.indices.astype(np.int64, copy=False)
    if idx.size == 0:
        gt = torch.zeros((T, H, W), dtype=torch.bool, device=device)
        areas = torch.zeros((T,), dtype=torch.int64, device=device)
        return gt, areas

    # ctorg decoding: z = idx % Z, yx = idx // Z
    z_all = (idx % Z).astype(np.int64, copy=False)
    yx_all = (idx // Z).astype(np.int64, copy=False)

    m = (z_all >= z0) & (z_all <= z1)
    if not np.any(m):
        gt = torch.zeros((T, H, W), dtype=torch.bool, device=device)
        areas = torch.zeros((T,), dtype=torch.int64, device=device)
        return gt, areas

    z_loc = (z_all[m] - z0).astype(np.int64, copy=False)
    yx = yx_all[m]
    ys = (yx // W).astype(np.int64, copy=False)
    xs = (yx %  W).astype(np.int64, copy=False)

    gt = torch.zeros((T, H, W), dtype=torch.bool, device=device)
    tz = torch.from_numpy(z_loc).to(device=device, non_blocking=True)
    ty = torch.from_numpy(ys).to(device=device, non_blocking=True)
    tx = torch.from_numpy(xs).to(device=device, non_blocking=True)
    gt[tz, ty, tx] = True

    areas = gt.flatten(1).sum(dim=1).to(dtype=torch.int64)
    return gt, areas

# =========================================================
# Image clip builder (local clip on GPU)
# image.npy is in [0,1], shape (1,512,512,Z)
# =========================================================
def _load_img_zyx01_from_npy(img_path: str) -> Optional[np.ndarray]:
    img = np.load(img_path, mmap_mode="r")
    if img.ndim != 4 or img.shape[0] != 1:
        return None
    # img: (1,512,512,Z) -> (Z,512,512)
    vol = np.asarray(img[0], dtype=np.float32)  # (H,W,Z)
    vol = np.transpose(vol, (2, 0, 1))          # (Z,H,W)
    return vol

def _resize_clip_to_512_rgb(img_zyx_01: np.ndarray, z0: int, z1: int) -> torch.Tensor:
    """
    img_zyx_01: (Z,512,512) float32 in [0,1]
    return: (T,3,512,512) on device, Imagenet-normalized
    """
    Z, H, W = img_zyx_01.shape
    z0 = max(0, int(z0))
    z1 = min(int(Z) - 1, int(z1))
    clip = img_zyx_01[z0:z1+1]  # (T,H,W)
    T = int(clip.shape[0])

    vol_u8 = np.rint(np.clip(clip, 0.0, 1.0) * 255.0).astype(np.uint8)
    out = np.zeros((T, 3, SAM2_INPUT_SIZE, SAM2_INPUT_SIZE), dtype=np.uint8)

    # H,W should already be 512; keep safe resize anyway
    for i in range(T):
        pil = Image.fromarray(vol_u8[i], mode="L").convert("RGB")
        if (H != SAM2_INPUT_SIZE) or (W != SAM2_INPUT_SIZE):
            pil = pil.resize((SAM2_INPUT_SIZE, SAM2_INPUT_SIZE))
        arr = np.array(pil, dtype=np.uint8).transpose(2, 0, 1)
        out[i] = arr

    x = out.astype(np.float32) / 255.0
    x = torch.from_numpy(x).to(_DEVICE, non_blocking=True)

    if _DEVICE == "cuda":
        x = (x - _IMG_MEAN) / _IMG_STD
    else:
        img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
        img_std  = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
        x = (x - img_mean) / img_std
    return x

# =========================================================
# Dice on GPU
# =========================================================
def _dice_binary_torch(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.bool()
    b = b.bool()
    inter = (a & b).sum()
    denom = a.sum() + b.sum()
    if denom.item() == 0:
        return 1.0
    d = (2.0 * inter.float()) / (denom.float() + eps)
    return float(d.clamp_(0.0, 1.0).item())

# =========================================================
# SAM2 union: separate propagation per bbox (stable)
# =========================================================
@torch.inference_mode()
def _medsam2_union_on_clip_from_bboxes_separate(
    predictor,
    img_clip: torch.Tensor,        # (T,3,512,512) on GPU
    z_mid_local: int,
    bbox_list_xyxy_512: List[List[int]],
) -> torch.Tensor:
    """
    Separate propagation per bbox (single obj per state), then union on GPU.
    Returns: union (T,512,512) bool on GPU
    """
    T = int(img_clip.shape[0])
    union = torch.zeros((T, SAM2_INPUT_SIZE, SAM2_INPUT_SIZE), dtype=torch.bool, device=img_clip.device)

    if len(bbox_list_xyxy_512) == 0:
        return union

    with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE, enabled=(_DEVICE == "cuda")):
        for bb in bbox_list_xyxy_512:
            box = np.array(bb, dtype=np.int32)

            state = predictor.init_state(img_clip, SAM2_INPUT_SIZE, SAM2_INPUT_SIZE)

            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=int(z_mid_local),
                obj_id=1,
                box=box,
            )

            lt = out_mask_logits
            if not torch.is_tensor(lt):
                lt = torch.as_tensor(lt, device=img_clip.device)
            else:
                lt = lt.to(device=img_clip.device, non_blocking=True)

            if lt.ndim == 4:
                pm = (lt[0, 0] > 0.0)
            elif lt.ndim == 3:
                pm = (lt[0] > 0.0)
            elif lt.ndim == 2:
                pm = (lt > 0.0)
            else:
                predictor.reset_state(state)
                continue

            mask_prompt_np = pm.to(torch.uint8).detach().cpu().numpy()
            predictor.add_new_mask(state, frame_idx=int(z_mid_local), obj_id=1, mask=mask_prompt_np)

            # forward
            try:
                for out_frame_idx, _, out_mask_logits2 in predictor.propagate_in_video(
                    state, start_frame_idx=int(z_mid_local), reverse=False
                ):
                    if out_frame_idx >= T:
                        break
                    lt2 = out_mask_logits2
                    if not torch.is_tensor(lt2):
                        lt2 = torch.as_tensor(lt2, device=img_clip.device)
                    else:
                        lt2 = lt2.to(device=img_clip.device, non_blocking=True)

                    if lt2.ndim == 4:
                        m = (lt2[:, 0] > 0.0).any(dim=0)
                    elif lt2.ndim == 3:
                        m = (lt2 > 0.0).any(dim=0)
                    elif lt2.ndim == 2:
                        m = (lt2 > 0.0)
                    else:
                        continue
                    union[int(out_frame_idx)] |= m
            except Exception:
                pass

            # backward: re-init from prompt mask
            predictor.reset_state(state)
            state = predictor.init_state(img_clip, SAM2_INPUT_SIZE, SAM2_INPUT_SIZE)
            predictor.add_new_mask(state, frame_idx=int(z_mid_local), obj_id=1, mask=mask_prompt_np)

            try:
                for out_frame_idx, _, out_mask_logits2 in predictor.propagate_in_video(
                    state, start_frame_idx=int(z_mid_local), reverse=True
                ):
                    if out_frame_idx < 0:
                        break
                    lt2 = out_mask_logits2
                    if not torch.is_tensor(lt2):
                        lt2 = torch.as_tensor(lt2, device=img_clip.device)
                    else:
                        lt2 = lt2.to(device=img_clip.device, non_blocking=True)

                    if lt2.ndim == 4:
                        m = (lt2[:, 0] > 0.0).any(dim=0)
                    elif lt2.ndim == 3:
                        m = (lt2 > 0.0).any(dim=0)
                    elif lt2.ndim == 2:
                        m = (lt2 > 0.0)
                    else:
                        continue
                    union[int(out_frame_idx)] |= m
            except Exception:
                pass

            predictor.reset_state(state)

    return union

# =========================================================
# Space reward (mIoU with max(N,M) denom)
# =========================================================
def space_reward_miou(pred_boxes_512: List[List[int]], gt_boxes_512: List[List[int]]) -> float:
    gt_norm = []
    for b in (gt_boxes_512 or []):
        bb = _normalize_gt_bbox_xyxy_inclusive(b, W=512, H=512)
        if bb is not None:
            gt_norm.append(_expand_bbox_xyxy_inclusive(bb, BBOX_SHIFT, 512, 512))

    pred_norm = []
    for b in (pred_boxes_512 or []):
        if b is None:
            continue
        pred_norm.append(_expand_bbox_xyxy_inclusive(b, BBOX_SHIFT, 512, 512))

    N = len(pred_norm)
    M = len(gt_norm)
    denom = float(max(1, max(N, M)))
    if N == 0 or M == 0:
        return 0.0

    sum_iou = _hungarian_max_iou(pred_norm, gt_norm)
    miou = float(sum_iou) / denom
    return float(max(0.0, min(1.0, miou)))

# =========================================================
# Time reward: GT area at pred_slice normalized by max area in window
# =========================================================
def time_reward_from_gt_areas(
    pred_slice: Optional[int],
    z0: int,
    z1: int,
    gt_areas_local: torch.Tensor,   # (T,)
) -> float:
    if pred_slice is None:
        return 0.0
    if pred_slice < z0 or pred_slice > z1:
        return 0.0
    T = int(z1 - z0 + 1)
    if gt_areas_local.numel() != T:
        return 0.0
    idx = int(pred_slice - z0)
    max_area = int(gt_areas_local.max().item()) if gt_areas_local.numel() > 0 else 0
    if max_area <= 0:
        return 0.0
    a = int(gt_areas_local[idx].item())
    return float(max(0.0, min(1.0, a / float(max_area))))

# =========================================================
# Consistency + time (CTOrg)
# =========================================================
def _compute_local_consistency_and_time(
    gt: Dict[str, Any],
    pred_boxes_512: List[List[int]],
    pred_slice: Optional[int],
    local_radius: int,
) -> Tuple[float, float]:
    if pred_slice is None or len(pred_boxes_512) == 0:
        return 0.0, 0.0

    image_rel_path = gt.get("image_rel_path", None)
    mask_rel_path = gt.get("mask_rel_path", None)
    label_id = gt.get("label_id", None)
    if image_rel_path is None or mask_rel_path is None or label_id is None:
        return 0.0, 0.0

    img_path = os.path.join(CTORG_NPY_ROOT, str(image_rel_path))
    mask_path = os.path.join(CTORG_NPY_ROOT, str(mask_rel_path))
    if (not os.path.exists(img_path)) or (not os.path.exists(mask_path)):
        return 0.0, 0.0

    vol_zyx = _load_img_zyx01_from_npy(img_path)
    if vol_zyx is None or vol_zyx.ndim != 3:
        return 0.0, 0.0

    Z, H, W = map(int, vol_zyx.shape)
    if pred_slice < 0 or pred_slice >= Z:
        return 0.0, 0.0

    z0, z1 = _get_local_window(pred_slice, Z, local_radius)
    _lazy_init_predictor()

    # image clip GPU
    img_clip = _resize_clip_to_512_rgb(vol_zyx, z0, z1)  # (T,3,512,512)
    z_mid_local = int(pred_slice - z0)

    # GT clip + areas (CRITICAL FIX)
    gt_clip, gt_areas = _gt_clip_and_areas_for_label_window_ctorg_sparse(
        mask_npz_path=mask_path,
        label_id=int(label_id),
        z0=z0,
        z1=z1,
        Z=Z,
        H=H,
        W=W,
        device=_DEVICE,
    )

    # time reward (pure GT)
    rt = time_reward_from_gt_areas(pred_slice=pred_slice, z0=z0, z1=z1, gt_areas_local=gt_areas)

    # SAM2 pred union
    pred_union = _medsam2_union_on_clip_from_bboxes_separate(
        _PREDICTOR, img_clip, z_mid_local, pred_boxes_512
    )  # (T,512,512) bool

    # dice
    T = min(int(pred_union.shape[0]), int(gt_clip.shape[0]))
    if T <= 0:
        return 0.0, rt
    dice = _dice_binary_torch(pred_union[:T], gt_clip[:T])

    return float(dice), float(rt)

# =========================================================
# Main compute_score (batch)
# =========================================================
def compute_score(
    reward_inputs: List[Dict[str, Any]],
    w_format: float = W_FORMAT,
    w_space: float = W_SPACE,
    w_time: float = W_TIME,
    w_consistency: float = W_CONSIST,
) -> List[Dict[str, float]]:
    scores: List[Dict[str, float]] = []

    for idx, inp in enumerate(reward_inputs):
        response_raw = inp.get("response", "")
        gt_raw = inp.get("ground_truth", {})

        response = _normalize_tag_spacing(str(response_raw))
        gt = _safe_json_loads(gt_raw)
        if not isinstance(gt, dict):
            gt = {}

        W = 512; H = 512

        # format
        rf = float(format_reward(response))

        # parse preds
        pred_boxes_512, pred_slice = _parse_pred_bboxes_and_slice(response, W=W, H=H)

        # GT boxes for spatial (optional)
        gt_boxes_512 = gt.get(
            "gt_bbox_2d_list_512",
            gt.get("bbox_list_512", gt.get("gt_bboxes", gt.get("gt_bbox_list_512", [])))
        )
        if not isinstance(gt_boxes_512, list):
            gt_boxes_512 = []

        # space
        rs = float(space_reward_miou(pred_boxes_512, gt_boxes_512))

        # consistency + time
        dice_pred_local, rt = _compute_local_consistency_and_time(
            gt=gt,
            pred_boxes_512=pred_boxes_512,
            pred_slice=pred_slice,
            local_radius=LOCAL_RADIUS,
        )
        rc = float(max(0.0, min(1.0, dice_pred_local)))
        rt = float(max(0.0, min(1.0, rt)))

        overall = (w_format * rf) + (w_space * rs) + (w_time * rt) + (w_consistency * rc)
        overall = float(max(0.0, min(1.0, overall)))

        if DEBUG_EVERY > 0 and (idx % DEBUG_EVERY == 0):
            print(
                f"[DEBUG reward] idx={idx} overall={overall:.4f} "
                f"format={rf:.4f} space={rs:.4f} time={rt:.4f} cons={rc:.4f} dice={dice_pred_local:.4f} "
                f"pred_slice={pred_slice} n_pred_box={len(pred_boxes_512)} n_gt_box={len(gt_boxes_512)}"
            )

        scores.append(
            {
                "overall": float(overall),
                "format": float(rf),
                "space": float(rs),
                "time": float(rt),
                "consistency": float(rc),
                "dice_pred_local": float(dice_pred_local),
            }
        )

    if _DEVICE == "cuda" and EMPTY_CACHE_AT_END_OF_BATCH:
        torch.cuda.empty_cache()

    return scores

# =========================================================
# Demo main
# =========================================================
if __name__ == "__main__":
    # Demo (need to point to real files under CTORG_NPY_ROOT)
    predict_resp = "<think>locate bladder</think><answer>[{\"slice\":65,\"bbox_2d_list\":[[395,410,619,592]]}]</answer>"

    gt = {
        "case_id": "volume-0",
        "label_name": "bladder",
        "label_id": 2,
        "image_rel_path": "volume-0/image.npy",
        "mask_rel_path": "volume-0/mask_(6,512,512,75).npz",
        "canon_size_xy": [512, 512],
        # optional for space reward:
        "bbox_list_512": [[200,205,309,296]],
    }

    reward_inputs = [{"response": predict_resp, "ground_truth": gt}]
    out = compute_score(reward_inputs)
    print(out[0])
