#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KiTS23 template-based reward (NO SAM2)

Reward = w_format * R_format + w_space * R_space + w_time * R_time

Input per sample (reward_inputs item):
{
  "response": str,
  "ground_truth": dict or json-string
}

Expected GT keys (new parquet):
- image_rel_path: e.g. "case_00000/image.npy"          (optional for Z fallback)
- mask_rel_path : e.g. "case_00000/mask_(14,512,512,611).npz"
- template_index: 0..13   (preferred)
Optional fallback:
- label_id      : 1..C     (old style compatibility)

Optional GT keys:
- gt_bbox_2d_list_512 / bbox_list_512 : list of [x1,y1,x2,y2] in 512 coords
- canon_size_xy : [W,H], default [512,512]

Model output STRICT:
  <think>...</think><answer>...</answer>

<answer> payload format:
  <answer>[{"slice": int, "bbox_2d_list": [[x1,y1,x2,y2], ...]}]</answer>
  where bbox coords are normalized to [0,1000].

Notes:
- SAM2 / MedSAM2 propagation is completely removed.
- No consistency / Dice reward.
- Time reward is computed only from GT mask slice areas.
"""

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse


# =========================================================
# Config (edit here)
# =========================================================
KITS23_NPY_ROOT = os.environ.get('KITS23_NPY_ROOT', 'data/kits23/kits23_npy_m3d')

# local clip radius around predicted slice: +/- LOCAL_RADIUS
LOCAL_RADIUS = 5

# bbox expansion in pixels for spatial IoU (optional)
BBOX_SHIFT = 0

# weights (sum to 1.0)
W_FORMAT = 0.30
W_SPACE = 0.35
W_TIME = 0.35

DEBUG_EVERY = int(os.getenv("DEBUG_REWARD_EVERY", "0") or "0")

REWARD_NAME = "kits23_format_space_time_no_sam2"
REWARD_TYPE = "batch"


# =========================================================
# Regex
# =========================================================
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
STRICT_THINK_ANSWER_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)


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


# =========================================================
# BBox helpers
# =========================================================
def _qwen_0_1000_to_pix_xyxy_inclusive(
    b: List[int],
    W: int = 512,
    H: int = 512,
) -> Optional[List[int]]:
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


def _normalize_gt_bbox_xyxy_inclusive(
    b: Any,
    W: int = 512,
    H: int = 512,
) -> Optional[List[int]]:
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
        # greedy fallback
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
# Format reward
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
                    if (
                        0 <= x1 <= 1000 and
                        0 <= x2 <= 1000 and
                        0 <= y1 <= 1000 and
                        0 <= y2 <= 1000 and
                        x1 < x2 and
                        y1 < y2
                    ):
                        valid += 1
                except Exception:
                    pass

    if total > 0:
        s4 = float(valid) / float(total)

    return float((s1 + s2 + s3 + s4) / 4.0)


# =========================================================
# Parse predicted bboxes (+ optional pred_slice)
# =========================================================
def _parse_pred_bboxes_and_slice(
    response: str,
    W: int = 512,
    H: int = 512,
) -> Tuple[List[List[int]], Optional[int]]:
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
# Sparse GT: only per-slice areas for one channel in local window
# mask shape: (C, H*W*Z), flatten idx = (y*W + x)*Z + z
# =========================================================
def _gt_areas_for_channel_window_sparse(
    mask_npz_path: str,
    channel_index: int,   # 0-based
    z0: int,
    z1: int,
    Z: int,
    H: int,
    W: int,
) -> np.ndarray:
    """
    Returns:
      areas: (T,) int64 numpy
    """
    z0 = max(0, int(z0))
    z1 = min(int(Z) - 1, int(z1))
    T = int(z1 - z0 + 1)

    if T <= 0:
        return np.zeros((0,), dtype=np.int64)

    try:
        sp = sparse.load_npz(mask_npz_path)
    except Exception:
        return np.zeros((T,), dtype=np.int64)

    r, c = sp.shape
    HWZ = int(H * W * Z)

    # expected: (C, HWZ)
    if c != HWZ:
        return np.zeros((T,), dtype=np.int64)

    C = int(r)
    ch = int(channel_index)
    if ch < 0 or ch >= C:
        return np.zeros((T,), dtype=np.int64)

    sp = sp.tocsr()
    row = sp.getrow(ch)
    idx = row.indices.astype(np.int64, copy=False)

    if idx.size == 0:
        return np.zeros((T,), dtype=np.int64)

    # decoding: z = idx % Z, yx = idx // Z
    z_all = (idx % Z).astype(np.int64, copy=False)

    m = (z_all >= z0) & (z_all <= z1)
    if not np.any(m):
        return np.zeros((T,), dtype=np.int64)

    z_loc = (z_all[m] - z0).astype(np.int64, copy=False)

    # count voxels per local slice
    areas = np.bincount(z_loc, minlength=T).astype(np.int64, copy=False)
    if areas.shape[0] > T:
        areas = areas[:T]

    return areas


# =========================================================
# Resolve channel index from current parquet GT
# =========================================================
def _resolve_channel_index_from_gt(gt: Dict[str, Any]) -> Optional[int]:
    """
    Priority:
      1) template_index (new parquet, 0-based)
      2) label_id      (old style, 1-based -> convert to 0-based)
    """
    template_index = gt.get("template_index", None)
    if template_index is not None:
        try:
            ch = int(template_index)
            if ch >= 0:
                return ch
        except Exception:
            pass

    label_id = gt.get("label_id", None)
    if label_id is not None:
        try:
            ch = int(label_id) - 1
            if ch >= 0:
                return ch
        except Exception:
            pass

    return None


# =========================================================
# Z inference helpers
# =========================================================
def _infer_z_from_mask_rel_path(mask_rel_path: Any) -> Optional[int]:
    """
    Parse Z from file name like:
      case_xxx/mask_(14,512,512,611).npz
    """
    s = str(mask_rel_path or "")
    m = re.search(r"mask_\((\d+),(\d+),(\d+),(\d+)\)\.npz$", s)
    if not m:
        return None
    try:
        return int(m.group(4))
    except Exception:
        return None


def _infer_z_from_image_npy(img_path: str) -> Optional[int]:
    """
    Expected image.npy shape: (1, H, W, Z)
    """
    try:
        img = np.load(img_path, mmap_mode="r")
    except Exception:
        return None

    if not hasattr(img, "shape"):
        return None
    if len(img.shape) != 4:
        return None
    if int(img.shape[0]) != 1:
        return None

    try:
        return int(img.shape[-1])
    except Exception:
        return None


# =========================================================
# Space reward: mIoU with max(N,M) denom
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
# Time reward: GT area at pred_slice normalized by max area in local window
# =========================================================
def time_reward_from_gt_areas(
    pred_slice: Optional[int],
    z0: int,
    z1: int,
    gt_areas_local: np.ndarray,
) -> float:
    if pred_slice is None:
        return 0.0
    if pred_slice < z0 or pred_slice > z1:
        return 0.0

    T = int(z1 - z0 + 1)
    if gt_areas_local is None or int(gt_areas_local.shape[0]) != T:
        return 0.0

    idx = int(pred_slice - z0)
    if idx < 0 or idx >= T:
        return 0.0

    max_area = int(np.max(gt_areas_local)) if gt_areas_local.size > 0 else 0
    if max_area <= 0:
        return 0.0

    a = int(gt_areas_local[idx])
    return float(max(0.0, min(1.0, a / float(max_area))))


# =========================================================
# Time only (NO SAM2, NO consistency)
# =========================================================
def _compute_time_only(
    gt: Dict[str, Any],
    pred_slice: Optional[int],
    local_radius: int,
) -> float:
    if pred_slice is None:
        return 0.0

    mask_rel_path = gt.get("mask_rel_path", None)
    channel_index = _resolve_channel_index_from_gt(gt)

    if mask_rel_path is None or channel_index is None:
        return 0.0

    mask_path = os.path.join(KITS23_NPY_ROOT, str(mask_rel_path))
    if not os.path.exists(mask_path):
        return 0.0

    # First try: parse Z directly from mask filename (fastest)
    Z = _infer_z_from_mask_rel_path(mask_rel_path)

    # Fallback: read image.npy shape
    if Z is None:
        image_rel_path = gt.get("image_rel_path", None)
        if image_rel_path is not None:
            img_path = os.path.join(KITS23_NPY_ROOT, str(image_rel_path))
            if os.path.exists(img_path):
                Z = _infer_z_from_image_npy(img_path)

    if Z is None or Z <= 0:
        return 0.0

    if pred_slice < 0 or pred_slice >= Z:
        return 0.0

    # By your current dataset convention
    H = 512
    W = 512

    z0, z1 = _get_local_window(pred_slice, Z, local_radius)

    gt_areas = _gt_areas_for_channel_window_sparse(
        mask_npz_path=mask_path,
        channel_index=int(channel_index),
        z0=z0,
        z1=z1,
        Z=Z,
        H=H,
        W=W,
    )

    rt = time_reward_from_gt_areas(
        pred_slice=pred_slice,
        z0=z0,
        z1=z1,
        gt_areas_local=gt_areas,
    )
    return float(rt)


# =========================================================
# Main compute_score (batch)
# =========================================================
def compute_score(
    reward_inputs: List[Dict[str, Any]],
    w_format: float = W_FORMAT,
    w_space: float = W_SPACE,
    w_time: float = W_TIME,
) -> List[Dict[str, float]]:
    scores: List[Dict[str, float]] = []

    for idx, inp in enumerate(reward_inputs):
        response_raw = inp.get("response", "")
        gt_raw = inp.get("ground_truth", {})

        response = _normalize_tag_spacing(str(response_raw))
        gt = _safe_json_loads(gt_raw)
        if not isinstance(gt, dict):
            gt = {}

        W = 512
        H = 512

        # 1) format
        rf = float(format_reward(response))

        # 2) parse predictions
        pred_boxes_512, pred_slice = _parse_pred_bboxes_and_slice(response, W=W, H=H)

        # 3) GT boxes for spatial reward
        gt_boxes_512 = gt.get(
            "gt_bbox_2d_list_512",
            gt.get("bbox_list_512", gt.get("gt_bboxes", gt.get("gt_bbox_list_512", []))),
        )
        if not isinstance(gt_boxes_512, list):
            gt_boxes_512 = []

        # 4) space reward
        rs = float(space_reward_miou(pred_boxes_512, gt_boxes_512))

        # 5) time reward only
        rt = _compute_time_only(
            gt=gt,
            pred_slice=pred_slice,
            local_radius=LOCAL_RADIUS,
        )
        rt = float(max(0.0, min(1.0, rt)))

        # 6) final weighted score
        overall = (w_format * rf) + (w_space * rs) + (w_time * rt)
        overall = float(max(0.0, min(1.0, overall)))

        if DEBUG_EVERY > 0 and (idx % DEBUG_EVERY == 0):
            dbg_ch = _resolve_channel_index_from_gt(gt)
            print(
                f"[DEBUG reward] idx={idx} overall={overall:.4f} "
                f"format={rf:.4f} space={rs:.4f} time={rt:.4f} "
                f"pred_slice={pred_slice} ch={dbg_ch} "
                f"n_pred_box={len(pred_boxes_512)} n_gt_box={len(gt_boxes_512)}"
            )

        scores.append(
            {
                "overall": float(overall),
                "format": float(rf),
                "space": float(rs),
                "time": float(rt),
            }
        )

    return scores


# =========================================================
# Demo main
# =========================================================
if __name__ == "__main__":
    predict_resp = (
        "<think>locate the lesion</think>"
        "<answer>[{\"slice\":120,\"bbox_2d_list\":[[300,300,500,500]]}]</answer>"
    )

    gt = {
        "case_id": "case_00000",
        "template_id": "tumor_only_global",
        "template_index": 4,   # 0-based
        "image_rel_path": "case_00000/image.npy",
        "mask_rel_path": "case_00000/mask_(14,512,512,611).npz",
        "canon_size_xy": [512, 512],
        "bbox_list_512": [[150, 180, 240, 260]],
    }

    reward_inputs = [{"response": predict_resp, "ground_truth": gt}]
    out = compute_score(reward_inputs)
    print(out[0])
