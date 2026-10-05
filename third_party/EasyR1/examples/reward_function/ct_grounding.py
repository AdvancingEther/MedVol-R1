import re
import json
from typing import Any, Dict, List, Optional, Tuple

# import numpy as np

# Metadata (match EasyR1 / veRL style)
REWARD_NAME = "ct_spatiotemporal_grounding"
REWARD_TYPE = "batch"


# ----------------------------
# Parsing helpers
# ----------------------------

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def _safe_json_loads(x: Any) -> Any:
    """ground_truth may be string or already a python object."""
    if isinstance(x, str):
        return json.loads(x)
    return x


def _extract_answer_payload(response: str) -> Optional[str]:
    m = ANSWER_RE.search(response or "")
    if not m:
        return None
    return m.group(1).strip()


def _parse_first_item_from_answer(payload: str) -> Optional[Dict[str, Any]]:
    """
    payload is inside <answer> ... </answer>, expected to be JSON list like:
      [{"frame_index":37,"bbox_2d":[279,571,383,666]}]
    We only take the first item as agreed.
    """
    try:
        data = json.loads(payload)
    except Exception:
        return None
    if not isinstance(data, list) or len(data) == 0:
        return None
    if not isinstance(data[0], dict):
        return None
    return data[0]


# ----------------------------
# Core rewards (single sample)
# ----------------------------

def format_reward(response: str) -> float:
    """
    Format reward in [0,1], averaged over 3 checks:
      1) Has <answer>...</answer>
      2) frame_index exists and is int-castable
      3) bbox_2d exists and is a list of length 4 (int-castable)
    """
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0

    payload = _extract_answer_payload(response)
    if payload is not None:
        s1 = 1.0
        item0 = _parse_first_item_from_answer(payload)
        if item0 is not None:
            # frame_index
            try:
                _ = int(item0.get("frame_index", None))
                s2 = 1.0
            except Exception:
                s2 = 0.0
            # bbox_2d
            bbox = item0.get("bbox_2d", None)
            if isinstance(bbox, list) and len(bbox) == 4:
                try:
                    _ = [int(round(float(v))) for v in bbox]
                    s3 = 1.0
                except Exception:
                    s3 = 0.0

    return float((s1 + s2 + s3) / 3.0)


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _qwen_0_1000_to_512_xyxy_inclusive(bbox_0_1000: List[int], W: int = 512, H: int = 512) -> Optional[List[int]]:
    """
    Map Qwen bbox in [0,1000] coord space to pixel-inclusive bbox in [0,W-1]/[0,H-1].
    """
    if not (isinstance(bbox_0_1000, list) and len(bbox_0_1000) == 4):
        return None

    x1, y1, x2, y2 = bbox_0_1000

    # clamp to [0,1000]
    try:
        x1 = _clamp_int(int(round(float(x1))), 0, 1000)
        y1 = _clamp_int(int(round(float(y1))), 0, 1000)
        x2 = _clamp_int(int(round(float(x2))), 0, 1000)
        y2 = _clamp_int(int(round(float(y2))), 0, 1000)
    except Exception:
        return None

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

    return [x1p, y1p, x2p, y2p]


def _iou_xyxy_inclusive(a: Optional[List[int]], b: Optional[List[int]]) -> float:
    """
    IoU for inclusive pixel boxes [x1,y1,x2,y2]
    """
    if a is None or b is None:
        return 0.0
    if not (isinstance(a, list) and isinstance(b, list) and len(a) == 4 and len(b) == 4):
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


def time_reward(response: str, ground_truth: Any) -> float:
    """
    Temporal reward in [0,1]:
      - if pred_frame not in key_slices_info -> 0
      - else area(pred_frame) / max_area
    Uses max_area from gt (as agreed), and key_slices_info from gt.
    """
    try:
        gt = _safe_json_loads(ground_truth)
        if not isinstance(gt, dict):
            return 0.0
        key_slices_info = gt.get("key_slices_info", None)
        max_area = gt.get("max_area", None)

        if not isinstance(key_slices_info, dict):
            return 0.0

        # parse pred
        payload = _extract_answer_payload(response)
        if payload is None:
            return 0.0
        item0 = _parse_first_item_from_answer(payload)
        if item0 is None:
            return 0.0
        pred_frame = int(item0.get("frame_index", -1))

        if str(pred_frame) not in key_slices_info:
            return 0.0

        # frame area
        frame_info = key_slices_info.get(str(pred_frame), {})
        area = frame_info.get("area", 0)

        try:
            area = float(area)
        except Exception:
            area = 0.0

        # max_area: preferred from gt; fallback compute max over keys (robust)
        try:
            max_area_val = float(max_area) if max_area is not None else None
        except Exception:
            max_area_val = None

        if max_area_val is None:
            # fallback
            vals = []
            for v in key_slices_info.values():
                if isinstance(v, dict):
                    try:
                        vals.append(float(v.get("area", 0)))
                    except Exception:
                        pass
            max_area_val = max(vals) if len(vals) > 0 else 0.0

        if max_area_val <= 0:
            # if hit but max_area invalid, treat as full credit (or could be 0.0)
            return 1.0

        r = area / max_area_val
        return float(max(0.0, min(1.0, r)))
    except Exception:
        return 0.0


def space_reward(response: str, ground_truth: Any, iou_thr: float = 0.5) -> float:
    """
    Spatial reward in {0,1}:
      - requires pred_frame in key_slices_info
      - IoU(pred_bbox, gt_bbox_at_pred_frame) > iou_thr -> 1 else 0
    Coordinate handling:
      - pred bbox: 0..1000 from Qwen -> map to 0..511
      - gt bbox: assumed already in 512 pixel coord (inclusive xyxy)
    """
    try:
        gt = _safe_json_loads(ground_truth)
        if not isinstance(gt, dict):
            return 0.0
        key_slices_info = gt.get("key_slices_info", None)
        canon_size_xy = gt.get("canon_size_xy", [512, 512])

        if not isinstance(key_slices_info, dict):
            return 0.0

        # parse pred
        payload = _extract_answer_payload(response)
        if payload is None:
            return 0.0
        item0 = _parse_first_item_from_answer(payload)
        if item0 is None:
            return 0.0

        pred_frame = int(item0.get("frame_index", -1))
        pred_bbox_raw = item0.get("bbox_2d", None)
        if not (isinstance(pred_bbox_raw, list) and len(pred_bbox_raw) == 4):
            return 0.0

        if str(pred_frame) not in key_slices_info:
            return 0.0

        gt_bbox = key_slices_info.get(str(pred_frame), {}).get("bbox_xyxy", None)
        if not (isinstance(gt_bbox, list) and len(gt_bbox) == 4):
            return 0.0

        # canon size (default 512x512)
        try:
            W = int(canon_size_xy[0])
            H = int(canon_size_xy[1])
        except Exception:
            W, H = 512, 512

        pred_bbox_0_1000 = [int(round(float(v))) for v in pred_bbox_raw]
        pred_bbox_512 = _qwen_0_1000_to_512_xyxy_inclusive(pred_bbox_0_1000, W=W, H=H)
        if pred_bbox_512 is None:
            return 0.0

        iou = _iou_xyxy_inclusive(gt_bbox, pred_bbox_512)
        return 1.0 if float(iou) > float(iou_thr) else 0.0

    except Exception:
        return 0.0


# ----------------------------
# Batch interface (match math.py)
# ----------------------------

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    w_format: float = 1.0 / 3.0,
    w_time: float = 1.0 / 3.0,
    w_space: float = 1.0 / 3.0,
    iou_thr: float = 0.5,
) -> List[Dict[str, float]]:
    """
    reward_inputs:
      [{
        "case_id": str (optional),
        "response": str,
        "ground_truth": {
            "gt_center_slice": int,
            "key_slices_info": {"48": {"area":..., "bbox_xyxy":[...]} , ...},
            "canon_size_xy": [512,512],
            "max_area": 1503   # REQUIRED by your design (recommended to include)
        }
      }, ...]

    Return per-sample dict:
      {"overall":..., "format":..., "time":..., "space":...}
    """
    scores: List[Dict[str, float]] = []
    for inp in reward_inputs:
        response_raw = inp.get("response", "")
        gt_raw = inp.get("ground_truth", {})

        # handle qwen2.5vl-32b tag spacing like math.py
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", str(response_raw))

        # allow python object ground_truth (dict); also allow string json
        ground_truth = gt_raw

        rf = format_reward(response)
        rt = time_reward(response, ground_truth)
        rs = space_reward(response, ground_truth, iou_thr=iou_thr)

        overall = float(w_format) * float(rf) + float(w_time) * float(rt) + float(w_space) * float(rs)
        # clamp to [0,1] (all components already [0,1], weights default sum to 1)
        overall = float(max(0.0, min(1.0, overall)))

        scores.append(
            {
                "overall": float(overall),
                "format": float(rf),
                "time": float(rt),
                "space": float(rs),
            }
        )

    return scores


if __name__ == "__main__":
    # sanity check (toy)
    reward_inputs = [
        {
            "response": '<answer>[{"frame_index":48,"bbox_2d":[0,0,1000,1000]}]</answer>',
            "ground_truth": {
                "gt_center_slice": 48,
                "max_area": 1503,
                "canon_size_xy": [512, 512],
                "key_slices_info": {
                    "48": {"area": 1503, "bbox_xyxy": [10, 10, 100, 100]},
                    "49": {"area": 800, "bbox_xyxy": [12, 12, 90, 90]},
                },
            },
        }
    ]
    print(compute_score(reward_inputs, iou_thr=0.5))
