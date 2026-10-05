#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

REWARD_NAME = "ct_spatiotemporal_grounding"
REWARD_TYPE = "batch"

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

# Strict: exactly one <think>...</think> then exactly one <answer>...</answer>, and NOTHING else
STRICT_THINK_ANSWER_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL | re.IGNORECASE
)


# ----------------------------
# Utils
# ----------------------------

def _safe_json_loads(x: Any) -> Any:
    """ground_truth may be string or already a python object."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def _normalize_tag_spacing(s: str) -> str:
    """
    Normalize tag spacing ONLY for think/answer tags.
    Avoid aggressive substitutions that could corrupt JSON/text.
    Examples:
      "< answer >" -> "<answer>"
      "</ think >" -> "</think>"
    """
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"<\s*think\s*>", "<think>", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*/\s*think\s*>", "</think>", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*answer\s*>", "<answer>", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*/\s*answer\s*>", "</answer>", s, flags=re.IGNORECASE)
    return s


def _count_tags(response: str) -> Dict[str, int]:
    s = response or ""
    return {
        "think_open": s.lower().count("<think>"),
        "think_close": s.lower().count("</think>"),
        "answer_open": s.lower().count("<answer>"),
        "answer_close": s.lower().count("</answer>"),
        "think_blocks": len(THINK_RE.findall(s)),
        "answer_blocks": len(ANSWER_RE.findall(s)),
    }


def _is_strict_single_think_answer(response: str) -> bool:
    """
    True iff:
      - whole string matches "<think>...</think><answer>...</answer>" with only whitespace outside
      - exactly one think block and exactly one answer block
      - tag open/close counts are also exactly 1 each (extra safety)
    """
    s = response or ""
    if not STRICT_THINK_ANSWER_RE.fullmatch(s):
        return False

    c = _count_tags(s)
    if not (
        c["think_open"] == 1
        and c["think_close"] == 1
        and c["answer_open"] == 1
        and c["answer_close"] == 1
        and c["think_blocks"] == 1
        and c["answer_blocks"] == 1
    ):
        return False

    # Ensure order: think before answer (defensive)
    return s.lower().find("<think>") < s.lower().find("<answer>")


def _extract_think_payload_first(response: str) -> Optional[str]:
    """Return the first <think> payload if exists; does NOT require strict."""
    m = THINK_RE.search(response or "")
    if not m:
        return None
    return m.group(1).strip()


def _extract_answer_payload_first(response: str) -> Optional[str]:
    """
    Return the first <answer> payload if exists; does NOT require strict.
    If multiple <answer> blocks exist, this returns the first one.
    """
    m = ANSWER_RE.search(response or "")
    if not m:
        return None
    return m.group(1).strip()


def _try_json_load(payload: str) -> Optional[Any]:
    """Try json.loads directly."""
    try:
        return json.loads(payload)
    except Exception:
        return None


def _extract_first_json_substring(payload: str) -> Optional[str]:
    """
    Try to extract first JSON object/list substring from payload, tolerant to extra text.
    We scan for first '{' or '[' and find the matching closing bracket.
    """
    if not isinstance(payload, str) or not payload:
        return None

    s = payload.strip()
    # find first json start
    starts = [(s.find("{"), "{"), (s.find("["), "[")]
    starts = [(idx, ch) for idx, ch in starts if idx != -1]
    if not starts:
        return None
    idx0, ch0 = min(starts, key=lambda x: x[0])
    opener = ch0
    closer = "}" if opener == "{" else "]"

    stack = []
    for i in range(idx0, len(s)):
        ch = s[i]
        if ch == opener:
            stack.append(opener)
        elif ch == closer:
            if not stack:
                continue
            stack.pop()
            if not stack:
                return s[idx0 : i + 1]
    return None


def _parse_single_item_from_answer(payload: str) -> Optional[Dict[str, Any]]:
    """
    payload inside <answer>...</answer>, expected:
      [{"slice":123,"bbox_2d":[279,571,383,666]}]
    Preferred: a JSON list with EXACTLY ONE object.

    Robustness:
      - If direct json.loads fails, try to extract first JSON substring and load again.
      - Also accept a single JSON object {"slice":..., "bbox_2d":...} as fallback.
    """
    data = _try_json_load(payload)
    if data is None:
        jsub = _extract_first_json_substring(payload)
        if jsub is None:
            return None
        data = _try_json_load(jsub)
        if data is None:
            return None

    if isinstance(data, list):
        if len(data) != 1 or not isinstance(data[0], dict):
            return None
        return data[0]

    if isinstance(data, dict):
        # fallback: allow dict directly
        return data

    return None


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _qwen_0_1000_to_512_xyxy_inclusive(
    bbox_0_1000: List[int], W: int = 512, H: int = 512
) -> Optional[List[int]]:
    """
    Map Qwen bbox in [0,1000] coord space to pixel-inclusive bbox in [0,W-1]/[0,H-1].
    """
    if not (isinstance(bbox_0_1000, list) and len(bbox_0_1000) == 4):
        return None

    x1, y1, x2, y2 = bbox_0_1000

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
    """IoU for inclusive pixel boxes [x1,y1,x2,y2]."""
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


# ----------------------------
# Core rewards (single sample)
# ----------------------------

def format_reward(response: str) -> float:
    """
    Format reward in [0,1], averaged over 5 checks:

      s1) Strict single "<think>...</think><answer>...</answer>" and nothing else
          (THIS is the strict format bonus)
      s2) <think> exists and is non-empty (after strip)
      s3) <answer> exists (at least one) and the FIRST <answer> parses into a valid JSON item
      s4) slice exists and is int-castable
      s5) bbox_2d exists and is list length 4 (int-castable) and valid range/order (0..1000, x1<x2,y1<y2)

    Key change vs your old code:
      - We DO NOT gate s3/s4/s5 on the strict think+answer format anymore.
        Missing <think> should mainly hurt s1/s2, but answer correctness can still earn s3/s4/s5.
    """
    s1 = s2 = s3 = s4 = s5 = 0.0

    # strict full pattern bonus
    if _is_strict_single_think_answer(response):
        s1 = 1.0

    # think exists and non-empty (independent)
    think_payload = _extract_think_payload_first(response)
    if think_payload is not None and len(think_payload.strip()) > 0:
        s2 = 1.0

    # answer parse (independent)
    ans_payload = _extract_answer_payload_first(response)
    if ans_payload is None:
        return float((s1 + s2 + s3 + s4 + s5) / 5.0)

    item0 = _parse_single_item_from_answer(ans_payload)
    if item0 is None:
        return float((s1 + s2 + s3 + s4 + s5) / 5.0)

    s3 = 1.0

    # slice
    try:
        _ = int(item0.get("slice", None))
        s4 = 1.0
    except Exception:
        s4 = 0.0

    # bbox_2d
    bbox = item0.get("bbox_2d", None)
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            vals = [int(round(float(v))) for v in bbox]
            x1, y1, x2, y2 = vals
            if (
                0 <= x1 <= 1000
                and 0 <= x2 <= 1000
                and 0 <= y1 <= 1000
                and 0 <= y2 <= 1000
                and x1 < x2
                and y1 < y2
            ):
                s5 = 1.0
        except Exception:
            s5 = 0.0

    return float((s1 + s2 + s3 + s4 + s5) / 5.0)


def time_reward(response: str, ground_truth: Any) -> float:
    """
    Temporal reward in [0,1]:

    IMPORTANT CHANGE:
      - No longer requires strict "<think>...</think><answer>...</answer>".
      - Only requires a parseable FIRST <answer> item with a valid pred_slice.

    Rules:
      - if pred_slice not in hit_slices_info -> 0
      - else area(pred_slice) / max_area
    """
    try:
        gt = _safe_json_loads(ground_truth)
        if not isinstance(gt, dict):
            return 0.0

        hit_slices_info = gt.get("hit_slices_info", None)
        max_area = gt.get("max_area", None)
        if not isinstance(hit_slices_info, dict):
            return 0.0

        payload = _extract_answer_payload_first(response)
        if payload is None:
            return 0.0
        item0 = _parse_single_item_from_answer(payload)
        if item0 is None:
            return 0.0

        pred_slice = int(item0.get("slice", -1))
        key = str(pred_slice)
        if key not in hit_slices_info:
            return 0.0

        slice_info = hit_slices_info.get(key, {})
        area = slice_info.get("area", 0)
        try:
            area = float(area)
        except Exception:
            area = 0.0

        try:
            max_area_val = float(max_area) if max_area is not None else None
        except Exception:
            max_area_val = None

        if max_area_val is None:
            vals = []
            for v in hit_slices_info.values():
                if isinstance(v, dict):
                    try:
                        vals.append(float(v.get("area", 0)))
                    except Exception:
                        pass
            max_area_val = max(vals) if len(vals) > 0 else 0.0

        if max_area_val <= 0:
            return 1.0

        r = area / max_area_val
        return float(max(0.0, min(1.0, r)))
    except Exception:
        return 0.0


def space_reward(response: str, ground_truth: Any, iou_thr: float = 0.5) -> float:
    """
    Spatial reward in {0,1}:

    IMPORTANT CHANGE:
      - No longer requires strict "<think>...</think><answer>...</answer>".
      - Only requires a parseable FIRST <answer> item with a valid pred_slice + bbox_2d.

    Rules:
      - requires pred_slice in hit_slices_info
      - IoU(pred_bbox, gt_bbox_at_pred_slice) > iou_thr -> 1 else 0
    """
    try:
        gt = _safe_json_loads(ground_truth)
        if not isinstance(gt, dict):
            return 0.0

        hit_slices_info = gt.get("hit_slices_info", None)
        canon_size_xy = gt.get("canon_size_xy", [512, 512])
        if not isinstance(hit_slices_info, dict):
            return 0.0

        payload = _extract_answer_payload_first(response)
        if payload is None:
            return 0.0
        item0 = _parse_single_item_from_answer(payload)
        if item0 is None:
            return 0.0

        pred_slice = int(item0.get("slice", -1))
        pred_bbox_raw = item0.get("bbox_2d", None)
        if not (isinstance(pred_bbox_raw, list) and len(pred_bbox_raw) == 4):
            return 0.0

        key = str(pred_slice)
        if key not in hit_slices_info:
            return 0.0

        gt_bbox = hit_slices_info.get(key, {}).get("bbox_xyxy", None)
        if not (isinstance(gt_bbox, list) and len(gt_bbox) == 4):
            return 0.0

        try:
            W = int(canon_size_xy[0])
            H = int(canon_size_xy[1])
        except Exception:
            W, H = 512, 512

        pred_bbox_0_1000 = [int(round(float(v))) for v in pred_bbox_raw]
        pred_bbox_pix = _qwen_0_1000_to_512_xyxy_inclusive(pred_bbox_0_1000, W=W, H=H)
        if pred_bbox_pix is None:
            return 0.0

        iou = _iou_xyxy_inclusive(gt_bbox, pred_bbox_pix)
        return 1.0 if float(iou) > float(iou_thr) else 0.0
    except Exception:
        return 0.0


# ----------------------------
# Batch interface
# ----------------------------

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    w_format: float = 0.2,
    w_time: float = 0.3,
    w_space: float = 0.5,
    iou_thr: float = 0.5,
) -> List[Dict[str, float]]:
    """
    reward_inputs:
      [{
        "response": str,
        "ground_truth": {
            "gt_center_slice": int,
            "max_area": float,
            "canon_size_xy": [W,H],
            "hit_slices_info": {"138": {"area":..., "bbox_xyxy":[...]} , ...},
            "selected_slices": [73,80,...,181]   # optional
        }
      }, ...]

    Return per-sample dict:
      {"overall":..., "format":..., "time":..., "space":...}
    """
    scores: List[Dict[str, float]] = []

    debug_every = int(os.getenv("DEBUG_REWARD_EVERY", "0") or "0")
    for idx, inp in enumerate(reward_inputs):
        response_raw = inp.get("response", "")
        gt_raw = inp.get("ground_truth", {})

        response = _normalize_tag_spacing(str(response_raw))

        rf = format_reward(response)
        rt = time_reward(response, gt_raw)
        rs = space_reward(response, gt_raw, iou_thr=iou_thr)

        overall = float(w_format) * float(rf) + float(w_time) * float(rt) + float(w_space) * float(rs)
        overall = float(max(0.0, min(1.0, overall)))

        if debug_every > 0 and (idx % debug_every == 0):
            ans = _extract_answer_payload_first(response)
            print(
                "[DEBUG reward] idx=", idx,
                "format=", rf, "time=", rt, "space=", rs, "overall=", overall,
                "\n  response_head=", response[:200].replace("\n", "\\n"),
                "\n  answer_payload_head=", (ans[:200].replace("\n", "\\n") if isinstance(ans, str) else None),
            )

        scores.append(
            {"overall": float(overall), "format": float(rf), "time": float(rt), "space": float(rs)}
        )
    return scores


if __name__ == "__main__":
    # sanity check (toy)
    reward_inputs = [
        {
            # No think, but valid answer -> time/space should still work now
            "response": "<answer>[{\"slice\":152,\"bbox_2d\":[0,0,1000,1000]}]</answer>",
            "ground_truth": {
                "gt_center_slice": 152,
                "max_area": 1503,
                "canon_size_xy": [512, 512],
                "hit_slices_info": {
                    "152": {"area": 1503, "bbox_xyxy": [10, 10, 100, 100]},
                    "145": {"area": 800, "bbox_xyxy": [12, 12, 90, 90]},
                },
            },
        },
        {
            # Repeated <answer> (format should be punished, but time/space uses the FIRST answer)
            "response": (
                "Reason."
                "<answer>[{\"slice\":152,\"bbox_2d\":[0,0,1000,1000]}]</answer>"
                "<answer>[{\"slice\":145,\"bbox_2d\":[0,0,1000,1000]}]</answer>"
            ),
            "ground_truth": {
                "gt_center_slice": 152,
                "max_area": 1503,
                "canon_size_xy": [512, 512],
                "hit_slices_info": {
                    "152": {"area": 1503, "bbox_xyxy": [10, 10, 100, 100]},
                    "145": {"area": 800, "bbox_xyxy": [12, 12, 90, 90]},
                },
            },
        },
        {
            # Strict format sample (gets s1/s2 bonus)
            "response": (
                "<think>Finding is most visible on slice 152 with clear margins.</think>"
                "<answer>[{\"slice\":152,\"bbox_2d\":[0,0,1000,1000]}]</answer>"
            ),
            "ground_truth": {
                "gt_center_slice": 152,
                "max_area": 1503,
                "canon_size_xy": [512, 512],
                "hit_slices_info": {
                    "152": {"area": 1503, "bbox_xyxy": [10, 10, 100, 100]},
                },
            },
        },
    ]
    print(compute_score(reward_inputs, iou_thr=0.5))
