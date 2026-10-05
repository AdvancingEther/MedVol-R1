#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Veason-style multi-entity reward (implicit miss/FP penalty via normalization):

- Parse FIRST <answer> as a list of predicted entities:
    [{"slice": int, "bbox_2d":[x1,y1,x2,y2]}, ...]
  where bbox_2d is in [0,1000] and will be mapped to pixel coords for IoU.

- Ground truth supports:
    gt["hit_slices_info_by_entity"] : {
        "0": {"131": {"bbox_xyxy":[...], "area":...}, ...},
        "1": {...},
        ...
    }
  canon_size_xy: [W,H]

- 1-to-1 matching:
  We build a "time-gated" score per (pred_i, gt_j):
    time_score_ij  = area_ratio(pred_slice in gt_j) else 0
    space_score_ij = IoU(pred_bbox_pix, gt_bbox_at_pred_slice) else 0
  Then we do Hungarian / assignment to maximize (1) sum of time_score_ij,
  tie-broken by (2) sum of space_score_ij.

- Veason-like normalization (implicit miss / over-detection penalty):
    denom = max(N_pred, N_gt)
    R_time  = (sum matched time_score_ij) / denom
    R_space = (sum matched space_score_ij) / denom
  So if N_pred != N_gt, reward is automatically reduced without explicit penalties.

- Total:
    R_total = w_format*R_format + w_time*R_time + w_space*R_space
  (no explicit -miss or -fp terms)
"""

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

REWARD_NAME = "ct_spatiotemporal_grounding"
REWARD_TYPE = "batch"

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

STRICT_THINK_ANSWER_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL | re.IGNORECASE
)


# ----------------------------
# Utils
# ----------------------------

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
    s = response or ""
    if not STRICT_THINK_ANSWER_RE.fullmatch(s):
        return False
    c = _count_tags(s)
    if not (
        c["think_open"] == 1 and c["think_close"] == 1 and
        c["answer_open"] == 1 and c["answer_close"] == 1 and
        c["think_blocks"] == 1 and c["answer_blocks"] == 1
    ):
        return False
    return s.lower().find("<think>") < s.lower().find("<answer>")


def _extract_think_payload_first(response: str) -> Optional[str]:
    m = THINK_RE.search(response or "")
    if not m:
        return None
    return m.group(1).strip()


def _extract_answer_payload_first(response: str) -> Optional[str]:
    m = ANSWER_RE.search(response or "")
    if not m:
        return None
    return m.group(1).strip()


def _try_json_load(payload: str) -> Optional[Any]:
    try:
        return json.loads(payload)
    except Exception:
        return None


def _extract_first_json_substring(payload: str) -> Optional[str]:
    if not isinstance(payload, str) or not payload:
        return None
    s = payload.strip()
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
                return s[idx0:i + 1]
    return None


def _parse_entity_list_from_answer(payload: str) -> Optional[List[Any]]:
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
    return max(lo, min(hi, v))


def _qwen_0_1000_to_pix_xyxy_inclusive(
    bbox_0_1000: List[int], W: int = 512, H: int = 512
) -> Optional[List[int]]:
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


def _normalize_gt_bbox_xyxy_inclusive(
    bbox: Any, W: int = 512, H: int = 512
) -> Optional[List[int]]:
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
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
    return [x1, y1, x2, y2]


def _iou_xyxy_inclusive(a: Optional[List[int]], b: Optional[List[int]]) -> float:
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
# GT / Pred preparation
# ----------------------------

def _get_gt_entities(gt: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    h_by_e = gt.get("hit_slices_info_by_entity", None)
    if isinstance(h_by_e, dict) and len(h_by_e) > 0:
        def _k_int(k: str) -> int:
            try:
                return int(k)
            except Exception:
                return 10**9

        for k in sorted(h_by_e.keys(), key=_k_int):
            h = h_by_e.get(k, None)
            if isinstance(h, dict):
                out.append({"entity_index": _k_int(k), "hit_slices_info": h})
        return out

    ents = gt.get("entities", None)
    if isinstance(ents, list) and len(ents) > 0:
        for e in ents:
            if not isinstance(e, dict):
                continue
            idx = e.get("entity_index", None)
            try:
                idx = int(idx)
            except Exception:
                idx = len(out)
            h = e.get("hit_slices_info", None)
            if isinstance(h, dict):
                out.append({"entity_index": idx, "hit_slices_info": h})
        if out:
            return out

    h = gt.get("hit_slices_info", None)
    if isinstance(h, dict):
        out.append({"entity_index": 0, "hit_slices_info": h})
    return out


def _parse_pred_entities(response: str, W: int, H: int) -> List[Dict[str, Any]]:
    payload = _extract_answer_payload_first(response)
    if payload is None:
        return []

    items = _parse_entity_list_from_answer(payload)
    if items is None or not isinstance(items, list) or len(items) == 0:
        return []

    preds: List[Dict[str, Any]] = []
    for it in items:
        pred = {"slice": None, "slice_valid": False, "bbox_pix": None}

        if not isinstance(it, dict):
            preds.append(pred)
            continue

        try:
            z = int(it.get("slice", None))
            pred["slice"] = z
            pred["slice_valid"] = True
        except Exception:
            pred["slice"] = None
            pred["slice_valid"] = False

        bbox = it.get("bbox_2d", None)
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                vals = [int(round(float(v))) for v in bbox]
                x1, y1, x2, y2 = vals
                if (
                    0 <= x1 <= 1000 and 0 <= x2 <= 1000 and
                    0 <= y1 <= 1000 and 0 <= y2 <= 1000 and
                    x1 < x2 and y1 < y2
                ):
                    pred["bbox_pix"] = _qwen_0_1000_to_pix_xyxy_inclusive(vals, W=W, H=H)
            except Exception:
                pred["bbox_pix"] = None

        preds.append(pred)

    return preds


# ----------------------------
# Matching (Veason-like: maximize time sum, tie-break by space sum)
# ----------------------------

def _best_time_space_matching(
    time_mat: List[List[float]],
    space_mat: List[List[float]],
) -> Tuple[List[Tuple[int, int]], float, float]:
    """
    1-1 assignment maximizing:
      (1) sum_time
      (2) sum_space (tie-break)

    time_mat: N x M (0..1)  (already gated; 0 if not hit)
    space_mat:N x M (0..1)  (0 if not hit or bbox invalid)

    Returns:
      matched_pairs: list of (pred_i, gt_j) actually matched (size <= min(N,M))
      sum_time over matched pairs
      sum_space over matched pairs
    """
    N = len(time_mat)
    M = len(time_mat[0]) if N > 0 else 0
    if N == 0 or M == 0:
        return [], 0.0, 0.0

    # backtracking over the smaller side (GT) tends to be fast because M <= 3 in your data
    order = list(range(M))

    best_sum_time = -1.0
    best_sum_space = -1.0
    best_assign: Dict[int, int] = {}  # gt_j -> pred_i
    used_preds: set = set()

    def dfs(t: int, sum_time: float, sum_space: float, assign: Dict[int, int]):
        nonlocal best_sum_time, best_sum_space, best_assign

        if t >= M:
            if (sum_time > best_sum_time) or (abs(sum_time - best_sum_time) < 1e-9 and sum_space > best_sum_space):
                best_sum_time = sum_time
                best_sum_space = sum_space
                best_assign = dict(assign)
            return

        j = order[t]

        # option: leave gt_j unmatched
        dfs(t + 1, sum_time, sum_space, assign)

        # option: match gt_j with any unused pred_i
        for i in range(N):
            if i in used_preds:
                continue
            used_preds.add(i)
            assign[j] = i
            dfs(
                t + 1,
                sum_time + float(time_mat[i][j]),
                sum_space + float(space_mat[i][j]),
                assign,
            )
            del assign[j]
            used_preds.remove(i)

    dfs(0, 0.0, 0.0, {})

    matched_pairs = [(pi, gj) for gj, pi in best_assign.items()]
    return matched_pairs, float(max(0.0, best_sum_time)), float(max(0.0, best_sum_space))


def _veason_metrics(
    response: str,
    ground_truth: Any,
    iou_thr: float = 0.5,
    binary_space_reward: bool = False,
) -> Dict[str, Any]:
    """
    Compute:
      denom = max(N_pred, N_gt)  (Veason-like implicit miss/FP penalty)

      R_time  = sum matched time_mat / denom
      R_space = sum matched space_mat / denom

    time_mat(i,j):
      if pred_slice in GT_j hit_slices_info:
         area_ratio = area_at_pred_slice / max_area_of_entity_j
      else 0

    space_mat(i,j):
      if pred_slice hits and both bboxes exist:
         IoU (or binary IoU>=thr)
      else 0
    """
    gt = _safe_json_loads(ground_truth)
    if not isinstance(gt, dict):
        return {"M": 0, "N": 0, "denom": 1, "R_time": 0.0, "R_space": 0.0, "sum_time": 0.0, "sum_space": 0.0}

    canon_size_xy = gt.get("canon_size_xy", [512, 512])
    try:
        W = int(canon_size_xy[0])
        H = int(canon_size_xy[1])
        if W <= 0 or H <= 0:
            W, H = 512, 512
    except Exception:
        W, H = 512, 512

    gt_entities = _get_gt_entities(gt)
    M = len(gt_entities)

    response = _normalize_tag_spacing(str(response))
    preds = _parse_pred_entities(response, W=W, H=H)
    N = len(preds)

    denom = float(max(1, max(N, M)))

    if M == 0 or N == 0:
        return {
            "M": int(M), "N": int(N), "denom": float(denom),
            "R_time": 0.0, "R_space": 0.0,
            "sum_time": 0.0, "sum_space": 0.0,
            "matched_pairs": [],
        }

    # Precompute per-entity max_area (over its hit_slices_info)
    ent_max_area: List[float] = []
    for ge in gt_entities:
        h = ge.get("hit_slices_info", {})
        mx = 0.0
        if isinstance(h, dict):
            for _, info in h.items():
                if isinstance(info, dict):
                    try:
                        mx = max(mx, float(info.get("area", 0.0)))
                    except Exception:
                        pass
        ent_max_area.append(mx if mx > 0 else 0.0)

    time_mat: List[List[float]] = [[0.0 for _ in range(M)] for _ in range(N)]
    space_mat: List[List[float]] = [[0.0 for _ in range(M)] for _ in range(N)]

    for i, p in enumerate(preds):
        if not p.get("slice_valid", False):
            continue
        z = int(p["slice"])
        z_key = str(z)

        for j, ge in enumerate(gt_entities):
            h = ge.get("hit_slices_info", {})
            if not isinstance(h, dict):
                continue
            if z_key not in h:
                continue

            # time: area ratio
            info = h.get(z_key, {}) if isinstance(h.get(z_key, {}), dict) else {}
            try:
                area = float(info.get("area", 0.0))
            except Exception:
                area = 0.0
            mx = float(ent_max_area[j]) if j < len(ent_max_area) else 0.0
            if mx <= 0:
                tscore = 1.0  # if missing max_area, treat as hit=1
            else:
                tscore = max(0.0, min(1.0, area / mx))
            time_mat[i][j] = float(tscore)

            # space: IoU
            pred_bbox = p.get("bbox_pix", None)
            gt_bbox = info.get("bbox_xyxy", None)
            gt_bbox = _normalize_gt_bbox_xyxy_inclusive(gt_bbox, W=W, H=H)
            if pred_bbox is not None and gt_bbox is not None:
                iou = _iou_xyxy_inclusive(gt_bbox, pred_bbox)
                if binary_space_reward:
                    space_mat[i][j] = 1.0 if float(iou) >= float(iou_thr) else 0.0
                else:
                    space_mat[i][j] = float(max(0.0, min(1.0, iou)))

    matched_pairs, sum_time, sum_space = _best_time_space_matching(time_mat, space_mat)

    R_time = float(sum_time) / denom
    R_space = float(sum_space) / denom

    return {
        "M": int(M),
        "N": int(N),
        "denom": float(denom),
        "sum_time": float(sum_time),
        "sum_space": float(sum_space),
        "R_time": float(max(0.0, min(1.0, R_time))),
        "R_space": float(max(0.0, min(1.0, R_space))),
        "matched_pairs": matched_pairs,
    }


# ----------------------------
# Format reward (multi-entity)
# ----------------------------

def format_reward(response: str) -> float:
    """
    Format reward in [0,1], averaged over 5 checks:

      s1) Strict single "<think>...</think><answer>...</answer>"
      s2) <think> exists and non-empty
      s3) FIRST <answer> parses into a JSON list (len>=1) or dict->list fallback
      s4) slice validity fraction over items
      s5) bbox validity fraction over items (0..1000 + order)

    NOTE:
      - independent of strictness for s3/s4/s5
      - smoother for multi-entity
    """
    s1 = s2 = s3 = s4 = s5 = 0.0

    if _is_strict_single_think_answer(response):
        s1 = 1.0

    think_payload = _extract_think_payload_first(response)
    if think_payload is not None and len(think_payload.strip()) > 0:
        s2 = 1.0

    ans_payload = _extract_answer_payload_first(response)
    if ans_payload is None:
        return float((s1 + s2 + s3 + s4 + s5) / 5.0)

    items = _parse_entity_list_from_answer(ans_payload)
    if items is None or not isinstance(items, list) or len(items) == 0:
        return float((s1 + s2 + s3 + s4 + s5) / 5.0)

    s3 = 1.0

    valid_slice = 0
    valid_bbox = 0
    total = 0
    for it in items:
        total += 1
        if not isinstance(it, dict):
            continue

        try:
            _ = int(it.get("slice", None))
            valid_slice += 1
        except Exception:
            pass

        bbox = it.get("bbox_2d", None)
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                vals = [int(round(float(v))) for v in bbox]
                x1, y1, x2, y2 = vals
                if (
                    0 <= x1 <= 1000 and 0 <= x2 <= 1000 and
                    0 <= y1 <= 1000 and 0 <= y2 <= 1000 and
                    x1 < x2 and y1 < y2
                ):
                    valid_bbox += 1
            except Exception:
                pass

    if total > 0:
        s4 = float(valid_slice) / float(total)
        s5 = float(valid_bbox) / float(total)

    return float((s1 + s2 + s3 + s4 + s5) / 5.0)


# ----------------------------
# Batch interface
# ----------------------------

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    w_format: float = 0.2,
    w_time: float = 0.3,
    w_space: float = 0.5,
    iou_thr: float = 0.5,
    binary_space_reward: bool = False,
) -> List[Dict[str, float]]:
    """
    Veason-like overall:
      overall = w_format*R_format + w_time*R_time + w_space*R_space

    No explicit miss/fp terms; they are implicitly handled by denom=max(N_pred,N_gt).
    """
    scores: List[Dict[str, float]] = []
    debug_every = int(os.getenv("DEBUG_REWARD_EVERY", "0") or "0")

    for idx, inp in enumerate(reward_inputs):
        response_raw = inp.get("response", "")
        gt_raw = inp.get("ground_truth", {})

        response = _normalize_tag_spacing(str(response_raw))

        rf = format_reward(response)
        met = _veason_metrics(
            response,
            gt_raw,
            iou_thr=iou_thr,
            binary_space_reward=binary_space_reward,
        )

        rt = float(met.get("R_time", 0.0))
        rs = float(met.get("R_space", 0.0))

        overall = float(w_format) * float(rf) + float(w_time) * float(rt) + float(w_space) * float(rs)
        overall = float(max(0.0, min(1.0, overall)))

        if debug_every > 0 and (idx % debug_every == 0):
            ans = _extract_answer_payload_first(response)
            print(
                "[DEBUG reward] idx=", idx,
                "M=", met.get("M"), "N=", met.get("N"), "denom=", met.get("denom"),
                "format=", rf, "R_time=", rt, "R_space=", rs, "overall=", overall,
                "sum_time=", met.get("sum_time"), "sum_space=", met.get("sum_space"),
                "\n  response_head=", response[:200].replace("\n", "\\n"),
                "\n  answer_payload_head=", (ans[:200].replace("\n", "\\n") if isinstance(ans, str) else None),
            )

        scores.append(
            {
                "overall": float(overall),
                "format": float(rf),
                "time": float(rt),   # Veason-style normalized sum_time / max(N,M)
                "space": float(rs),  # Veason-style normalized sum_iou / max(N,M)
                "M": float(met.get("M", 0)),
                "N": float(met.get("N", 0)),
                "denom": float(met.get("denom", 1)),
            }
        )

    return scores


if __name__ == "__main__":
    # sanity check (toy)
    reward_inputs = [
            {
                "response": (
                    "<think>I'll locate two regions.</think>"
                    "<answer>["
                    "{\"slice\":131,\"bbox_2d\":[650,430,900,900]},"
                    "{\"slice\":67,\"bbox_2d\":[150,350,450,900]}"
                    "]</answer>"
                ),
                "ground_truth": {
                    "canon_size_xy": [512, 512],
                    "hit_slices_info_by_entity": {
                        "0": {
                            "131": {"bbox_xyxy": [325, 220, 461, 461], "area": 18342},
                            "119": {"bbox_xyxy": [345, 227, 459, 433], "area": 15600},
                        },
                        "1": {
                            "67": {"bbox_xyxy": [87, 199, 217, 416], "area": 20844},
                            "71": {"bbox_xyxy": [85, 198, 202, 397], "area": 17839},
                        },
                    },
                },
            },
            {
                # extra prediction (FP): 3 preds but only 2 GT entities
                "response": (
                    "<answer>["
                    "{\"slice\":131,\"bbox_2d\":[650,430,900,900]},"
                    "{\"slice\":67,\"bbox_2d\":[150,350,450,900]},"
                    "{\"slice\":999,\"bbox_2d\":[0,0,1000,1000]}"
                    "]</answer>"
                ),
                "ground_truth": {
                    "canon_size_xy": [512, 512],
                    "hit_slices_info_by_entity": {
                        "0": {"131": {"bbox_xyxy": [325, 220, 461, 461], "area": 18342}},
                        "1": {"67": {"bbox_xyxy": [87, 199, 217, 416], "area": 20844}},
                    },
                },
            },
            {
                # miss one entity: only predicts slice for entity 0
                "response": "<answer>[{\"slice\":131,\"bbox_2d\":[636,430,902,902]}]</answer>",
                "ground_truth": {
                    "canon_size_xy": [512, 512],
                    "hit_slices_info_by_entity": {
                        "0": {"131": {"bbox_xyxy": [325, 220, 461, 461], "area": 18342}},
                        "1": {"67": {"bbox_xyxy": [87, 199, 217, 416], "area": 20844}},
                    },
                },
            },
        ]

    print(compute_score(reward_inputs, iou_thr=0.5, binary_space_reward=True))
