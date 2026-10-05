# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    # Hungarian
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


# Metadata (match EasyR1 / veRL style)
REWARD_NAME = "vision_reasoner"
REWARD_TYPE = "batch"


# ----------------------------
# Core rewards (single sample)
# ----------------------------

def format_reward(response: str) -> float:
    """
    Format reward = thinking_format + segmentation_json_format
    Range approx: [0, 3]
      - thinking_format: {0,1}  require fullmatch <think>...</think><answer>...</answer>
      - segmentation_json_format: [0,2]  bbox_2d (+1) + point_2d (+1), averaged within list
    """
    # strict format for think + answer tags
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, response, re.DOTALL)
    thinking_format = 1.0 if match else 0.0

    def segmentation_format(resp: str) -> float:
        seg_reward = 0.0
        try:
            json_match = re.search(r"<answer>\s*(.*?)\s*</answer>", resp, re.DOTALL)
            if not json_match:
                return 0.0
            data = json.loads(json_match.group(1))
            if not isinstance(data, list) or len(data) == 0:
                return 0.0

            data_cnt = len(data)
            for item in data:
                cur = 0.0
                if isinstance(item, dict):
                    if "bbox_2d" in item:
                        bbox = item["bbox_2d"]
                        if isinstance(bbox, list) and len(bbox) == 4:
                            cur += 1.0
                    if "point_2d" in item:
                        pt = item["point_2d"]
                        if isinstance(pt, list) and len(pt) == 2:
                            cur += 1.0
                seg_reward += cur / data_cnt
        except Exception:
            return 0.0
        return float(seg_reward)

    return float(thinking_format + segmentation_format(response))


def _safe_json_loads(x: Any) -> Any:
    """ground_truth may be string or already a python object."""
    if isinstance(x, str):
        return json.loads(x)
    return x


def accuracy_reward(
    response: str,
    ground_truth: str,
    max_objects: int = 120,
    iou_thr: float = 0.5,
    l1_thr: float = 10.0,
    point_thr: float = 30.0,
) -> float:
    """
    Accuracy reward in [0,3] (normalized by max(len(pred), len(gt))).
    Uses Hungarian matching on 3 signals:
      - IoU > iou_thr
      - bbox L1 mean < l1_thr
      - point distance < point_thr AND pred point inside pred bbox
    """
    try:
        # parse gt
        gt_data = _safe_json_loads(ground_truth)
        if not isinstance(gt_data, list) or len(gt_data) == 0:
            return 0.0

        gt_bboxes = [item.get("bbox_2d", None) for item in gt_data if isinstance(item, dict)]
        gt_points = [item.get("point_2d", None) for item in gt_data if isinstance(item, dict)]
        gt_pairs = [(b, p) for b, p in zip(gt_bboxes, gt_points) if isinstance(b, list) and len(b) == 4 and isinstance(p, list) and len(p) == 2]
        if len(gt_pairs) == 0:
            return 0.0

        gt_bboxes = [b for b, _ in gt_pairs][:max_objects]
        gt_points = [p for _, p in gt_pairs][:max_objects]

        # parse pred from <answer>...</answer>
        json_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
        if not json_match:
            return 0.0

        pred_data = json.loads(json_match.group(1))
        if not isinstance(pred_data, list) or len(pred_data) == 0:
            return 0.0

        pred_bboxes = [item.get("bbox_2d", None) for item in pred_data if isinstance(item, dict)]
        pred_points = [item.get("point_2d", None) for item in pred_data if isinstance(item, dict)]
        pred_pairs = [(b, p) for b, p in zip(pred_bboxes, pred_points) if isinstance(b, list) and len(b) == 4 and isinstance(p, list) and len(p) == 2]
        if len(pred_pairs) == 0:
            return 0.0

        pred_bboxes = [b for b, _ in pred_pairs][:max_objects]
        pred_points = [p for _, p in pred_pairs][:max_objects]

        M, N = len(pred_bboxes), len(gt_bboxes)
        if M == 0 or N == 0:
            return 0.0

        pred_bboxes = np.asarray(pred_bboxes, dtype=np.float32)  # (M,4)
        pred_points = np.asarray(pred_points, dtype=np.float32)  # (M,2)
        gt_bboxes = np.asarray(gt_bboxes, dtype=np.float32)      # (N,4)
        gt_points = np.asarray(gt_points, dtype=np.float32)      # (N,2)

        iou_matrix = batch_iou(pred_bboxes, gt_bboxes)                 # (M,N)
        l1_matrix = batch_l1_distance(pred_bboxes, gt_bboxes)          # (M,N)
        points_dist_matrix = batch_points_distance(pred_points, gt_points)  # (M,N)
        points_in_box = batch_points_in_box(pred_points, pred_bboxes)       # (M,)

        iou_reward = (iou_matrix > iou_thr).astype(np.float32)
        bbox_l1_reward = (l1_matrix < l1_thr).astype(np.float32)
        point_reward = ((points_dist_matrix < point_thr) & (points_in_box[:, None])).astype(np.float32)

        # each pair max 3
        pair_reward = iou_reward + bbox_l1_reward + point_reward  # (M,N)
        cost_matrix = 3.0 - pair_reward

        if linear_sum_assignment is not None:
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            total = float(pair_reward[row_idx, col_idx].sum())
        else:
            # fallback: greedy match by best reward
            total = 0.0
            used_rows = set()
            used_cols = set()
            flat = [(pair_reward[i, j], i, j) for i in range(M) for j in range(N)]
            flat.sort(reverse=True, key=lambda x: x[0])
            for r, i, j in flat:
                if i in used_rows or j in used_cols:
                    continue
                used_rows.add(i)
                used_cols.add(j)
                total += float(r)
                if len(used_rows) == min(M, N):
                    break

        denom = float(max(M, N))
        return float(total / denom)

    except Exception:
        return 0.0


def non_repeat_reward(response: str) -> float:
    """
    Penalize exact repeated sentences split by '.'.
    If repeats >= 2 -> 0 else 1.
    """
    r = 1.0
    try:
        sentences = [s.strip() for s in response.split(".") if s.strip()]
        seen = set()
        repeats = 0
        for s in sentences:
            if s in seen:
                repeats += 1
            if repeats >= 2:
                r = 0.0
                break
            seen.add(s)
    except Exception:
        return 1.0
    return float(r)


# ----------------------------
# Batch interface (match math.py)
# ----------------------------

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
    non_repeat_weight: float = 1.0,
    max_objects: int = 120,
    iou_thr: float = 0.5,
    l1_thr: float = 10.0,
    point_thr: float = 30.0,
) -> List[Dict[str, float]]:
    """
    Match math.py interface:
      reward_inputs: [{"response": str, "ground_truth": str, ...}, ...]
    Return:
      [{"overall":..., "format":..., "accuracy":..., "non_repeat":...}, ...]
    """

    scores: List[Dict[str, float]] = []
    for inp in reward_inputs:
        # required keys
        response_raw = inp.get("response", "")
        gt_raw = inp.get("ground_truth", "")

        # handle qwen2.5vl-32b tag spacing like math.py
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", str(response_raw))

        # ground_truth: allow python object
        if isinstance(gt_raw, str):
            ground_truth = gt_raw
        else:
            # safe json dump for dict/list
            ground_truth = json.dumps(gt_raw, ensure_ascii=False)

        fmt = format_reward(response)
        acc = accuracy_reward(
            response,
            ground_truth,
            max_objects=max_objects,
            iou_thr=iou_thr,
            l1_thr=l1_thr,
            point_thr=point_thr,
        )
        nr = non_repeat_reward(response)

        # overall: keep math.py style mixing format/accuracy, then add non_repeat
        overall = (1.0 - float(format_weight)) * float(acc) + float(format_weight) * float(fmt) + float(non_repeat_weight) * float(nr)

        scores.append(
            {
                "overall": float(overall),
                "format": float(fmt),
                "accuracy": float(acc),
                "non_repeat": float(nr),
            }
        )

    return scores


# ----------------------------
# Vectorized helpers
# ----------------------------

def batch_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    # boxes1: (M,4), boxes2: (N,4)
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)  # (M,1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)  # (N,1)

    xA = np.maximum(x11, x21.T)  # (M,N)
    yA = np.maximum(y11, y21.T)
    xB = np.minimum(x12, x22.T)
    yB = np.minimum(y12, y22.T)

    inter = np.maximum(0.0, xB - xA + 1.0) * np.maximum(0.0, yB - yA + 1.0)
    area1 = (x12 - x11 + 1.0) * (y12 - y11 + 1.0)  # (M,1)
    area2 = (x22 - x21 + 1.0) * (y22 - y21 + 1.0)  # (N,1)
    union = area1 + area2.T - inter
    # avoid div0
    return inter / np.maximum(union, 1e-6)


def batch_l1_distance(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    # mean abs diff over 4 coords
    b1 = boxes1[:, None, :]  # (M,1,4)
    b2 = boxes2[None, :, :]  # (1,N,4)
    return np.mean(np.abs(b1 - b2), axis=2)  # (M,N)


def batch_points_distance(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    p1 = points1[:, None, :]  # (M,1,2)
    p2 = points2[None, :, :]  # (1,N,2)
    return np.sqrt(np.sum((p1 - p2) ** 2, axis=2))  # (M,N)


def batch_points_in_box(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    # points: (M,2), boxes: (M,4)
    x_ok = (points[:, 0] >= boxes[:, 0]) & (points[:, 0] <= boxes[:, 2])
    y_ok = (points[:, 1] >= boxes[:, 1]) & (points[:, 1] <= boxes[:, 3])
    return x_ok & y_ok


if __name__ == "__main__":
    # quick sanity test
    reward_inputs = [
        {
            "response": """
<think>ok</think>
<answer>
[{"bbox_2d":[10,100,398,423], "point_2d":[283,169]}]
</answer>
""",
            "ground_truth": """
[{"bbox_2d":[416,7,833,553], "point_2d":[648,249]}]
""",
        }
    ]
    print(compute_score(reward_inputs))
