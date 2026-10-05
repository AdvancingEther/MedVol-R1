#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

# 统一坐标到 512x512
CANON_W = 512
CANON_H = 512

PRINT_EVERY = 200


def extract_answer_text(s: str) -> str:
    m = ANSWER_RE.search(s or "")
    return m.group(1).strip() if m else ""


def safe_load_list_of_dicts(s: str) -> Optional[List[Dict[str, Any]]]:
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
            return obj
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
            return obj
    except Exception:
        pass
    return None


def parse_pred_from_predict_field(predict_str: str) -> Tuple[Optional[int], Optional[List[int]], str]:
    """
    predict: <answer>[{"frame_index":46,"bbox_2d":[643,670,717,728]}]</answer>
    Return: (frame_index, bbox_2d_0_1000, reason)
    """
    ans = extract_answer_text(predict_str)
    if not ans:
        return None, None, "no_answer_tag"

    lst = safe_load_list_of_dicts(ans)
    if not lst or len(lst) == 0:
        return None, None, "answer_parse_fail"

    item0 = lst[0]
    if not isinstance(item0, dict):
        return None, None, "answer_not_dict"

    frame = item0.get("frame_index", None)
    bbox = item0.get("bbox_2d", None)

    try:
        frame = int(frame)
    except Exception:
        return None, None, "frame_parse_fail"

    if not (isinstance(bbox, list) and len(bbox) == 4):
        return frame, None, "bbox_missing_or_bad"

    try:
        bbox_int = [int(round(float(v))) for v in bbox]
    except Exception:
        return frame, None, "bbox_parse_fail"

    return frame, bbox_int, "ok"


def load_text_info_json(sample_dir: Path) -> Dict[str, Any]:
    """
    Prefer text_info.json, fallback to text.json.
    """
    p1 = sample_dir / "text_info.json"
    p2 = sample_dir / "text.json"
    if p1.exists():
        with open(p1, "r", encoding="utf-8") as f:
            return json.load(f)
    if p2.exists():
        with open(p2, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"Missing text_info.json/text.json in {sample_dir}")


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def qwen_0_1000_to_512_xyxy_inclusive(b: List[int], W: int = CANON_W, H: int = CANON_H) -> Optional[List[int]]:
    """
    将 Qwen 输出的 0..1000 坐标映射到 0..(W-1)/(H-1)
    输出为 inclusive xyxy 像素框.
    """
    if not (isinstance(b, list) and len(b) == 4):
        return None

    x1, y1, x2, y2 = b

    # clamp 到 [0,1000]
    x1 = clamp(int(x1), 0, 1000)
    x2 = clamp(int(x2), 0, 1000)
    y1 = clamp(int(y1), 0, 1000)
    y2 = clamp(int(y2), 0, 1000)

    def map_x(xn: int) -> int:
        return int(round((xn / 1000.0) * (W - 1)))

    def map_y(yn: int) -> int:
        return int(round((yn / 1000.0) * (H - 1)))

    x1p, x2p = map_x(x1), map_x(x2)
    y1p, y2p = map_y(y1), map_y(y2)

    x1p = clamp(x1p, 0, W - 1)
    x2p = clamp(x2p, 0, W - 1)
    y1p = clamp(y1p, 0, H - 1)
    y2p = clamp(y2p, 0, H - 1)

    if x2p < x1p:
        x1p, x2p = x2p, x1p
    if y2p < y1p:
        y1p, y2p = y2p, y1p

    return [x1p, y1p, x2p, y2p]


def iou_xyxy_inclusive(a: Optional[List[int]], b: Optional[List[int]]) -> float:
    """
    IoU for inclusive pixel boxes [x1,y1,x2,y2]
    """
    if a is None or b is None:
        return 0.0
    if not (len(a) == 4 and len(b) == 4):
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


def main():
    inference_jsonl_results = os.environ.get('MEDVOL_EVAL_SLICE_HIT_BY_FRAME_INFERENCE_JSONL_RESULTS', 'outputs/grounding/ct_train_preds_filtered_for_seg.jsonl')
    data_info_json = os.environ.get('MEDVOL_EVAL_SLICE_HIT_BY_FRAME_DATA_INFO_JSON', 'data/external/LLM4SAM/segx/Data/train/ct_grounding_data_for_seg_filtered.json')
    data_info_root = os.environ.get('MEDVOL_EVAL_SLICE_HIT_BY_FRAME_DATA_INFO_ROOT', 'data/external/LLM4SAM/3d_data_downsample_for_qwen/ct_data/train')

    out_jsonl = inference_jsonl_results.replace(".jsonl", "_eval_512.jsonl")

    # 读 inference jsonl
    infer_results = []
    with open(inference_jsonl_results, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                infer_results.append(json.loads(line))

    # 读 data_info json
    with open(data_info_json, "r", encoding="utf-8") as f:
        data_info = json.load(f)

    n = min(len(infer_results), len(data_info))
    if len(infer_results) != len(data_info):
        print(f"[Warn] length mismatch: infer={len(infer_results)} data={len(data_info)} -> use n={n}")

    valid_cnt = 0
    parse_fail_cnt = 0
    hit_cnt = 0

    sum_iou_all = 0.0            # 非命中/无GT bbox 记 0
    sum_iou_hit = 0.0            # 只统计 hit 且 gt bbox 非空
    hit_with_bbox_cnt = 0

    with open(out_jsonl, "w", encoding="utf-8") as wf:
        for i in range(n):
            infer_item = infer_results[i]
            data_item = data_info[i]

            case_id = data_item.get("id", f"idx_{i}")
            sample_dir = Path(data_info_root) / case_id

            pred_frame, pred_bbox_0_1000, parse_reason = parse_pred_from_predict_field(infer_item.get("predict", ""))

            # 读 text_info.json/text.json
            meta_err = None
            key_slices_info = {}
            z_span = None
            center = None

            try:
                meta = load_text_info_json(sample_dir)
                ms = meta.get("mask_stats", {})
                key_slices_info = ms.get("key_slices_info", {}) or {}
                z_span = ms.get("z_span", None)
                center = ms.get("max_area_slice_index", ms.get("gt_center_slice", None))
            except Exception as e:
                meta_err = str(e)

            # 计算 hit / iou
            hit = 0
            gt_bbox_512 = None
            pred_bbox_512 = None
            iou = 0.0

            if pred_frame is None or pred_bbox_0_1000 is None:
                parse_fail_cnt += 1
            else:
                valid_cnt += 1
                hit = 1 if (isinstance(key_slices_info, dict) and str(pred_frame) in key_slices_info) else 0
                if hit:
                    hit_cnt += 1
                    gt_bbox_512 = key_slices_info.get(str(pred_frame), {}).get("bbox_xyxy", None)

                pred_bbox_512 = qwen_0_1000_to_512_xyxy_inclusive(pred_bbox_0_1000, W=CANON_W, H=CANON_H)

                # IoU：只有命中且 GT bbox 非空时才会 >0，否则记 0
                if hit and gt_bbox_512 is not None and pred_bbox_512 is not None:
                    iou = iou_xyxy_inclusive(gt_bbox_512, pred_bbox_512)
                    sum_iou_hit += iou
                    hit_with_bbox_cnt += 1

                sum_iou_all += iou

            row = {
                "case_id": case_id,
                "parse_reason": parse_reason,
                "meta_err": meta_err,

                "pred_frame_index": pred_frame,
                "pred_bbox_qwen_0_1000": pred_bbox_0_1000,
                "pred_bbox_512_xyxy": pred_bbox_512,

                "hit_in_key_slices": int(hit),
                "gt_bbox_512_xyxy": gt_bbox_512,
                "iou_512": float(iou),

                # context
                "z_span": z_span,
                "gt_center_slice": center,
                "num_key_slices": len(key_slices_info) if isinstance(key_slices_info, dict) else None,
                "canon_size_xy": [CANON_W, CANON_H],
            }
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (i + 1) % PRINT_EVERY == 0:
                mean_iou_all = sum_iou_all / max(1, valid_cnt)
                hit_rate = hit_cnt / max(1, valid_cnt)
                print(f"[{i+1}/{n}] valid={valid_cnt} hit={hit_cnt} hit_rate={hit_rate:.4f} mean_iou_all={mean_iou_all:.4f}")

    hit_rate = hit_cnt / valid_cnt if valid_cnt > 0 else 0.0
    mean_iou_all = sum_iou_all / valid_cnt if valid_cnt > 0 else 0.0
    mean_iou_hit = (sum_iou_hit / hit_with_bbox_cnt) if hit_with_bbox_cnt > 0 else 0.0

    print("\n========== Summary (512x512) ==========")
    print(f"Total pairs used: {n}")
    print(f"Valid preds: {valid_cnt} | Parse fail: {parse_fail_cnt}")
    print(f"Hit count: {hit_cnt} | Hit rate: {hit_rate:.4f}")
    print(f"Mean IoU (all valid, non-hit as 0): {mean_iou_all:.4f}")
    print(f"Mean IoU (hit & has_gt_bbox only): {mean_iou_hit:.4f} (n={hit_with_bbox_cnt})")
    print(f"Saved details to: {out_jsonl}")


if __name__ == "__main__":
    main()
