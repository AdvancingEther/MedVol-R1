import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


# =========================
# User config (edit here)
# =========================
pred_result_json_path = os.environ.get('MEDVOL_EVAL_SLICE_HIT_FRAME16_VIS_MULTI_PRED_RESULT_JSON_PATH', 'outputs/grounding/output_folder/ct_train_16_frames_multi_filtered_final.jsonl')
case_info_json = os.environ.get('MEDVOL_EVAL_SLICE_HIT_FRAME16_VIS_MULTI_CASE_INFO_JSON', 'data/grounding/ct_data_ori_npy_multi_finding/ct_train_16_frames_multi_filtered_final.json')
case_info_root_path = os.environ.get('MEDVOL_EVAL_SLICE_HIT_FRAME16_VIS_MULTI_CASE_INFO_ROOT_PATH', 'data/grounding/ct_data_ori_npy_multi_finding/train_16_frames_guarantee')
OUTPUT_DIR = os.environ.get('MEDVOL_EVAL_SLICE_HIT_FRAME16_VIS_MULTI_OUTPUT_DIR', 'outputs/grounding/vis_output_dir')

# hit_slices_info["..."]["bbox_xyxy"] 是 512x512 像素坐标
IMG_W, IMG_H = 512, 512

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

# =========================
# 可视化相关开关
# =========================
SAVE_VIZ = True
VIZ_SUBDIR_NAME = "viz_bbox_multi"
ONLY_SAVE_PARSE_OK = False          # True: 只保存 parse_reason == "ok" 的 pred 可视化
DRAW_LINE_WIDTH = 3
DRAW_LABEL = True
SAVE_FAIL_PLACEHOLDER = False       # slice png找不到时是否保存占位图

# GT 画图策略：每个 entity 从其 hit_slices_info 里选一个“代表性的 GT slice”
GT_PICK_STRATEGY = "max_area"       # {"max_area", "min_z", "max_z", "middle"}


# =========================
# I/O helpers
# =========================
def safe_load_json(path: str) -> Any:
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


# =========================
# Pred parsing (multi-entity)
# =========================
def extract_pred_list(pred_text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Return:
      pred_list: list of dicts, each has:
        - pred_idx
        - entity (optional)
        - slice (int or None)
        - bbox_norm1000 (list[int] or None)
        - reason (str)
      parse_reason_overall: str
    """
    if pred_text is None:
        return [], "predict_is_none"

    m = ANSWER_RE.search(pred_text)
    if not m:
        return [], "no_answer_tag"

    inside = m.group(1).strip()
    try:
        obj = json.loads(inside)
    except Exception as e:
        return [], f"answer_json_parse_fail: {type(e).__name__}: {e}"

    if not isinstance(obj, list):
        return [], "answer_not_list"

    pred_list: List[Dict[str, Any]] = []
    any_ok = False
    for k, d in enumerate(obj):
        item: Dict[str, Any] = {
            "pred_idx": k,
            "entity": None,
            "slice": None,
            "bbox_norm1000": None,
            "reason": "ok",
        }

        if not isinstance(d, dict):
            item["reason"] = "pred_item_not_dict"
            pred_list.append(item)
            continue

        if "entity" in d:
            try:
                item["entity"] = int(d["entity"])
            except Exception:
                # entity 不影响匹配，只记录
                item["entity"] = d.get("entity", None)

        if "slice" not in d or "bbox_2d" not in d:
            item["reason"] = "missing_slice_or_bbox_2d"
            pred_list.append(item)
            continue

        try:
            s = int(d["slice"])
            item["slice"] = s
        except Exception:
            item["reason"] = "slice_not_int"
            pred_list.append(item)
            continue

        bbox = d["bbox_2d"]
        if (not isinstance(bbox, list)) or len(bbox) != 4:
            item["reason"] = "bbox_2d_not_len4"
            pred_list.append(item)
            continue

        try:
            bbox_int = [int(round(float(x))) for x in bbox]
            item["bbox_norm1000"] = bbox_int
        except Exception:
            item["reason"] = "bbox_2d_not_numeric"
            pred_list.append(item)
            continue

        x1, y1, x2, y2 = bbox_int
        if not (x1 < x2 and y1 < y2):
            # 仍然允许后面 clamp/swap，但标记一下
            item["reason"] = "bbox_invalid_order"

        any_ok = True
        pred_list.append(item)

    if not any_ok and len(pred_list) > 0:
        # 有条目但都不合格
        return pred_list, "no_valid_pred_items"
    return pred_list, "ok"


# =========================
# BBox utils
# =========================
def norm1000_to_512_bbox(b: List[int], w: int = IMG_W, h: int = IMG_H) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    px1 = x1 / 1000.0 * w
    py1 = y1 / 1000.0 * h
    px2 = x2 / 1000.0 * w
    py2 = y2 / 1000.0 * h
    return px1, py1, px2, py2


def clamp_bbox_xyxy(
    b: Tuple[float, float, float, float],
    w: int = IMG_W,
    h: int = IMG_H
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    x1 = max(0.0, min(float(w), float(x1)))
    x2 = max(0.0, min(float(w), float(x2)))
    y1 = max(0.0, min(float(h), float(y1)))
    y2 = max(0.0, min(float(h), float(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


# =========================
# Viz helpers
# =========================
def try_load_font(size: int = 14) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_bbox(
    img: Image.Image,
    bbox_xyxy: Tuple[float, float, float, float],
    color: Tuple[int, int, int],
    label: str,
    line_width: int = 3,
    font: Optional[ImageFont.ImageFont] = None,
) -> None:
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox_xyxy
    x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    for k in range(line_width):
        draw.rectangle([x1i - k, y1i - k, x2i + k, y2i + k], outline=color)

    if DRAW_LABEL and label:
        if font is None:
            font = ImageFont.load_default()
        tx, ty = x1i, max(0, y1i - 18)
        try:
            tb = draw.textbbox((tx, ty), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = 80, 14
        pad = 2
        draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad], fill=(0, 0, 0))
        draw.text((tx + pad, ty + pad), label, fill=color, font=font)


def find_slice_image(case_dir: Path, slice_idx: int) -> Optional[Path]:
    candidates = [
        case_dir / f"slice_{slice_idx}.png",
        case_dir / f"slice_{slice_idx:03d}.png",
        case_dir / f"slice_{slice_idx:04d}.png",
        case_dir / f"slice_{slice_idx:05d}.png",
        case_dir / f"slice_{slice_idx:06d}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    for p in sorted(case_dir.glob("slice_*.png")):
        m = re.search(r"slice_(\d+)\.png$", p.name)
        if m and int(m.group(1)) == int(slice_idx):
            return p
    return None


def save_viz_image(
    img_path: Optional[Path],
    out_path: Path,
    pred_bbox_px: Optional[Tuple[float, float, float, float]],
    gt_bbox_px: Optional[Tuple[float, float, float, float]],
    meta_text: str,
    pred_label: str = "P",
    gt_label: str = "GT",
) -> bool:
    if img_path is None or (not img_path.exists()):
        if not SAVE_FAIL_PLACEHOLDER:
            return False
        img = Image.new("RGB", (IMG_W, IMG_H), (0, 0, 0))
    else:
        img = Image.open(img_path).convert("RGB")

    font = try_load_font(14)

    if DRAW_LABEL and meta_text:
        draw = ImageDraw.Draw(img)
        pad = 3
        x0, y0 = 2, 2
        try:
            tb = draw.textbbox((x0, y0), meta_text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = 260, 14
        draw.rectangle([x0, y0, x0 + tw + 2 * pad, y0 + th + 2 * pad], fill=(0, 0, 0))
        draw.text((x0 + pad, y0 + pad), meta_text, fill=(255, 255, 255), font=font)

    if gt_bbox_px is not None:
        draw_bbox(img, gt_bbox_px, color=(0, 255, 0), label=gt_label, line_width=DRAW_LINE_WIDTH, font=font)
    if pred_bbox_px is not None:
        draw_bbox(img, pred_bbox_px, color=(255, 0, 0), label=pred_label, line_width=DRAW_LINE_WIDTH, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return True


# =========================
# GT helpers (per-entity)
# =========================
def pick_gt_slice_from_hit_slices_info(
    hit_slices_info: Dict[str, Any],
    strategy: str = "max_area",
) -> Tuple[Optional[int], Optional[List[float]]]:
    if not isinstance(hit_slices_info, dict) or len(hit_slices_info) == 0:
        return None, None

    items = []
    for k, v in hit_slices_info.items():
        try:
            s = int(k)
        except Exception:
            continue
        if not isinstance(v, dict):
            continue
        b = v.get("bbox_xyxy", None)
        if isinstance(b, list) and len(b) == 4:
            x1, y1, x2, y2 = map(float, b)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            items.append((s, [x1, y1, x2, y2], area))

    if len(items) == 0:
        return None, None

    items_sorted = sorted(items, key=lambda x: x[0])
    if strategy == "min_z":
        s, b, _ = items_sorted[0]
        return s, b
    if strategy == "max_z":
        s, b, _ = items_sorted[-1]
        return s, b
    if strategy == "middle":
        mid = items_sorted[len(items_sorted) // 2]
        return mid[0], mid[1]

    best = max(items, key=lambda x: x[2])
    return best[0], best[1]


def safe_reason_token(s: str, maxlen: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))[:maxlen]


# =========================
# Matching (prediction-centric, slice hard constraint, 1-1)
# =========================
def build_iou_candidates(
    pred_items: List[Dict[str, Any]],
    gt_entities: List[Dict[str, Any]],
) -> Tuple[List[List[Optional[float]]], List[List[Optional[Tuple[float, float, float, float]]]]]:
    """
    Returns:
      iou_mat[p][g] = IoU if gt entity g has bbox at pred_slice, else None
      gt_bbox_on_pred[p][g] = gt bbox px on pred_slice if exists else None
    """
    iou_mat: List[List[Optional[float]]] = []
    gt_bbox_on_pred: List[List[Optional[Tuple[float, float, float, float]]]] = []

    for p in pred_items:
        row_iou: List[Optional[float]] = []
        row_gtbbox: List[Optional[Tuple[float, float, float, float]]] = []
        s = p.get("slice", None)
        bbox_norm = p.get("bbox_norm1000", None)

        if s is None or bbox_norm is None:
            # 全 None
            for _ in gt_entities:
                row_iou.append(None)
                row_gtbbox.append(None)
            iou_mat.append(row_iou)
            gt_bbox_on_pred.append(row_gtbbox)
            continue

        pred_bbox_px = clamp_bbox_xyxy(norm1000_to_512_bbox(bbox_norm), IMG_W, IMG_H)

        for g in gt_entities:
            hsi = g.get("hit_slices_info", {})
            if not isinstance(hsi, dict) or str(s) not in hsi:
                row_iou.append(None)
                row_gtbbox.append(None)
                continue
            v = hsi.get(str(s), {})
            gb = v.get("bbox_xyxy", None) if isinstance(v, dict) else None
            if not (isinstance(gb, list) and len(gb) == 4):
                row_iou.append(None)
                row_gtbbox.append(None)
                continue
            gt_bbox_px = clamp_bbox_xyxy((float(gb[0]), float(gb[1]), float(gb[2]), float(gb[3])), IMG_W, IMG_H)
            iou = iou_xyxy(pred_bbox_px, gt_bbox_px)
            row_iou.append(iou)          # 注意：允许 iou=0，仍然可视作“命中slice但框不准”
            row_gtbbox.append(gt_bbox_px)

        iou_mat.append(row_iou)
        gt_bbox_on_pred.append(row_gtbbox)

    return iou_mat, gt_bbox_on_pred


def solve_best_matching(iou_mat: List[List[Optional[float]]]) -> Tuple[List[Optional[int]], float, int]:
    """
    Find best 1-1 assignment from preds to gts (or unmatched),
    with hard constraint already encoded by None (unavailable).
    Objective:
      1) maximize total_iou
      2) if tie, maximize matched_count

    Returns:
      match_pred_to_gt: list len P, each is gt index or None
      best_total_iou
      best_matched_count
    """
    P = len(iou_mat)
    G = len(iou_mat[0]) if P > 0 else 0

    best_total_iou = -1.0
    best_matched = -1
    best_assign: List[Optional[int]] = [None] * P

    used = [False] * G
    cur_assign: List[Optional[int]] = [None] * P

    def dfs(pi: int, cur_iou: float, cur_m: int) -> None:
        nonlocal best_total_iou, best_matched, best_assign

        if pi == P:
            if (cur_iou > best_total_iou) or (abs(cur_iou - best_total_iou) < 1e-12 and cur_m > best_matched):
                best_total_iou = cur_iou
                best_matched = cur_m
                best_assign = list(cur_assign)
            return

        # option 0: leave unmatched
        cur_assign[pi] = None
        dfs(pi + 1, cur_iou, cur_m)

        # option 1: match to any available gt not used
        for gj in range(G):
            if used[gj]:
                continue
            v = iou_mat[pi][gj]
            if v is None:
                continue  # slice hard constraint
            used[gj] = True
            cur_assign[pi] = gj
            dfs(pi + 1, cur_iou + float(v), cur_m + 1)
            used[gj] = False
            cur_assign[pi] = None

    dfs(0, 0.0, 0)
    if best_total_iou < 0:
        best_total_iou = 0.0
        best_matched = 0
        best_assign = [None] * P
    return best_assign, best_total_iou, best_matched


# =========================
# Main
# =========================
def main():
    case_info = safe_load_json(case_info_json)
    case_ids = [x["id"] for x in case_info]
    preds = read_jsonl(pred_result_json_path)

    n = min(len(case_ids), len(preds))
    if len(case_ids) != len(preds):
        print(f"[WARN] length mismatch: case_info={len(case_ids)} vs preds={len(preds)}. Using first {n} aligned by order.")

    root = Path(case_info_root_path)
    out_dir = Path(OUTPUT_DIR)
    viz_dir = out_dir / VIZ_SUBDIR_NAME
    viz_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    # global accumulators (entity-centric)
    total_gt_entities = 0
    matched_gt_entities = 0
    sum_iou_matched = 0.0

    # pred-centric accumulators (optional)
    total_pred_items = 0
    matched_pred_items = 0

    for i in tqdm(range(n), desc="Evaluating+Vis (multi)"):
        case_id = case_ids[i]
        pred_item = preds[i]
        pred_text = pred_item.get("predict", "")

        pred_list, parse_reason = extract_pred_list(pred_text)

        case_dir = root / case_id
        text_json_path = case_dir / "text.json"

        row: Dict[str, Any] = {
            "index": i,
            "case_id": case_id,
            "parse_reason": parse_reason,
            "text_json_exists": text_json_path.exists(),

            "num_pred_items": len(pred_list),
            "num_gt_entities": 0,

            # per-case metrics
            "matched_gt_entities": 0,
            "matched_pred_items": 0,
            "entity_hit_rate": 0.0,
            "mean_iou_all_entities": 0.0,
            "mean_iou_on_matched": 0.0,

            # details (json strings for csv)
            "pred_items": None,
            "gt_entities": None,
            "match_pred_to_gt": None,
            "pred_ious": None,

            # viz outputs
            "viz_pred_pngs": None,
            "viz_gt_pngs": None,
        }

        if not text_json_path.exists():
            # still save pred parsing info
            row["pred_items"] = pred_list
            results.append(row)
            continue

        try:
            meta = safe_load_json(str(text_json_path))
        except Exception as e:
            row["parse_reason"] = f"text_json_load_fail: {type(e).__name__}: {e}"
            row["pred_items"] = pred_list
            results.append(row)
            continue

        # --------
        # Build GT entities from mask_stats list
        # --------
        mask_stats = meta.get("mask_stats", None)
        gt_entities: List[Dict[str, Any]] = []
        if isinstance(mask_stats, list):
            for ms in mask_stats:
                if not isinstance(ms, dict):
                    continue
                ent_idx = ms.get("entity_index", None)
                try:
                    ent_idx_int = int(ent_idx) if ent_idx is not None else None
                except Exception:
                    ent_idx_int = None

                hsi = ms.get("hit_slices_info", {}) or {}
                if not isinstance(hsi, dict):
                    hsi = {}

                gt_slice, gt_bbox_list = pick_gt_slice_from_hit_slices_info(hsi, strategy=GT_PICK_STRATEGY)

                gt_entities.append({
                    "entity_index": ent_idx_int,
                    "entity_value": ms.get("entity_value", None),
                    "num_hit_slices": len(hsi) if isinstance(hsi, dict) else 0,
                    "hit_slices_info": hsi,  # keep for matching (will be json-dumped later)
                    "repr_slice": gt_slice,
                    "repr_bbox_xyxy_512": gt_bbox_list,
                })
        else:
            # unexpected schema
            row["parse_reason"] = f"{row['parse_reason']}|mask_stats_not_list"
            gt_entities = []

        G = len(gt_entities)
        P = len(pred_list)

        row["num_gt_entities"] = G
        total_gt_entities += G
        total_pred_items += P

        # if no gt entities, still dump info
        if G == 0:
            row["pred_items"] = pred_list
            row["gt_entities"] = gt_entities
            results.append(row)
            continue

        # --------
        # Prepare iou matrix (pred slice hard constraint)
        # --------
        iou_mat, gt_bbox_on_pred = build_iou_candidates(pred_list, gt_entities)
        match_pred_to_gt, best_total_iou, best_matched = solve_best_matching(iou_mat)

        # compute per-pred iou and per-gt matched
        pred_ious: List[float] = []
        gt_matched_flags = [0] * G
        for pi in range(P):
            gj = match_pred_to_gt[pi]
            if gj is None:
                pred_ious.append(0.0)
            else:
                v = iou_mat[pi][gj]
                pred_ious.append(float(v) if v is not None else 0.0)
                gt_matched_flags[gj] = 1

        matched_gt = int(sum(gt_matched_flags))
        matched_pred = int(sum(1 for x in match_pred_to_gt if x is not None))

        matched_gt_entities += matched_gt
        matched_pred_items += matched_pred
        sum_iou_matched += float(best_total_iou)

        row["matched_gt_entities"] = matched_gt
        row["matched_pred_items"] = matched_pred

        # entity-centric metrics (recommended)
        row["entity_hit_rate"] = (matched_gt / G) if G > 0 else 0.0
        row["mean_iou_all_entities"] = (best_total_iou / G) if G > 0 else 0.0
        row["mean_iou_on_matched"] = (best_total_iou / matched_gt) if matched_gt > 0 else 0.0

        # details
        # 为了避免 json 太大：pred_items/gt_entities 里 hit_slices_info 会很大，但你之前也会 dump details 到 json，OK。
        row["pred_items"] = pred_list
        row["gt_entities"] = gt_entities
        row["match_pred_to_gt"] = match_pred_to_gt
        row["pred_ious"] = pred_ious

        # --------
        # Visualization
        # --------
        viz_pred_pngs: List[str] = []
        viz_gt_pngs: List[str] = []

        if SAVE_VIZ:
            case_out_dir = viz_dir / case_id
            case_out_dir.mkdir(parents=True, exist_ok=True)

            # (A) Pred images: one per pred item (valid slice+bbox)
            for p in pred_list:
                if p.get("slice", None) is None or p.get("bbox_norm1000", None) is None:
                    continue
                if ONLY_SAVE_PARSE_OK and parse_reason != "ok":
                    continue

                pi = int(p.get("pred_idx", 0))
                s = int(p["slice"])
                pred_bbox_px = clamp_bbox_xyxy(norm1000_to_512_bbox(p["bbox_norm1000"]), IMG_W, IMG_H)

                pred_img_path = find_slice_image(case_dir, s)

                gj = match_pred_to_gt[pi] if pi < len(match_pred_to_gt) else None
                iou_val = pred_ious[pi] if pi < len(pred_ious) else 0.0

                gt_bbox_px = None
                gt_label = "GT"
                if gj is not None:
                    gt_bbox_px = gt_bbox_on_pred[pi][gj] if (pi < len(gt_bbox_on_pred) and gj < len(gt_bbox_on_pred[pi])) else None
                    ent_idx = gt_entities[gj].get("entity_index", None)
                    gt_label = f"GT{ent_idx if ent_idx is not None else gj}"

                safe_r = safe_reason_token(p.get("reason", "ok"))
                out_png = case_out_dir / f"pred{pi}_slice{s}_m{gj if gj is not None else 'None'}_iou{iou_val:.4f}_{safe_r}.png"
                meta_text = f"[PRED] {case_id} p{pi} slice={s} match={gj} iou={iou_val:.4f}"

                ok = save_viz_image(
                    img_path=pred_img_path if pred_img_path is not None else None,
                    out_path=out_png,
                    pred_bbox_px=pred_bbox_px,
                    gt_bbox_px=gt_bbox_px,
                    meta_text=meta_text,
                    pred_label=f"P{pi}",
                    gt_label=gt_label,
                )
                if ok:
                    viz_pred_pngs.append(str(out_png))

            # (B) GT images: one per GT entity (representative slice)
            for gj, g in enumerate(gt_entities):
                gt_slice = g.get("repr_slice", None)
                gt_bbox_list = g.get("repr_bbox_xyxy_512", None)
                if gt_slice is None or gt_bbox_list is None or (not isinstance(gt_bbox_list, list)) or len(gt_bbox_list) != 4:
                    continue
                s = int(gt_slice)
                gt_bbox_px = clamp_bbox_xyxy((float(gt_bbox_list[0]), float(gt_bbox_list[1]), float(gt_bbox_list[2]), float(gt_bbox_list[3])), IMG_W, IMG_H)
                gt_img_path = find_slice_image(case_dir, s)

                ent_idx = g.get("entity_index", None)
                out_png = case_out_dir / f"gt_ent{ent_idx if ent_idx is not None else gj}_slice{s}_{GT_PICK_STRATEGY}.png"
                meta_text = f"[GT] {case_id} ent={ent_idx} slice={s} {GT_PICK_STRATEGY}"

                ok = save_viz_image(
                    img_path=gt_img_path if gt_img_path is not None else None,
                    out_path=out_png,
                    pred_bbox_px=None,
                    gt_bbox_px=gt_bbox_px,
                    meta_text=meta_text,
                    pred_label="",
                    gt_label=f"GT{ent_idx if ent_idx is not None else gj}",
                )
                if ok:
                    viz_gt_pngs.append(str(out_png))

        row["viz_pred_pngs"] = viz_pred_pngs
        row["viz_gt_pngs"] = viz_gt_pngs

        results.append(row)

    # --------
    # Global summary
    # --------
    denom_gt = total_gt_entities if total_gt_entities > 0 else 1
    denom_pred = total_pred_items if total_pred_items > 0 else 1
    denom_matched_gt = matched_gt_entities if matched_gt_entities > 0 else 1

    entity_hit_rate_global = matched_gt_entities / denom_gt
    mean_iou_all_entities_global = sum_iou_matched / denom_gt
    mean_iou_on_matched_global = sum_iou_matched / denom_matched_gt
    pred_match_rate_global = matched_pred_items / denom_pred

    print("=" * 80)
    print(f"Total cases used:                {n}")
    print(f"Total GT entities:               {total_gt_entities}")
    print(f"Matched GT entities:             {matched_gt_entities}")
    print(f"Entity hit rate (global):        {entity_hit_rate_global:.6f}")
    print(f"Mean IoU (all GT entities):      {mean_iou_all_entities_global:.6f}")
    print(f"Mean IoU (on matched entities):  {mean_iou_on_matched_global:.6f}")
    print(f"Total pred items:                {total_pred_items}")
    print(f"Matched pred items:              {matched_pred_items}")
    print(f"Pred match rate (global):        {pred_match_rate_global:.6f}")
    print(f"Viz dir:                         {viz_dir}")
    print("=" * 80)

    out_json = out_dir / (Path(pred_result_json_path).stem + ".eval_multi_hit_iou512.with_gt_vis.json")
    out_csv = out_dir / (Path(pred_result_json_path).stem + ".eval_multi_hit_iou512.with_gt_vis.csv")

    # save json (full details)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pred_result_json_path": pred_result_json_path,
                "case_info_json": case_info_json,
                "case_info_root_path": case_info_root_path,
                "output_dir": OUTPUT_DIR,
                "image_wh": [IMG_W, IMG_H],
                "gt_pick_strategy": GT_PICK_STRATEGY,
                "num_cases": n,
                "total_gt_entities": total_gt_entities,
                "matched_gt_entities": matched_gt_entities,
                "entity_hit_rate_global": entity_hit_rate_global,
                "mean_iou_all_entities_global": mean_iou_all_entities_global,
                "mean_iou_on_matched_global": mean_iou_on_matched_global,
                "total_pred_items": total_pred_items,
                "matched_pred_items": matched_pred_items,
                "pred_match_rate_global": pred_match_rate_global,
                "viz_dir": str(viz_dir),
                "details": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # save csv (compact; heavy fields json-dumped)
    fieldnames = [
        "index", "case_id",
        "parse_reason", "text_json_exists",
        "num_pred_items", "num_gt_entities",
        "matched_gt_entities", "matched_pred_items",
        "entity_hit_rate", "mean_iou_all_entities", "mean_iou_on_matched",
        "pred_items", "gt_entities", "match_pred_to_gt", "pred_ious",
        "viz_pred_pngs", "viz_gt_pngs",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=fieldnames)
        wcsv.writeheader()
        for r in results:
            rr = dict(r)
            for k in ["pred_items", "gt_entities", "match_pred_to_gt", "pred_ious", "viz_pred_pngs", "viz_gt_pngs"]:
                if rr.get(k) is not None:
                    rr[k] = json.dumps(rr[k], ensure_ascii=False)
            wcsv.writerow(rr)

    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_csv}")


if __name__ == "__main__":
    main()
