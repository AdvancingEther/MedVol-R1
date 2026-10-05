#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


# =========================
# User config (edit here)
# =========================
pred_result_json_path = os.environ.get('MEDVOL_EVAL_SLICE_HIT_MULTI_FRAME64_PRED_RESULT_JSON_PATH', 'outputs/grounding/output_folder/ct_val_64_frames_vqa_multi_entity_all_finding.jsonl')
case_info_json = os.environ.get('MEDVOL_EVAL_SLICE_HIT_MULTI_FRAME64_CASE_INFO_JSON', 'data/grounding/ct_data_val/ct_val_64_frames_vqa_multi_entity_all_finding.json')
case_info_root_path = os.environ.get('MEDVOL_EVAL_SLICE_HIT_MULTI_FRAME64_CASE_INFO_ROOT_PATH', 'data/grounding/ct_data_val/val_downsample_64_all_finding_export64_png')
OUTPUT_DIR = os.environ.get('MEDVOL_EVAL_SLICE_HIT_MULTI_FRAME64_OUTPUT_DIR', 'outputs/grounding/eval_multi_entity_hungarian')

# image size for GT bbox (xyxy in 512 space)
IMG_W, IMG_H = 512, 512

# IoU threshold for "hit_rate_iou"
IOU_HIT_TH = 0.1

# If True: require pred JSON array non-empty; else empty -> skip as invalid
REQUIRE_NONEMPTY_PRED = True

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

# =========================
# Visualization switches
# =========================
SAVE_VIZ = True
VIZ_SUBDIR_NAME = "viz_bbox"
DRAW_LINE_WIDTH = 3
DRAW_LABEL = True
ONLY_SAVE_PARSE_OK = False          # True -> only viz when pred_parse_reason == "ok"
SAVE_FAIL_PLACEHOLDER = False       # True -> if slice png missing, save black placeholder
MAX_VIZ_SLICES_PER_CASE = 32        # safety: at most this many slices per case (union of pred/gt slices); -1 for no limit


# =========================
# IO helpers
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
# Parse prediction: multi-entity
# =========================
def extract_pred_entities(pred_text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Expect:
      <answer>[{"entity":0,"slice":43,"bbox_2d":[...]} , ...]</answer>
    Return:
      entities: list of {"slice":int,"bbox_2d":[int,int,int,int],"entity":optional}
      reason: "ok" or error string
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

    if REQUIRE_NONEMPTY_PRED and len(obj) == 0:
        return [], "answer_empty_list"

    entities: List[Dict[str, Any]] = []
    for it in obj:
        if not isinstance(it, dict):
            continue
        if "slice" not in it or "bbox_2d" not in it:
            continue

        try:
            s = int(it["slice"])
        except Exception:
            continue

        bbox = it["bbox_2d"]
        if (not isinstance(bbox, list)) or len(bbox) != 4:
            continue

        try:
            bbox_int = [int(round(float(x))) for x in bbox]
        except Exception:
            continue

        # keep even if invalid order (will iou=0)
        rec = {"slice": s, "bbox_2d": bbox_int}
        if "entity" in it:
            try:
                rec["entity"] = int(it["entity"])
            except Exception:
                pass
        entities.append(rec)

    if (REQUIRE_NONEMPTY_PRED and len(entities) == 0):
        return [], "no_valid_entity_items"

    # stable order if entity exists
    if len(entities) > 0 and all(("entity" in e) for e in entities):
        entities.sort(key=lambda x: int(x.get("entity", 10**9)))

    return entities, "ok"


# =========================
# bbox + IoU utilities
# =========================
def clamp_bbox_xyxy(
    b: Tuple[float, float, float, float],
    w: int = IMG_W,
    h: int = IMG_H
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    x1 = max(0.0, min(float(w), x1))
    x2 = max(0.0, min(float(w), x2))
    y1 = max(0.0, min(float(h), y1))
    y2 = max(0.0, min(float(h), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def norm1000_to_512_bbox(b: List[int], w: int = IMG_W, h: int = IMG_H) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    px1 = x1 / 1000.0 * w
    py1 = y1 / 1000.0 * h
    px2 = x2 / 1000.0 * w
    py2 = y2 / 1000.0 * h
    return px1, py1, px2, py2


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


def extract_bbox_xyxy_512_from_item(d: Any) -> Optional[List[float]]:
    if not isinstance(d, dict):
        return None
    for k in ["bbox_xyxy", "bbox_xyxy_512", "xyxy", "x1y1x2y2", "bbox"]:
        v = d.get(k, None)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            try:
                return [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
            except Exception:
                return None
    if all(kk in d for kk in ["x1", "y1", "x2", "y2"]):
        try:
            return [float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])]
        except Exception:
            return None
    return None


def ensure_bbox_in_512_space(b: List[float]) -> List[float]:
    if b is None or len(b) != 4:
        return b
    mx = max(b)
    if mx <= 512.0 + 1e-6:
        return b
    if mx <= 1000.0 + 1e-6:
        return [x / 1000.0 * 512.0 for x in b]
    return b


# =========================
# GT extraction from text.json (multi-entity)
# =========================
def _as_list_mask_stats(tj: Dict[str, Any]) -> List[Dict[str, Any]]:
    ms = tj.get("mask_stats", None)
    if isinstance(ms, list):
        return [x for x in ms if isinstance(x, dict)]
    if isinstance(ms, dict):
        inner = ms.get("mask_stats", None)
        if isinstance(inner, list):
            return [x for x in inner if isinstance(x, dict)]
    return []


def _as_dict_hit_slices_info(obj: Any) -> Dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def extract_gt_boxes_from_text_json(tj: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Set[int]]:
    """
    Return:
      gt_boxes: list of {
        "gt_index": int,
        "entity": int,
        "slice": int,
        "bbox_xyxy_512": [x1,y1,x2,y2],
        "area": float
      }
      gt_slice_set: set of slice indices that have any GT box
    """
    gt_boxes: List[Dict[str, Any]] = []
    gt_slice_set: Set[int] = set()

    ms_list = _as_list_mask_stats(tj)

    # fallback: old single-entity dict style
    if len(ms_list) == 0:
        ms = tj.get("mask_stats", None)
        if isinstance(ms, dict):
            hsi = _as_dict_hit_slices_info(ms.get("hit_slices_info", {}))
            if len(hsi) > 0:
                ms_list = [{"entity_index": 0, "hit_slices_info": hsi}]

    if len(ms_list) == 0:
        return [], set()

    gt_idx = 0
    for ent_i, ent in enumerate(ms_list):
        entity_id = ent_i
        if "entity_index" in ent:
            try:
                entity_id = int(ent["entity_index"])
            except Exception:
                entity_id = ent_i

        hsi = _as_dict_hit_slices_info(ent.get("hit_slices_info", {}))
        if len(hsi) > 0:
            for k, v in hsi.items():
                try:
                    s = int(k)
                except Exception:
                    continue
                bbox = extract_bbox_xyxy_512_from_item(v)
                if bbox is None:
                    continue
                bbox = ensure_bbox_in_512_space(bbox)
                bbox_px = clamp_bbox_xyxy((bbox[0], bbox[1], bbox[2], bbox[3]), IMG_W, IMG_H)
                area = 0.0
                if isinstance(v, dict):
                    try:
                        area = float(v.get("area", 0.0))
                    except Exception:
                        area = 0.0

                gt_boxes.append({
                    "gt_index": gt_idx,
                    "entity": int(entity_id),
                    "slice": int(s),
                    "bbox_xyxy_512": [float(bbox_px[0]), float(bbox_px[1]), float(bbox_px[2]), float(bbox_px[3])],
                    "area": float(area),
                })
                gt_slice_set.add(int(s))
                gt_idx += 1
            continue

        # single hit slice
        h = ent.get("hit_slice_info", None)
        if isinstance(h, dict):
            try:
                s = int(h.get("slice"))
            except Exception:
                s = None
            bbox = extract_bbox_xyxy_512_from_item(h)
            if s is not None and bbox is not None:
                bbox = ensure_bbox_in_512_space(bbox)
                bbox_px = clamp_bbox_xyxy((bbox[0], bbox[1], bbox[2], bbox[3]), IMG_W, IMG_H)
                area = 0.0
                try:
                    area = float(h.get("area", 0.0))
                except Exception:
                    area = 0.0

                gt_boxes.append({
                    "gt_index": gt_idx,
                    "entity": int(entity_id),
                    "slice": int(s),
                    "bbox_xyxy_512": [float(bbox_px[0]), float(bbox_px[1]), float(bbox_px[2]), float(bbox_px[3])],
                    "area": float(area),
                })
                gt_slice_set.add(int(s))
                gt_idx += 1
                continue

        # fallback: max_area_slice + bbox
        if "max_area_slice_index" in ent and ("bbox_xyxy" in ent or "bbox_xyxy_512" in ent):
            try:
                s = int(ent["max_area_slice_index"])
            except Exception:
                s = None
            bbox = extract_bbox_xyxy_512_from_item(ent)
            if s is not None and bbox is not None:
                bbox = ensure_bbox_in_512_space(bbox)
                bbox_px = clamp_bbox_xyxy((bbox[0], bbox[1], bbox[2], bbox[3]), IMG_W, IMG_H)
                gt_boxes.append({
                    "gt_index": gt_idx,
                    "entity": int(entity_id),
                    "slice": int(s),
                    "bbox_xyxy_512": [float(bbox_px[0]), float(bbox_px[1]), float(bbox_px[2]), float(bbox_px[3])],
                    "area": 0.0,
                })
                gt_slice_set.add(int(s))
                gt_idx += 1

    return gt_boxes, gt_slice_set


# =========================
# Hungarian algorithm (min cost), requires cols >= rows
# =========================
def hungarian_min_cost(cost: List[List[float]]) -> List[int]:
    """
    Solve min cost assignment for rectangular matrix with n_rows <= n_cols.
    Return: assignment list of length n_rows, each is chosen col index (0-based).
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if m == 0:
        return []

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    ans = [-1] * (n + 1)
    for j in range(1, m + 1):
        if p[j] != 0:
            ans[p[j]] = j
    return [ans[i] - 1 for i in range(1, n + 1)]


def maximize_iou_assignment(iou_mat: List[List[float]]) -> List[int]:
    """
    iou_mat: n_pred x n_gt
    Return assignment: length n_pred, each matched gt index or -1 (dummy)
    """
    n = len(iou_mat)
    if n == 0:
        return []
    m0 = len(iou_mat[0]) if n > 0 else 0

    m = max(m0, n)
    cost = []
    for i in range(n):
        row = []
        for j in range(m):
            iou = 0.0
            if j < m0:
                iou = float(iou_mat[i][j])
            iou = max(0.0, min(1.0, iou))
            row.append(1.0 - iou)
        cost.append(row)

    assign_cols = hungarian_min_cost(cost)
    out = []
    for j in assign_cols:
        out.append(int(j) if j < m0 else -1)
    return out


# =========================
# Case id alignment
# =========================
def build_pred_map(preds: List[Dict[str, Any]]) -> Optional[Dict[str, Dict[str, Any]]]:
    ok = True
    mp: Dict[str, Dict[str, Any]] = {}
    for it in preds:
        if not isinstance(it, dict) or "id" not in it:
            ok = False
            break
        mp[str(it["id"])] = it
    return mp if ok else None


# =========================
# Visualization helpers
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
            tw, th = 120, 14
        pad = 2
        draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad], fill=(0, 0, 0))
        draw.text((tx + pad, ty + pad), label, fill=color, font=font)


def safe_token(s: str, maxlen: int = 120) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(s))[:maxlen]


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
    # fallback search
    for p in sorted(case_dir.glob("slice_*.png")):
        m = re.search(r"slice_(\d+)\.png$", p.name)
        if m and int(m.group(1)) == int(slice_idx):
            return p
    return None


def save_viz_slice(
    img_path: Optional[Path],
    out_path: Path,
    header_text: str,
    gt_boxes_px: List[Tuple[Tuple[float, float, float, float], str]],
    pred_boxes_px: List[Tuple[Tuple[float, float, float, float], str]],
) -> bool:
    if img_path is None or (not img_path.exists()):
        if not SAVE_FAIL_PLACEHOLDER:
            return False
        img = Image.new("RGB", (IMG_W, IMG_H), (0, 0, 0))
    else:
        img = Image.open(img_path).convert("RGB")

    font = try_load_font(14)

    # header
    if DRAW_LABEL and header_text:
        draw = ImageDraw.Draw(img)
        pad = 3
        x0, y0 = 2, 2
        try:
            tb = draw.textbbox((x0, y0), header_text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = 300, 14
        draw.rectangle([x0, y0, x0 + tw + 2 * pad, y0 + th + 2 * pad], fill=(0, 0, 0))
        draw.text((x0 + pad, y0 + pad), header_text, fill=(255, 255, 255), font=font)

    # draw gt first (green), then pred (red)
    for bbox, lab in gt_boxes_px:
        draw_bbox(img, bbox, color=(0, 255, 0), label=lab, line_width=DRAW_LINE_WIDTH, font=font)
    for bbox, lab in pred_boxes_px:
        draw_bbox(img, bbox, color=(255, 0, 0), label=lab, line_width=DRAW_LINE_WIDTH, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return True


# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_dir = Path(OUTPUT_DIR)
    out_json = out_dir / (Path(pred_result_json_path).stem + ".multi_entity_hungarian.eval.json")
    out_csv = out_dir / (Path(pred_result_json_path).stem + ".multi_entity_hungarian.eval.csv")

    viz_dir = out_dir / VIZ_SUBDIR_NAME
    if SAVE_VIZ:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # case ids
    case_info = safe_load_json(case_info_json)
    case_ids = [str(x["id"]) for x in case_info if isinstance(x, dict) and "id" in x]

    preds = read_jsonl(pred_result_json_path)

    pred_map = build_pred_map(preds)
    use_mapping = pred_map is not None and len(pred_map) > 0

    aligned_items: List[Tuple[str, Dict[str, Any]]] = []
    missing_pred_by_id = 0

    if use_mapping:
        for cid in case_ids:
            it = pred_map.get(cid, None)
            if it is None:
                missing_pred_by_id += 1
                continue
            aligned_items.append((cid, it))
    else:
        n = min(len(case_ids), len(preds))
        if len(case_ids) != len(preds):
            print(f"[WARN] length mismatch: case_info={len(case_ids)} vs preds={len(preds)}. Using first {n} aligned by order.")
        for i in range(n):
            aligned_items.append((case_ids[i], preds[i]))

    root = Path(case_info_root_path)

    total_cases = 0
    total_preds = 0
    sum_iou_over_preds = 0.0
    sum_case_mean_iou = 0.0
    sum_hit_slice_over_preds = 0
    sum_hit_iou_over_preds = 0

    missing_text = 0
    pred_parse_fail = 0
    empty_gt = 0

    details: List[Dict[str, Any]] = []

    for case_id, pred_item in tqdm(aligned_items, desc="Eval Hungarian (multi-entity)+Viz"):
        case_dir = root / case_id
        text_path = case_dir / "text.json"
        if not text_path.exists():
            missing_text += 1
            continue

        try:
            tj = safe_load_json(str(text_path))
        except Exception:
            missing_text += 1
            continue

        gt_boxes, gt_slice_set = extract_gt_boxes_from_text_json(tj)
        if len(gt_boxes) == 0:
            empty_gt += 1
            continue

        pred_text = pred_item.get("predict", "")
        pred_entities, reason = extract_pred_entities(pred_text)

        if reason != "ok":
            pred_parse_fail += 1
            pred_entities = []

        # prepare pred list
        pred_list = []
        for pe in pred_entities:
            s = pe.get("slice", None)
            b = pe.get("bbox_2d", None)
            if s is None or b is None:
                continue
            if not (isinstance(b, list) and len(b) == 4):
                continue
            try:
                s = int(s)
                b = [int(round(float(x))) for x in b]
            except Exception:
                continue
            pred_list.append({"slice": s, "bbox_2d": b})

        P = len(pred_list)
        G = len(gt_boxes)

        # build IoU matrix
        iou_mat: List[List[float]] = []
        for i in range(P):
            ps = int(pred_list[i]["slice"])
            pb = clamp_bbox_xyxy(norm1000_to_512_bbox(pred_list[i]["bbox_2d"]), IMG_W, IMG_H)
            row = []
            for j in range(G):
                gs = int(gt_boxes[j]["slice"])
                if ps != gs:
                    row.append(0.0)
                    continue
                gb_list = gt_boxes[j]["bbox_xyxy_512"]
                gb = clamp_bbox_xyxy((float(gb_list[0]), float(gb_list[1]), float(gb_list[2]), float(gb_list[3])), IMG_W, IMG_H)
                row.append(iou_xyxy(pb, gb))
            iou_mat.append(row)

        assign = maximize_iou_assignment(iou_mat) if P > 0 else []

        per_pred = []
        sum_iou_case = 0.0
        hit_slice_case = 0
        hit_iou_case = 0

        for i in range(P):
            ps = int(pred_list[i]["slice"])
            j = assign[i] if i < len(assign) else -1

            hit_slice_any = 1 if ps in gt_slice_set else 0

            matched_iou = 0.0
            matched_gt = None
            if j is not None and int(j) >= 0 and int(j) < G:
                matched_iou = float(iou_mat[i][j]) if (P > 0 and G > 0) else 0.0
                matched_gt = {
                    "gt_index": int(gt_boxes[j]["gt_index"]),
                    "entity": int(gt_boxes[j]["entity"]),
                    "slice": int(gt_boxes[j]["slice"]),
                    "bbox_xyxy_512": gt_boxes[j]["bbox_xyxy_512"],
                }

            sum_iou_case += matched_iou
            hit_slice_case += hit_slice_any
            hit_iou_case += (1 if matched_iou >= IOU_HIT_TH else 0)

            per_pred.append({
                "pred_index": i,
                "pred_slice": ps,
                "pred_bbox_norm1000": pred_list[i]["bbox_2d"],
                "assigned_gt": matched_gt,
                "matched_iou": matched_iou,
                "hit_slice_any": hit_slice_any,
                "hit_iou": int(matched_iou >= IOU_HIT_TH),
            })

        mean_iou_pred = (sum_iou_case / P) if P > 0 else 0.0
        hit_rate_slice = (hit_slice_case / P) if P > 0 else 0.0
        hit_rate_iou = (hit_iou_case / P) if P > 0 else 0.0

        # global
        total_cases += 1
        sum_case_mean_iou += mean_iou_pred

        total_preds += P
        sum_iou_over_preds += sum_iou_case
        sum_hit_slice_over_preds += hit_slice_case
        sum_hit_iou_over_preds += hit_iou_case

        # =========================
        # Visualization (per slice)
        # =========================
        if SAVE_VIZ:
            if (ONLY_SAVE_PARSE_OK and reason != "ok"):
                pass
            else:
                # decide which slices to render
                pred_slices = set([int(x["slice"]) for x in pred_list])
                gt_slices = set([int(x["slice"]) for x in gt_boxes])
                slices_to_draw = sorted(list(pred_slices.union(gt_slices)))

                if MAX_VIZ_SLICES_PER_CASE > 0 and len(slices_to_draw) > MAX_VIZ_SLICES_PER_CASE:
                    # heuristic: keep all pred slices, and fill with gt slices (sorted) until limit
                    keep = set(sorted(list(pred_slices)))
                    for s in sorted(list(gt_slices)):
                        if len(keep) >= MAX_VIZ_SLICES_PER_CASE:
                            break
                        keep.add(int(s))
                    slices_to_draw = sorted(list(keep))

                case_out_dir = viz_dir / case_id
                case_out_dir.mkdir(parents=True, exist_ok=True)

                # build slice -> list of gt boxes (px) with labels
                gt_by_slice: Dict[int, List[Tuple[Tuple[float, float, float, float], str]]] = {}
                for g in gt_boxes:
                    s = int(g["slice"])
                    bb = g["bbox_xyxy_512"]
                    bbox_px = clamp_bbox_xyxy((float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])), IMG_W, IMG_H)
                    lab = f"G{int(g['entity'])}#{int(g['gt_index'])}"
                    gt_by_slice.setdefault(s, []).append((bbox_px, lab))

                # build slice -> list of pred boxes (px) with labels (include matching info)
                pred_by_slice: Dict[int, List[Tuple[Tuple[float, float, float, float], str]]] = {}
                for pp in per_pred:
                    s = int(pp["pred_slice"])
                    pb = pp["pred_bbox_norm1000"]
                    bbox_px = clamp_bbox_xyxy(norm1000_to_512_bbox(pb), IMG_W, IMG_H)

                    lab = f"P{int(pp['pred_index'])}"
                    if pp.get("assigned_gt", None) is not None:
                        ag = pp["assigned_gt"]
                        lab += f"->G{ag['entity']}#{ag['gt_index']} iou={pp['matched_iou']:.2f}"
                    else:
                        lab += f"->None iou={pp['matched_iou']:.2f}"

                    pred_by_slice.setdefault(s, []).append((bbox_px, lab))

                # render each slice
                for s in slices_to_draw:
                    img_path = find_slice_image(case_dir, int(s))
                    header = f"{case_id} | slice={s} | P={P} G={G} | meanIoU={mean_iou_pred:.3f} | hitS={hit_rate_slice:.2f} hitI={hit_rate_iou:.2f} | {reason}"
                    out_name = f"slice_{int(s):03d}_meanIoU{mean_iou_pred:.3f}_hitS{hit_rate_slice:.2f}_hitI{hit_rate_iou:.2f}_{safe_token(reason)}.png"
                    out_path = case_out_dir / out_name

                    save_viz_slice(
                        img_path=img_path,
                        out_path=out_path,
                        header_text=header,
                        gt_boxes_px=gt_by_slice.get(int(s), []),
                        pred_boxes_px=pred_by_slice.get(int(s), []),
                    )

        details.append({
            "case_id": case_id,
            "pred_parse_reason": reason,
            "num_pred": P,
            "num_gt": G,
            "mean_iou_pred": mean_iou_pred,
            "hit_rate_slice": hit_rate_slice,
            "hit_rate_iou": hit_rate_iou,
            "per_pred": per_pred,
        })

    # overall
    overall_mean_iou_weighted = (sum_iou_over_preds / total_preds) if total_preds > 0 else 0.0
    overall_hit_rate_slice_weighted = (sum_hit_slice_over_preds / total_preds) if total_preds > 0 else 0.0
    overall_hit_rate_iou_weighted = (sum_hit_iou_over_preds / total_preds) if total_preds > 0 else 0.0
    overall_mean_iou_by_case = (sum_case_mean_iou / total_cases) if total_cases > 0 else 0.0

    summary = {
        "pred_result_json_path": pred_result_json_path,
        "case_info_json": case_info_json,
        "case_info_root_path": case_info_root_path,
        "output_dir": OUTPUT_DIR,
        "viz_dir": str(viz_dir) if SAVE_VIZ else None,
        "image_wh": [IMG_W, IMG_H],
        "iou_hit_threshold": IOU_HIT_TH,
        "use_id_mapping": bool(use_mapping),
        "missing_pred_by_id": int(missing_pred_by_id) if use_mapping else 0,
        "total_cases_evaluated": total_cases,
        "total_pred_entities": total_preds,
        "missing_text_json": missing_text,
        "pred_parse_fail_cases": pred_parse_fail,
        "empty_gt_cases": empty_gt,
        "overall_mean_iou_weighted_by_pred": overall_mean_iou_weighted,
        "overall_hit_rate_slice_weighted_by_pred": overall_hit_rate_slice_weighted,
        "overall_hit_rate_iou_weighted_by_pred": overall_hit_rate_iou_weighted,
        "overall_mean_iou_by_case": overall_mean_iou_by_case,
    }

    # save json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details}, f, ensure_ascii=False, indent=2)

    # save csv
    fieldnames = [
        "case_id", "pred_parse_reason",
        "num_pred", "num_gt",
        "mean_iou_pred", "hit_rate_slice", "hit_rate_iou",
        "per_pred_json"
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for d in details:
            w.writerow({
                "case_id": d["case_id"],
                "pred_parse_reason": d["pred_parse_reason"],
                "num_pred": d["num_pred"],
                "num_gt": d["num_gt"],
                "mean_iou_pred": f"{d['mean_iou_pred']:.6f}",
                "hit_rate_slice": f"{d['hit_rate_slice']:.6f}",
                "hit_rate_iou": f"{d['hit_rate_iou']:.6f}",
                "per_pred_json": json.dumps(d["per_pred"], ensure_ascii=False),
            })

    print("=" * 90)
    print("[SUMMARY]")
    for k, v in summary.items():
        print(f"{k:40s}: {v}")
    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_csv}")
    if SAVE_VIZ:
        print(f"[Saved] viz -> {viz_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
