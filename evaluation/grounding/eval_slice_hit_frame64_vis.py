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


pred_result_json_path = os.environ.get('MEDVOL_EVAL_SLICE_HIT_FRAME64_VIS_PRED_RESULT_JSON_PATH', 'outputs/grounding/output_folder/ct_val_single_64_frames_vqa_exp2.jsonl')
case_info_json = os.environ.get('MEDVOL_EVAL_SLICE_HIT_FRAME64_VIS_CASE_INFO_JSON', 'data/grounding/ct_data_val/ct_val_64_frames_vqa.json')
case_info_root_path = os.environ.get('MEDVOL_EVAL_SLICE_HIT_FRAME64_VIS_CASE_INFO_ROOT_PATH', 'data/grounding/ct_data_val/val_downsample_64_export64_png')
OUTPUT_DIR = os.environ.get('MEDVOL_EVAL_SLICE_HIT_FRAME64_VIS_OUTPUT_DIR', 'outputs/grounding/vis_output_dir/ct_val_single_64_frames_vqa_exp2')

# hit_slices_info["..."]["bbox_xyxy"] 是 512x512 像素坐标
IMG_W, IMG_H = 512, 512

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

# =========================
# 可视化相关开关
# =========================
SAVE_VIZ = True
VIZ_SUBDIR_NAME = "viz_bbox"
ONLY_SAVE_PARSE_OK = False          # True: 只保存 parse_reason == "ok" 的 pred 可视化
DRAW_LINE_WIDTH = 3
DRAW_LABEL = True
SAVE_FAIL_PLACEHOLDER = False       # slice png找不到时是否保存占位图

# GT 画图策略：从 hit_slices_info 里选一个“代表性的 GT slice”来画（不依赖 pred_slice）
GT_PICK_STRATEGY = "max_area"       # {"max_area", "min_z", "max_z", "middle"}


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


def extract_pred_slice_bbox(pred_text: str) -> Tuple[Optional[int], Optional[List[int]], str]:
    if pred_text is None:
        return None, None, "predict_is_none"

    m = ANSWER_RE.search(pred_text)
    if not m:
        return None, None, "no_answer_tag"

    inside = m.group(1).strip()
    try:
        obj = json.loads(inside)
    except Exception as e:
        return None, None, f"answer_json_parse_fail: {type(e).__name__}: {e}"

    if not isinstance(obj, list) or len(obj) != 1 or not isinstance(obj[0], dict):
        return None, None, "answer_not_singleton_list"

    d = obj[0]
    if "slice" not in d or "bbox_2d" not in d:
        return None, None, "missing_slice_or_bbox_2d"

    try:
        s = int(d["slice"])
    except Exception:
        return None, None, "slice_not_int"

    bbox = d["bbox_2d"]
    if (not isinstance(bbox, list)) or len(bbox) != 4:
        return None, None, "bbox_2d_not_len4"

    try:
        bbox_int = [int(round(float(x))) for x in bbox]
    except Exception:
        return None, None, "bbox_2d_not_numeric"

    x1, y1, x2, y2 = bbox_int
    if not (x1 < x2 and y1 < y2):
        return s, bbox_int, "bbox_invalid_order"

    return s, bbox_int, "ok"


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
    x1 = max(0.0, min(float(w), x1))
    x2 = max(0.0, min(float(w), x2))
    y1 = max(0.0, min(float(h), y1))
    y2 = max(0.0, min(float(h), y2))
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
            tw, th = 200, 14
        draw.rectangle([x0, y0, x0 + tw + 2 * pad, y0 + th + 2 * pad], fill=(0, 0, 0))
        draw.text((x0 + pad, y0 + pad), meta_text, fill=(255, 255, 255), font=font)

    if gt_bbox_px is not None:
        draw_bbox(img, gt_bbox_px, color=(0, 255, 0), label="GT", line_width=DRAW_LINE_WIDTH, font=font)
    if pred_bbox_px is not None:
        draw_bbox(img, pred_bbox_px, color=(255, 0, 0), label="P", line_width=DRAW_LINE_WIDTH, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return True


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


def main():
    case_info = safe_load_json(case_info_json)
    case_ids = [x["id"] for x in case_info]
    preds = read_jsonl(pred_result_json_path)

    n = min(len(case_ids), len(preds))
    if len(case_ids) != len(preds):
        print(f"[WARN] length mismatch: case_info={len(case_ids)} vs preds={len(preds)}. Using first {n} aligned by order.")

    root = Path(case_info_root_path)
    results: List[Dict[str, Any]] = []

    hit_count = 0
    iou_sum = 0.0

    out_dir = Path(OUTPUT_DIR)
    viz_dir = out_dir / VIZ_SUBDIR_NAME
    viz_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(n), desc="Evaluating+Vis"):
        case_id = case_ids[i]
        pred_item = preds[i]
        pred_text = pred_item.get("predict", "")

        pred_slice, pred_bbox_norm, parse_reason = extract_pred_slice_bbox(pred_text)

        case_dir = root / case_id
        text_json_path = case_dir / "text.json"

        row: Dict[str, Any] = {
            "index": i,
            "case_id": case_id,
            "pred_slice": pred_slice,
            "pred_bbox_norm1000": pred_bbox_norm,
            "parse_reason": parse_reason,
            "hit": 0,
            "iou": 0.0,
            "gt_slice": None,
            "gt_bbox_xyxy_512": None,
            "text_json_exists": text_json_path.exists(),
            "pred_img_exists": False,
            "gt_img_exists": False,
            "viz_pred_png": None,
            "viz_gt_png": None,
        }

        if not text_json_path.exists():
            results.append(row)
            continue

        try:
            meta = safe_load_json(str(text_json_path))
        except Exception as e:
            row["parse_reason"] = f"text_json_load_fail: {type(e).__name__}: {e}"
            results.append(row)
            continue

        hit_slices_info = meta.get("mask_stats", {}).get("hit_slices_info", {}) or {}

        # 选GT slice（不依赖pred）
        gt_slice, gt_bbox_list = pick_gt_slice_from_hit_slices_info(hit_slices_info, strategy=GT_PICK_STRATEGY)
        row["gt_slice"] = gt_slice
        row["gt_bbox_xyxy_512"] = gt_bbox_list

        gt_bbox_px: Optional[Tuple[float, float, float, float]] = None
        if gt_bbox_list is not None and len(gt_bbox_list) == 4:
            gt_bbox_px = clamp_bbox_xyxy((gt_bbox_list[0], gt_bbox_list[1], gt_bbox_list[2], gt_bbox_list[3]), IMG_W, IMG_H)

        pred_bbox_px: Optional[Tuple[float, float, float, float]] = None
        if pred_slice is not None and pred_bbox_norm is not None:
            pred_bbox_px = clamp_bbox_xyxy(norm1000_to_512_bbox(pred_bbox_norm), IMG_W, IMG_H)

        # hit判定：pred_slice是否在hit_slices_info里
        if pred_slice is not None and isinstance(hit_slices_info, dict):
            if str(pred_slice) in hit_slices_info:
                row["hit"] = 1
                hit_count += 1

        # IoU：只有 pred_slice 命中且该 slice 有 gt bbox 才算
        if row["hit"] == 1 and pred_slice is not None and pred_bbox_px is not None:
            gt_on_pred = hit_slices_info.get(str(pred_slice), {}) if isinstance(hit_slices_info, dict) else {}
            gb = gt_on_pred.get("bbox_xyxy", None) if isinstance(gt_on_pred, dict) else None
            if isinstance(gb, list) and len(gb) == 4:
                gt_on_pred_px = clamp_bbox_xyxy((float(gb[0]), float(gb[1]), float(gb[2]), float(gb[3])), IMG_W, IMG_H)
                iou = iou_xyxy(pred_bbox_px, gt_on_pred_px)
                row["iou"] = iou
                iou_sum += iou

        # =========================
        # 可视化输出：同一个 case 子文件夹里保存 pred_* 和 gt_*
        # =========================
        if SAVE_VIZ:
            case_out_dir = viz_dir / case_id
            case_out_dir.mkdir(parents=True, exist_ok=True)

            # (A) Pred图：画在 pred_slice 对应img上（红框；命中时可选同时画GT绿框）
            if pred_slice is not None and pred_bbox_px is not None:
                if not (ONLY_SAVE_PARSE_OK and parse_reason != "ok"):
                    pred_img_path = find_slice_image(case_dir, int(pred_slice))
                    row["pred_img_exists"] = bool(pred_img_path is not None and pred_img_path.exists())

                    safe_r = safe_reason_token(parse_reason)
                    iou_str = f"{row['iou']:.4f}"
                    hit_str = f"hit{row['hit']}"

                    out_pred_png = case_out_dir / f"pred_slice_{int(pred_slice)}_{hit_str}_iou{iou_str}_{safe_r}.png"
                    meta_text = f"[PRED] {case_id} slice={pred_slice} {hit_str} iou={iou_str} {safe_r}"

                    # Pred图上：仅在 pred_slice 命中时，把该slice的GT也叠加（便于对比）
                    gt_on_pred_px = None
                    if pred_slice is not None and isinstance(hit_slices_info, dict) and str(pred_slice) in hit_slices_info:
                        gt_on_pred = hit_slices_info.get(str(pred_slice), {})
                        gb = gt_on_pred.get("bbox_xyxy", None) if isinstance(gt_on_pred, dict) else None
                        if isinstance(gb, list) and len(gb) == 4:
                            gt_on_pred_px = clamp_bbox_xyxy((float(gb[0]), float(gb[1]), float(gb[2]), float(gb[3])), IMG_W, IMG_H)

                    ok = save_viz_image(
                        img_path=pred_img_path if pred_img_path is not None else None,
                        out_path=out_pred_png,
                        pred_bbox_px=pred_bbox_px,
                        gt_bbox_px=gt_on_pred_px,
                        meta_text=meta_text,
                    )
                    if ok:
                        row["viz_pred_png"] = str(out_pred_png)

            # (B) GT图：画在 GT 选择的 slice 上（绿框；不依赖pred是否命中）
            if gt_slice is not None and gt_bbox_px is not None:
                gt_img_path = find_slice_image(case_dir, int(gt_slice))
                row["gt_img_exists"] = bool(gt_img_path is not None and gt_img_path.exists())

                out_gt_png = case_out_dir / f"gt_slice_{int(gt_slice)}_{GT_PICK_STRATEGY}.png"
                meta_text = f"[GT] {case_id} slice={gt_slice} strategy={GT_PICK_STRATEGY}"

                ok = save_viz_image(
                    img_path=gt_img_path if gt_img_path is not None else None,
                    out_path=out_gt_png,
                    pred_bbox_px=None,
                    gt_bbox_px=gt_bbox_px,
                    meta_text=meta_text,
                )
                if ok:
                    row["viz_gt_png"] = str(out_gt_png)

        results.append(row)

    total = n if n > 0 else 1
    hit_rate = hit_count / total
    mean_iou_all = iou_sum / total

    print("=" * 80)
    print(f"Total cases used: {n}")
    print(f"Hit count:       {hit_count}")
    print(f"Hit rate:        {hit_rate:.6f}")
    print(f"Mean IoU(all):   {mean_iou_all:.6f}")
    print(f"Viz dir:         {viz_dir}")
    print("=" * 80)

    out_json = out_dir / (Path(pred_result_json_path).stem + ".eval_hit_iou512.with_gt_vis.json")
    out_csv = out_dir / (Path(pred_result_json_path).stem + ".eval_hit_iou512.with_gt_vis.csv")

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
                "hit_rate": hit_rate,
                "mean_iou_all": mean_iou_all,
                "viz_dir": str(viz_dir),
                "details": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    fieldnames = [
        "index", "case_id",
        "pred_slice", "pred_bbox_norm1000", "parse_reason",
        "hit", "iou",
        "gt_slice", "gt_bbox_xyxy_512",
        "text_json_exists",
        "pred_img_exists", "gt_img_exists",
        "viz_pred_png", "viz_gt_png",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=fieldnames)
        wcsv.writeheader()
        for r in results:
            rr = dict(r)
            for k in ["pred_bbox_norm1000", "gt_bbox_xyxy_512"]:
                if rr.get(k) is not None:
                    rr[k] = json.dumps(rr[k], ensure_ascii=False)
            wcsv.writerow(rr)

    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_csv}")


if __name__ == "__main__":
    main()
