import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# =========================================================
# Config (edit here)
# =========================================================
ROOT_NPY = os.environ.get('MEDVOL_GENERATE_SFT_ROOT_NPY', 'data/ctorg/ct_org_npy')
ROOT_PNG = os.environ.get('MEDVOL_GENERATE_SFT_ROOT_PNG', 'data/ctorg/ct_org_png')

TRAIN_CSV = os.environ.get('MEDVOL_GENERATE_SFT_TRAIN_CSV', 'data/ctorg/summary_all_train.csv')
TEST_CSV  = os.environ.get('MEDVOL_GENERATE_SFT_TEST_CSV', 'data/ctorg/summary_all_test.csv')

OUT_TRAIN_JSON = os.environ.get('MEDVOL_GENERATE_SFT_OUT_TRAIN_JSON', 'data/ctorg/ct_org_frames_refseg_train.json')
OUT_TEST_JSON = os.environ.get('MEDVOL_GENERATE_SFT_OUT_TEST_JSON', 'data/ctorg/ct_org_frames_refseg_test.json')
TERM_DICT_JSON = os.environ.get('MEDVOL_GENERATE_SFT_TERM_DICT_JSON', 'data_preparation/ctorg/term_dictionary.json')

# ✅ 不强制 64 张（不足也保留），只做统计
REQUIRE_K_IMAGES: Optional[int] = None

SEED = 42
SKIP_IF_MISSING_PNG = True
SKIP_IF_BBOX_NONE = True
SKIP_IF_KEYSLICE_NOT_IN_SELECTED = True

USE_SLICE_TAGS = True
BBOX_NORM_TO_1000 = True

# image size for bbox normalization
IMG_SIZE = 512


# =========================================================
# Prompt builders (你说 prompt 要改，但没给新版本；这里给一个更通用、且兼容你旧格式的版本)
# =========================================================
def build_user_prompt_multi_images(slice_ids: List[int], description_text: str) -> str:
    header = (
        "You are an expert in radiological imaging.\n"
        "You are given multiple axial CT slices.\n"
        f"Referring description: {description_text}\n\n"
        "Task:\n"
        "1) Pick ONE key slice where the referred region is most visible.\n"
        "2) On that slice, output bounding boxes for ALL visible target instances.\n\n"
        "Output format (STRICT):\n"
        "Return ONLY one <answer>...</answer> block.\n"
        "Inside <answer>, output a JSON array with EXACTLY ONE object:\n"
        "[{\"slice\": N, \"bbox_2d_list\": [[x1,y1,x2,y2], ...]}]\n\n"
        "Rules:\n"
        "- N must be one of the provided slice tags.\n"
        "- bbox_2d_list may contain one or multiple boxes.\n"
        "- Coordinates are integers normalized to [0,1000].\n"
        "- For each box: x1<x2 and y1<y2.\n"
        "Example:\n"
        "<answer>[{\"slice\": 369, \"bbox_2d_list\": [[100,200,300,400],[500,100,700,350]]}]</answer>\n\n"
    )

    parts = [header]
    if USE_SLICE_TAGS:
        parts.append(f"Slices ({len(slice_ids)}):\n")
        for sid in slice_ids:
            parts.append(f"<slice {sid}>\n<image>\n")

    parts.append(
        "\nNow answer using the STRICT <answer> JSON format above. "
        "Do not output anything else.\n"
    )
    return "".join(parts)


def _clamp_px(v: int) -> int:
    return max(0, min(IMG_SIZE - 1, int(v)))


def _bbox512_to_bbox1000(b: List[int]) -> List[int]:
    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        raise ValueError(f"bad bbox: {b}")

    x1, y1, x2, y2 = map(int, b)

    # allow x2/y2 be 512; clamp into [0,512]
    x1 = max(0, min(IMG_SIZE, x1))
    y1 = max(0, min(IMG_SIZE, y1))
    x2 = max(0, min(IMG_SIZE, x2))
    y2 = max(0, min(IMG_SIZE, y2))

    # into [0..511]
    x1 = _clamp_px(x1)
    y1 = _clamp_px(y1)
    x2 = _clamp_px(x2)
    y2 = _clamp_px(y2)

    if x2 <= x1:
        x2 = min(IMG_SIZE - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(IMG_SIZE - 1, y1 + 1)

    def to_1000(v_px: int) -> int:
        return int(round(v_px / float(IMG_SIZE) * 1000.0))

    x1n, y1n, x2n, y2n = map(to_1000, (x1, y1, x2, y2))

    def clamp_1000(v: int) -> int:
        return max(0, min(1000, int(v)))

    x1n, y1n, x2n, y2n = map(clamp_1000, (x1n, y1n, x2n, y2n))
    if x2n <= x1n:
        x2n = min(1000, x1n + 1)
    if y2n <= y1n:
        y2n = min(1000, y1n + 1)

    return [x1n, y1n, x2n, y2n]


def build_assistant_answer(slice_id: int, bbox_list_512: List[List[int]]) -> str:
    if not isinstance(bbox_list_512, list):
        bbox_list_512 = []

    cleaned: List[List[int]] = []
    for b in bbox_list_512:
        if isinstance(b, (list, tuple)) and len(b) == 4:
            cleaned.append([int(x) for x in b])

    if BBOX_NORM_TO_1000:
        normed = [_bbox512_to_bbox1000(b) for b in cleaned]
        bbox_str = ",".join([f"[{a},{b},{c},{d}]" for a, b, c, d in normed])
        return f"<answer>[{{\"slice\":{int(slice_id)},\"bbox_2d_list\":[{bbox_str}]}}]</answer>"

    bbox_str = ",".join([f"[{b[0]},{b[1]},{b[2]},{b[3]}]" for b in cleaned])
    return f"<answer>[{{\"slice\":{int(slice_id)},\"bbox_2d_list\":[{bbox_str}]}}]</answer>"


# =========================================================
# Helpers
# =========================================================
def read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_term_dictionary(p: str) -> Dict[str, List[str]]:
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    out: Dict[str, List[str]] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(k, str) and isinstance(v, list):
                out[k] = [str(x) for x in v if isinstance(x, (str, int, float))]
    return out


def pick_random_description(term_dict: Dict[str, List[str]], label_name: str) -> Optional[str]:
    for k in [label_name, label_name.lower()]:
        if k in term_dict and len(term_dict[k]) > 0:
            return random.choice(term_dict[k]).strip()
    return None


def read_case_ids_from_csv(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    if "case" not in df.columns:
        raise ValueError(f"{csv_path} 缺少 'case' 列，当前列：{list(df.columns)}")
    cases = df["case"].astype(str).tolist()
    # 去重保持顺序
    seen = set()
    uniq = []
    for c in cases:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def parse_slice_id_from_filename(name: str) -> Optional[int]:
    """
    现在文件名形如 slice_0369.png
    返回 int(369)
    """
    m = re.search(r"slice_(\d+)", name)
    if not m:
        # fallback：任何数字
        m2 = re.search(r"(\d+)", name)
        if not m2:
            return None
        return int(m2.group(1))
    return int(m.group(1))


def list_case_images_png64(png_64_dir: Path) -> Dict[int, Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    out: Dict[int, Path] = {}
    for p in png_64_dir.iterdir():
        if not p.is_file():
            continue
        if p.name.lower() == "text.json":
            continue
        if p.suffix.lower() not in exts:
            continue
        sid = parse_slice_id_from_filename(p.name)
        if sid is None:
            continue
        out[sid] = p
    return out


def _get_bbox_list_from_label_item(li: Dict[str, Any]) -> Optional[List[List[int]]]:
    """
    支持：
      - bbox_2d_list: [[x1,y1,x2,y2], ...]
      - bbox_2d: [x1,y1,x2,y2]
    """
    if not isinstance(li, dict):
        return None

    if isinstance(li.get("bbox_2d_list", None), list):
        bbl = li["bbox_2d_list"]
        cleaned = []
        for b in bbl:
            if isinstance(b, (list, tuple)) and len(b) == 4:
                cleaned.append([int(x) for x in b])
        return cleaned

    b = li.get("bbox_2d", None)
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return [[int(x) for x in b]]

    return None


def get_case_paths(case_id: str) -> Tuple[Path, Path]:
    """
    NPY: ROOT_NPY/<case_id>/
    PNG: ROOT_PNG/<case_id>/png_64slices/
    """
    npy_case_dir = Path(ROOT_NPY) / case_id
    png_64_dir = Path(ROOT_PNG) / case_id / "png_64slices"
    return npy_case_dir, png_64_dir


# =========================================================
# Main
# =========================================================
def generate_refseg_vqa_for_cases(case_ids: List[str], out_json_path: str, term_dict_json: str):
    random.seed(SEED)

    out_json_path = Path(out_json_path)
    term_dict = load_term_dictionary(term_dict_json)

    samples: List[Dict[str, Any]] = []

    # stats
    case_missing_png64_dir = 0
    case_missing_text = 0
    missing_selected = 0
    bad_k_images = 0
    missing_label_info = 0
    case_missing_png_for_selected = 0
    label_no_desc = 0
    label_no_bbox = 0
    key_not_in_selected = 0
    kept = 0

    for case_id in case_ids:
        npy_case_dir, png_64_dir = get_case_paths(case_id)

        if not png_64_dir.exists():
            case_missing_png64_dir += 1
            continue

        text_path = png_64_dir / "text.json"
        if not text_path.exists():
            case_missing_text += 1
            continue

        tj = read_json(text_path)

        selected_slices = tj.get("selected_slices", None)
        if not isinstance(selected_slices, list) or len(selected_slices) == 0:
            missing_selected += 1
            continue
        selected_slices = [int(x) for x in selected_slices]

        if REQUIRE_K_IMAGES is not None and len(selected_slices) != int(REQUIRE_K_IMAGES):
            bad_k_images += 1  # only stats

        # ✅ 新版字段是 "labels"，旧版可能是 "label_info"
        label_info = tj.get("labels", None)
        if label_info is None:
            label_info = tj.get("label_info", None)

        if not isinstance(label_info, list) or len(label_info) == 0:
            missing_label_info += 1
            continue

        # ✅ 过滤 present=false（可选但强烈建议）
        label_info = [li for li in label_info if isinstance(li, dict) and li.get("present", True)]
        if len(label_info) == 0:
            # 全部 present=false 也算缺失有效标签
            missing_label_info += 1
            continue

        img_map = list_case_images_png64(png_64_dir)

        if SKIP_IF_MISSING_PNG:
            if any((sid not in img_map) for sid in selected_slices):
                case_missing_png_for_selected += 1
                continue

        images_abs = [img_map[sid].as_posix() for sid in selected_slices if sid in img_map]
        if len(images_abs) == 0:
            case_missing_png_for_selected += 1
            continue

        for li in label_info:
            if not isinstance(li, dict):
                continue

            label_name = str(li.get("label_name", "")).strip()
            key_slice = li.get("key_slice", None)
            if key_slice is None:
                label_no_bbox += 1
                continue
            key_slice = int(key_slice)

            if SKIP_IF_KEYSLICE_NOT_IN_SELECTED and (key_slice not in selected_slices):
                key_not_in_selected += 1
                continue

            bbox_list_512 = _get_bbox_list_from_label_item(li)
            if SKIP_IF_BBOX_NONE and (bbox_list_512 is None or len(bbox_list_512) == 0):
                label_no_bbox += 1
                continue

            desc = pick_random_description(term_dict, label_name)
            if not desc:
                label_no_desc += 1
                continue

            user_content = build_user_prompt_multi_images(selected_slices, desc)
            assistant_content = build_assistant_answer(key_slice, bbox_list_512)

            sample_id = f"{case_id}__{label_name}"

            samples.append({
                "id": sample_id,
                "images": images_abs,
                "conversations": [
                    {"from": "human", "value": user_content},
                    {"from": "gpt", "value": assistant_content},
                ],
                "meta": {
                    "case_id": case_id,
                    "label_name": label_name,
                    "description": desc,
                    "key_slice": key_slice,
                    "bbox_list_512": bbox_list_512,
                    "num_images": len(images_abs),

                    # 额外把路径也记下来（方便你后续对齐）
                    "png_64_dir": png_64_dir.as_posix(),
                    "npy_case_dir": npy_case_dir.as_posix(),
                }
            })
            kept += 1

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote -> {out_json_path} (num_samples={len(samples)})")
    print("[STATS]")
    print(f"  total_cases_in_split              = {len(case_ids)}")
    print(f"  case_missing_png64_dir            = {case_missing_png64_dir}")
    print(f"  missing_text.json                 = {case_missing_text}")
    print(f"  missing_selected_slices           = {missing_selected}")
    print(f"  bad_k_images (stats only)         = {bad_k_images} (REQUIRE_K_IMAGES={REQUIRE_K_IMAGES})")
    print(f"  missing_label_info                = {missing_label_info}")
    print(f"  case_missing_png_for_selected     = {case_missing_png_for_selected} (SKIP_IF_MISSING_PNG={SKIP_IF_MISSING_PNG})")
    print(f"  label_no_description              = {label_no_desc}")
    print(f"  label_no_bbox/key_slice/list      = {label_no_bbox}")
    print(f"  key_slice_not_in_selected         = {key_not_in_selected}")
    print(f"  kept_samples                      = {kept}")


def main():
    train_cases = read_case_ids_from_csv(TRAIN_CSV)
    test_cases = read_case_ids_from_csv(TEST_CSV)

    print("========== Split Sizes ==========")
    print(f"train cases: {len(train_cases)} from {TRAIN_CSV}")
    print(f"test  cases: {len(test_cases)} from {TEST_CSV}")
    print("ROOT_NPY:", ROOT_NPY)
    print("ROOT_PNG:", ROOT_PNG)
    print("================================\n")

    print("========== Generate TRAIN VQA ==========")
    generate_refseg_vqa_for_cases(
        case_ids=train_cases,
        out_json_path=OUT_TRAIN_JSON,
        term_dict_json=TERM_DICT_JSON,
    )

    print("\n========== Generate TEST VQA ==========")
    generate_refseg_vqa_for_cases(
        case_ids=test_cases,
        out_json_path=OUT_TEST_JSON,
        term_dict_json=TERM_DICT_JSON,
    )


if __name__ == "__main__":
    main()
