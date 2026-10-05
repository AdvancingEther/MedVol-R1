import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =========================================================
# Config (edit here)
# =========================================================
template_path = os.environ.get('MEDVOL_STEP2_GEN_SFT_VQA_VIA_TMPLATE_TEMPLATE_PATH', 'data_preparation/kits23/template.json')

entity_info_file = os.environ.get('MEDVOL_STEP2_GEN_SFT_VQA_VIA_TMPLATE_ENTITY_INFO_FILE', 'data/kits23/final_vqa_gen/entity_with_templates.json')

root_npy = os.environ.get('MEDVOL_STEP2_GEN_SFT_VQA_VIA_TMPLATE_ROOT_NPY', 'data/kits23/kits23_npy_m3d')
root_png = os.environ.get('MEDVOL_STEP2_GEN_SFT_VQA_VIA_TMPLATE_ROOT_PNG', 'data/kits23/kits23_png_m3d')

split_file = os.environ.get('MEDVOL_STEP2_GEN_SFT_VQA_VIA_TMPLATE_SPLIT_FILE', 'data/kits23/split.json')

OUT_TRAIN_JSON = os.environ.get('MEDVOL_STEP2_GEN_SFT_VQA_VIA_TMPLATE_OUT_TRAIN_JSON', 'data/kits23/kits23_frames_refseg_train.json')
OUT_TEST_JSON  = os.environ.get('MEDVOL_STEP2_GEN_SFT_VQA_VIA_TMPLATE_OUT_TEST_JSON', 'data/kits23/kits23_frames_refseg_test.json')

SEED = 42
SKIP_IF_MISSING_PNG = True
SKIP_IF_BBOX_NONE = True
SKIP_IF_KEYSLICE_NOT_IN_SELECTED = True

USE_SLICE_TAGS = True
BBOX_NORM_TO_1000 = True

# image size for bbox normalization
IMG_SIZE = 512

# 你当前确认的先验（若后面发现方向反了，改这里即可）
KIDNEY_SIDE_MAP = {
    "kidney_1": "the right kidney",
    "kidney_2": "the left kidney",
}

# =========================================================
# Template whitelist
# None -> use all template_ids
# Otherwise -> only keep these template_ids
# 当前默认：去掉 highest/lowest，只保留 largest
# =========================================================
ALLOWED_TEMPLATE_IDS = {
    "kidney_only_with_tumor",
    "kidney_only_with_cyst",
    "kidney_only_with_both_tumor_and_cyst",
    "kidney_only_healthy",

    "tumor_only_global",
    "tumor_only_in_kidney",
    "tumor_largest_in_kidney",

    "cyst_only_global",
    "cyst_only_in_kidney",
    "cyst_largest_in_kidney",
}


# =========================================================
# Prompt builders
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

    x1 = max(0, min(IMG_SIZE, x1))
    y1 = max(0, min(IMG_SIZE, y1))
    x2 = max(0, min(IMG_SIZE, x2))
    y2 = max(0, min(IMG_SIZE, y2))

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
# JSON / dictionary helpers
# =========================================================
def read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_template_dictionary(p: str) -> Dict[str, List[str]]:
    """
    Supports:
    1) flat format:
       { "tumor_only_global": ["...", "..."] }

    2) nested format:
       { "tumor_only_global": { "direct": [...], "descriptive": [...] } }
    """
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)

    out: Dict[str, List[str]] = {}
    if not isinstance(d, dict):
        return out

    for k, v in d.items():
        if not isinstance(k, str):
            continue

        collected: List[str] = []

        if isinstance(v, list):
            collected.extend([str(x).strip() for x in v if isinstance(x, (str, int, float))])

        elif isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, list):
                    collected.extend([str(x).strip() for x in vv if isinstance(x, (str, int, float))])

        if collected:
            out[k] = collected

    return out


def pick_random_template_description(
    template_dict: Dict[str, List[str]],
    template_id: str,
    entity: Dict[str, Any]
) -> Optional[str]:
    if template_id not in template_dict or len(template_dict[template_id]) == 0:
        return None

    text = random.choice(template_dict[template_id]).strip()

    # fill placeholder if needed
    if "{kidney_side}" in text:
        belongs_to = entity.get("belongs_to", None)
        kidney_side = KIDNEY_SIDE_MAP.get(str(belongs_to), "this kidney")
        text = text.format(kidney_side=kidney_side)

    return text


# =========================================================
# Split helpers
# =========================================================
def parse_case_id_from_split_path(s: str) -> Optional[str]:
    """
    Example:
      0004/case_00077/image.npy
      0004/case_00153/mask_(3, 512, 512, 69).npz
    Only case_xxxxx is useful.
    """
    if not isinstance(s, str):
        return None
    m = re.search(r"(case_\d+)", s)
    if not m:
        return None
    return m.group(1)


def read_case_ids_from_split_json(split_json_path: str) -> Tuple[List[str], List[str]]:
    sj = read_json(Path(split_json_path))

    def collect_cases(items: Any) -> List[str]:
        out: List[str] = []
        seen = set()

        if not isinstance(items, list):
            return out

        for it in items:
            if not isinstance(it, dict):
                continue

            cand = None
            if "image" in it:
                cand = parse_case_id_from_split_path(str(it["image"]))
            if cand is None and "label" in it:
                cand = parse_case_id_from_split_path(str(it["label"]))

            if cand is not None and cand not in seen:
                seen.add(cand)
                out.append(cand)

        return out

    train_cases = collect_cases(sj.get("train", []))
    test_cases = collect_cases(sj.get("test", []))
    return train_cases, test_cases


# =========================================================
# PNG helpers
# =========================================================
def parse_slice_id_from_filename(name: str) -> Optional[int]:
    m = re.search(r"slice_(\d+)", name)
    if not m:
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


# =========================================================
# Text.json helpers
# =========================================================
def _get_bbox_list_from_template_item(ti: Dict[str, Any]) -> Optional[List[List[int]]]:
    """
    Supports:
      - bbox_2d_list: [[x1,y1,x2,y2], ...]
      - bbox_2d: [x1,y1,x2,y2]
    """
    if not isinstance(ti, dict):
        return None

    if isinstance(ti.get("bbox_2d_list", None), list):
        bbl = ti["bbox_2d_list"]
        cleaned = []
        for b in bbl:
            if isinstance(b, (list, tuple)) and len(b) == 4:
                cleaned.append([int(x) for x in b])
        return cleaned

    b = ti.get("bbox_2d", None)
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return [[int(x) for x in b]]

    return None


def build_template_info_map_from_text_json(tj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    text.json now uses:
      "templates": [...]
    fallback:
      "labels": [...]
    """
    raw = tj.get("templates", None)
    if raw is None:
        raw = tj.get("labels", None)

    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw, list):
        return out

    for item in raw:
        if not isinstance(item, dict):
            continue

        name = item.get("template_name", None)
        if name is None:
            name = item.get("label_name", None)

        if not isinstance(name, str):
            continue

        out[name] = item

    return out


# =========================================================
# Entity helpers
# =========================================================
def build_template_owner_map_for_case(case_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Map:
      template_id -> {
          "entity": entity_dict,
          "template_meta": applicable_template_item
      }

    By construction, one template should map to at most one entity in a case.
    If collision happens, later one is ignored.
    """
    out: Dict[str, Dict[str, Any]] = {}

    entities = case_data.get("entities", [])
    if not isinstance(entities, list):
        return out

    for ent in entities:
        if not isinstance(ent, dict):
            continue

        app = ent.get("applicable_templates", [])
        if not isinstance(app, list):
            continue

        for tp in app:
            if not isinstance(tp, dict):
                continue

            template_id = tp.get("template_id", None)
            if not isinstance(template_id, str):
                continue

            if template_id not in out:
                out[template_id] = {
                    "entity": ent,
                    "template_meta": tp,
                }

    return out


# =========================================================
# Paths
# =========================================================
def get_case_paths(case_id: str) -> Tuple[Path, Path]:
    """
    NPY: root_npy/<case_id>/
    PNG: root_png/<case_id>/png_64slices/
    """
    npy_case_dir = Path(root_npy) / case_id
    png_64_dir = Path(root_png) / case_id / "png_64slices"
    return npy_case_dir, png_64_dir


# =========================================================
# Main generator
# =========================================================
def generate_refseg_vqa_for_cases(
    case_ids: List[str],
    out_json_path: str,
    template_dict_json: str,
    entity_info_json: str,
):
    random.seed(SEED)

    out_json_path = Path(out_json_path)
    template_dict = load_template_dictionary(template_dict_json)
    entity_all = read_json(Path(entity_info_json))

    samples: List[Dict[str, Any]] = []

    # stats
    case_missing_png64_dir = 0
    case_missing_text = 0
    case_missing_entity = 0
    missing_selected = 0
    missing_template_info = 0
    case_missing_png_for_selected = 0
    template_no_desc = 0
    template_no_bbox = 0
    key_not_in_selected = 0
    filtered_out_by_template = 0
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

        if case_id not in entity_all:
            case_missing_entity += 1
            continue

        case_entity_data = entity_all[case_id]

        tj = read_json(text_path)

        selected_slices = tj.get("selected_slices", None)
        if not isinstance(selected_slices, list) or len(selected_slices) == 0:
            missing_selected += 1
            continue
        selected_slices = [int(x) for x in selected_slices]

        template_info_map = build_template_info_map_from_text_json(tj)
        if len(template_info_map) == 0:
            missing_template_info += 1
            continue

        template_owner_map = build_template_owner_map_for_case(case_entity_data)
        if len(template_owner_map) == 0:
            # no applicable template in this case
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

        # iterate over templates that are actually assigned in this case
        for template_id, owner in template_owner_map.items():
            if ALLOWED_TEMPLATE_IDS is not None and template_id not in ALLOWED_TEMPLATE_IDS:
                filtered_out_by_template += 1
                continue

            entity = owner["entity"]
            template_meta = owner["template_meta"]

            # find corresponding template entry in text.json
            ti = template_info_map.get(template_id, None)
            if ti is None:
                continue

            # strongly prefer present templates only
            if not bool(ti.get("present", True)):
                continue

            key_slice = ti.get("key_slice", None)
            if key_slice is None:
                template_no_bbox += 1
                continue
            key_slice = int(key_slice)

            if SKIP_IF_KEYSLICE_NOT_IN_SELECTED and (key_slice not in selected_slices):
                key_not_in_selected += 1
                continue

            bbox_list_512 = _get_bbox_list_from_template_item(ti)
            if SKIP_IF_BBOX_NONE and (bbox_list_512 is None or len(bbox_list_512) == 0):
                template_no_bbox += 1
                continue

            desc = pick_random_template_description(template_dict, template_id, entity)
            if not desc:
                template_no_desc += 1
                continue

            user_content = build_user_prompt_multi_images(selected_slices, desc)
            assistant_content = build_assistant_answer(key_slice, bbox_list_512)

            entity_id = str(entity.get("entity_id", "unknown"))
            sample_id = f"{case_id}__{template_id}__{entity_id}"

            samples.append({
                "id": sample_id,
                "images": images_abs,
                "conversations": [
                    {"from": "human", "value": user_content},
                    {"from": "gpt", "value": assistant_content},
                ],
                "meta": {
                    "case_id": case_id,
                    "template_id": template_id,
                    "question_type": template_meta.get("question_type", None),

                    "entity_id": entity_id,
                    "entity_type": entity.get("entity_type", None),
                    "belongs_to": entity.get("belongs_to", None),

                    "description": desc,
                    "key_slice": key_slice,
                    "bbox_list_512": bbox_list_512,
                    "num_images": len(images_abs),

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
    print(f"  case_missing_entity_info          = {case_missing_entity}")
    print(f"  missing_selected_slices           = {missing_selected}")
    print(f"  missing_template_info             = {missing_template_info}")
    print(f"  case_missing_png_for_selected     = {case_missing_png_for_selected} (SKIP_IF_MISSING_PNG={SKIP_IF_MISSING_PNG})")
    print(f"  template_no_description           = {template_no_desc}")
    print(f"  template_no_bbox/key_slice/list   = {template_no_bbox}")
    print(f"  key_slice_not_in_selected         = {key_not_in_selected}")
    print(f"  filtered_out_by_template          = {filtered_out_by_template}")
    print(f"  kept_samples                      = {kept}")


def main():
    train_cases, test_cases = read_case_ids_from_split_json(split_file)

    print("========== Split Sizes ==========")
    print(f"train cases: {len(train_cases)} from {split_file}")
    print(f"test  cases: {len(test_cases)} from {split_file}")
    print("root_npy:", root_npy)
    print("root_png:", root_png)
    print("template_path:", template_path)
    print("entity_info_file:", entity_info_file)
    print("ALLOWED_TEMPLATE_IDS:", sorted(list(ALLOWED_TEMPLATE_IDS)) if ALLOWED_TEMPLATE_IDS is not None else None)
    print("================================\n")

    print("========== Generate TRAIN VQA ==========")
    generate_refseg_vqa_for_cases(
        case_ids=train_cases,
        out_json_path=OUT_TRAIN_JSON,
        template_dict_json=template_path,
        entity_info_json=entity_info_file,
    )

    print("\n========== Generate TEST VQA ==========")
    generate_refseg_vqa_for_cases(
        case_ids=test_cases,
        out_json_path=OUT_TEST_JSON,
        template_dict_json=template_path,
        entity_info_json=entity_info_file,
    )


if __name__ == "__main__":
    main()
