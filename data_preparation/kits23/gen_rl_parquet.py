#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KiTS23 -> Verl/Easy-R1 parquet generator (template-based, current data layout)

Compared with the old CT-Org version, this script:
1) Uses split_file (0004.json) to get train/test case ids
2) Uses entity_with_templates.json to know which template_id is valid for each case
3) Uses template whitelist to keep only selected template_ids
4) Uses root_png/<case_id>/png_64slices/text.json to read:
   - selected_slices
   - per-template key_slice
   - per-template bbox_2d_list
5) Uses root_npy/<case_id>/image.npy and mask_(14,...).npz
6) Writes reward-friendly answer JSON containing:
   - template_id
   - template_index
   - gt_key_slice
   - gt_bbox_2d_list_512
   - bbox_list_512 (compat)
   - image_rel_path / mask_rel_path
"""

import os
import json
import re
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from datasets import Dataset, Sequence
from datasets import Image as ImageData


# =========================================================
# Config (edit here)
# =========================================================
template_path = os.environ.get('MEDVOL_GEN_RL_PARQUET_TEMPLATE_PATH', 'data_preparation/kits23/template.json')
entity_info_file = os.environ.get('MEDVOL_GEN_RL_PARQUET_ENTITY_INFO_FILE', 'data/kits23/final_vqa_gen/entity_with_templates.json')

root_npy = os.environ.get('MEDVOL_GEN_RL_PARQUET_ROOT_NPY', 'data/kits23/kits23_npy_m3d')
root_png = os.environ.get('MEDVOL_GEN_RL_PARQUET_ROOT_PNG', 'data/kits23/kits23_png_m3d')

split_file = os.environ.get('MEDVOL_GEN_RL_PARQUET_SPLIT_FILE', 'data/kits23/split.json')

out_dir = os.environ.get('MEDVOL_GEN_RL_PARQUET_OUT_DIR', 'data/kits23/verl_parquet')
dataset_name_train = "kits23_refseg_verl_train"
dataset_name_test = "kits23_refseg_verl_test"

RANDOM_SEED = 42

STRICT_REQUIRE_ALL_PNG = True
REQUIRE_K_IMAGES: Optional[int] = None   # None = allow variable number of slices (<64 is fine)
LIMIT = -1  # -1 means no limit

GENERATE_TRAIN = True
GENERATE_TEST = True

# If true, skip emitting samples whose text.json doesn't provide bbox for that template
SKIP_IF_NO_GT_BBOX = True

# deterministic per-(case,template) emission
DETERMINISTIC_PER_SAMPLE = True

# known template order (must match your M3D conversion)
TEMPLATE_ORDER = [
    "kidney_only_with_tumor",
    "kidney_only_with_cyst",
    "kidney_only_with_both_tumor_and_cyst",
    "kidney_only_healthy",

    "tumor_only_global",
    "tumor_only_in_kidney",
    "tumor_largest_in_kidney",
    "tumor_highest_in_kidney",
    "tumor_lowest_in_kidney",

    "cyst_only_global",
    "cyst_only_in_kidney",
    "cyst_largest_in_kidney",
    "cyst_highest_in_kidney",
    "cyst_lowest_in_kidney",
]
TEMPLATE_TO_INDEX = {t: i for i, t in enumerate(TEMPLATE_ORDER)}

# whitelist: keep only these template ids
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

# left/right prior
KIDNEY_SIDE_MAP = {
    "kidney_1": "the right kidney",
    "kidney_2": "the left kidney",
}


# =========================================================
# Helpers
# =========================================================
def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(str(x).strip())
    except Exception:
        return default


def _read_png_as_hf_image_path(p: str) -> Dict[str, Optional[bytes]]:
    # store PATH only (avoid arrow offset overflow)
    return {"path": os.path.abspath(p), "bytes": None}


def _slice_path(case_png_dir: Path, z: int) -> str:
    return str(case_png_dir / f"slice_{int(z):04d}.png")


def _resolve_mask_npz(case_npy_dir: Path) -> Optional[Path]:
    cands = sorted(case_npy_dir.glob("mask_*.npz"))
    if not cands:
        return None
    # strongly prefer the template mask
    for c in cands:
        if c.name.startswith("mask_(14,"):
            return c
    return cands[0]


def _infer_volume_shape_from_image_npy(image_npy: Path) -> Optional[Tuple[int, int, int]]:
    """
    image.npy is expected current M3D style:
      (1,H,W,Z)
    Return (Z,H,W)
    Also keep a few fallbacks.
    """
    if not image_npy.exists():
        return None

    import numpy as np
    arr = np.load(str(image_npy), mmap_mode="r")
    arr = np.asarray(arr)

    if arr.ndim == 4:
        # expected: (1,H,W,Z)
        if arr.shape[0] == 1:
            return (int(arr.shape[3]), int(arr.shape[1]), int(arr.shape[2]))
        # fallback: (1,Z,H,W)
        return (int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3]))

    if arr.ndim == 3:
        # fallback
        if arr.shape[0] == 512 and arr.shape[1] == 512:
            return (int(arr.shape[2]), int(arr.shape[0]), int(arr.shape[1]))
        if arr.shape[1] == 512 and arr.shape[2] == 512:
            return (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
        return (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))

    return None


def load_template_dictionary(p: str) -> Dict[str, List[str]]:
    """
    Supports:
    1) flat format:
       { "tumor_only_global": ["...", "..."] }

    2) nested format:
       { "tumor_only_global": { "direct": [...], "descriptive": [...] } }
    """
    if not p or (not os.path.exists(p)):
        return {}

    d = _load_json(p)
    out: Dict[str, List[str]] = {}
    if not isinstance(d, dict):
        return out

    for k, v in d.items():
        if not isinstance(k, str):
            continue

        collected: List[str] = []

        if isinstance(v, list):
            collected.extend([str(x).strip() for x in v if str(x).strip()])

        elif isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, list):
                    collected.extend([str(x).strip() for x in vv if str(x).strip()])

        if collected:
            out[k.strip()] = collected

    return out


def pick_random_template_description(
    template_dict: Dict[str, List[str]],
    template_id: str,
    entity: Dict[str, Any],
    rng: random.Random,
) -> str:
    pool = template_dict.get(template_id, None)
    if not pool:
        return template_id

    text = rng.choice(pool).strip()

    if "{kidney_side}" in text:
        belongs_to = str(entity.get("belongs_to", ""))
        kidney_side = KIDNEY_SIDE_MAP.get(belongs_to, "this kidney")
        text = text.format(kidney_side=kidney_side)

    return text


def _deterministic_hash_prob(key: str) -> float:
    h = 0
    for ch in key:
        h = (h * 131 + ord(ch)) % 1000003
    return (h % 1000000) / 1000000.0


# =========================================================
# Split helpers
# =========================================================
def parse_case_id_from_split_path(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    m = re.search(r"(case_\d+)", s)
    if not m:
        return None
    return m.group(1)


def read_case_ids_from_split_json(split_json_path: str) -> Tuple[List[str], List[str]]:
    sj = _load_json(split_json_path)

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
def _resolve_png_case_dir(root_png_dir: Path, case_id: str) -> Optional[Path]:
    """
    Return the directory that actually contains slice_*.png.
    Tries:
      1) root_png/case_id/png_64slices
      2) any subdir under root_png/case_id that contains slice_*.png
      3) root_png/case_id itself (if contains slice_*.png)
    """
    case_root = root_png_dir / case_id
    if not case_root.exists():
        return None

    cand = case_root / "png_64slices"
    if cand.exists() and any(cand.glob("slice_*.png")):
        return cand

    for d in sorted([p for p in case_root.iterdir() if p.is_dir()]):
        if any(d.glob("slice_*.png")):
            return d

    if any(case_root.glob("slice_*.png")):
        return case_root

    return None


def _load_selected_slices(case_png_dir: Path, case_root_dir: Path) -> List[int]:
    """
    Prefer text.json["selected_slices"] from:
      1) case_png_dir/text.json
      2) case_root_dir/text.json
    Else scan case_png_dir for slice_*.png
    """
    for tj in [case_png_dir / "text.json", case_root_dir / "text.json"]:
        if tj.exists():
            try:
                meta = _load_json(str(tj))
                ss = meta.get("selected_slices", None)
                if isinstance(ss, list) and len(ss) > 0:
                    tmp = []
                    for x in ss:
                        iv = _safe_int(x, None)
                        if iv is not None:
                            tmp.append(int(iv))
                    tmp = sorted(list(set(tmp)))
                    if len(tmp) > 0:
                        return tmp
            except Exception:
                pass

    out = []
    for p in case_png_dir.glob("slice_*.png"):
        m = re.search(r"slice_(\d+)\.png$", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(list(set(out)))


# =========================================================
# text.json / entity helpers
# =========================================================
def _pick_text_json_path(case_png_dir: Path, case_root_dir: Path) -> Optional[Path]:
    for tj in [case_png_dir / "text.json", case_root_dir / "text.json"]:
        if tj.exists():
            return tj
    return None


def _get_bbox_list_from_template_item(ti: Dict[str, Any]) -> List[List[int]]:
    if not isinstance(ti, dict):
        return []

    bbl = ti.get("bbox_2d_list", [])
    if not isinstance(bbl, list):
        return []

    out_boxes: List[List[int]] = []
    for b in bbl:
        if not (isinstance(b, list) and len(b) == 4):
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in b]
        except Exception:
            continue

        x1 = max(0, min(511, x1))
        y1 = max(0, min(511, y1))
        x2 = max(0, min(511, x2))
        y2 = max(0, min(511, y2))

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 <= x1:
            x2 = min(511, x1 + 1)
        if y2 <= y1:
            y2 = min(511, y1 + 1)

        out_boxes.append([x1, y1, x2, y2])

    return out_boxes


def build_template_info_map_from_text_json(text_json_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """
    text.json uses:
      "templates": [...]
    fallback:
      "labels": [...]
    """
    if text_json_path is None or (not text_json_path.exists()):
        return {}

    try:
        obj = _load_json(str(text_json_path))
    except Exception:
        return {}

    raw = obj.get("templates", None)
    if raw is None:
        raw = obj.get("labels", None)

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


def build_template_owner_map_for_case(case_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Map:
      template_id -> {
          "entity": entity_dict,
          "template_meta": applicable_template_item
      }

    One template should map to at most one entity in a case.
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

            if ALLOWED_TEMPLATE_IDS is not None and template_id not in ALLOWED_TEMPLATE_IDS:
                continue

            if template_id not in out:
                out[template_id] = {
                    "entity": ent,
                    "template_meta": tp,
                }

    return out


# =========================================================
# Generator
# =========================================================
def generate_data(case_ids: List[str]) -> Iterator[Dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)

    root_png_dir = Path(root_png)
    root_npy_dir = Path(root_npy)
    template_dict = load_template_dictionary(template_path)
    entity_all = _load_json(entity_info_file)

    # stats
    total_candidates = len(case_ids)
    used_cases = 0
    skipped_missing_png_dir = 0
    skipped_missing_npy_dir = 0
    skipped_missing_files = 0
    skipped_bad_selected = 0
    skipped_missing_png = 0
    skipped_no_text_json = 0
    skipped_no_entity = 0
    skipped_no_template_owner = 0
    skipped_no_template_info = 0
    skipped_no_gt_bbox = 0
    skipped_key_not_in_selected = 0
    emitted = 0

    for case_id in case_ids:
        if LIMIT > 0 and emitted >= LIMIT:
            break

        case_root_png_dir = root_png_dir / case_id
        case_npy_dir = root_npy_dir / case_id
        case_png_dir = _resolve_png_case_dir(root_png_dir, case_id)

        if case_png_dir is None or (not case_png_dir.exists()):
            skipped_missing_png_dir += 1
            continue

        if not case_npy_dir.exists():
            skipped_missing_npy_dir += 1
            continue

        image_npy = case_npy_dir / "image.npy"
        mask_npz = _resolve_mask_npz(case_npy_dir)
        if (not image_npy.exists()) or (mask_npz is None) or (not mask_npz.exists()):
            skipped_missing_files += 1
            continue

        vol_shape = _infer_volume_shape_from_image_npy(image_npy)
        if vol_shape is None:
            skipped_missing_files += 1
            continue
        Z, H, W = vol_shape

        selected_slices = _load_selected_slices(case_png_dir, case_root_png_dir)
        if len(selected_slices) == 0:
            skipped_bad_selected += 1
            continue
        if REQUIRE_K_IMAGES is not None and len(selected_slices) != int(REQUIRE_K_IMAGES):
            skipped_bad_selected += 1
            continue

        slice_paths = [_slice_path(case_png_dir, z) for z in selected_slices]
        exists = [os.path.exists(p) for p in slice_paths]
        if STRICT_REQUIRE_ALL_PNG and (not all(exists)):
            skipped_missing_png += 1
            continue

        images: List[Dict[str, Optional[bytes]]] = []
        kept_slices: List[int] = []
        for z, p in zip(selected_slices, slice_paths):
            if not os.path.exists(p):
                continue
            images.append(_read_png_as_hf_image_path(p))
            kept_slices.append(int(z))

        if len(images) == 0:
            skipped_missing_png += 1
            continue

        text_json_path = _pick_text_json_path(case_png_dir, case_root_png_dir)
        if text_json_path is None:
            skipped_no_text_json += 1
            continue

        if case_id not in entity_all:
            skipped_no_entity += 1
            continue

        case_entity_data = entity_all[case_id]
        template_owner_map = build_template_owner_map_for_case(case_entity_data)
        if len(template_owner_map) == 0:
            skipped_no_template_owner += 1
            continue

        template_info_map = build_template_info_map_from_text_json(text_json_path)
        if len(template_info_map) == 0:
            skipped_no_template_info += 1
            continue

        used_cases += 1

        # emit per (case, template)
        for template_id, owner in template_owner_map.items():
            entity = owner["entity"]
            template_meta = owner["template_meta"]

            ti = template_info_map.get(template_id, None)
            if ti is None:
                continue

            if ti.get("present", True) is False:
                continue

            gt_key_slice = ti.get("key_slice", None)
            if gt_key_slice is None:
                skipped_no_gt_bbox += 1
                continue
            try:
                gt_key_slice = int(gt_key_slice)
            except Exception:
                skipped_no_gt_bbox += 1
                continue

            if gt_key_slice not in kept_slices:
                skipped_key_not_in_selected += 1
                continue

            gt_bbox_2d_list_512 = _get_bbox_list_from_template_item(ti)
            if SKIP_IF_NO_GT_BBOX and len(gt_bbox_2d_list_512) == 0:
                skipped_no_gt_bbox += 1
                continue

            if DETERMINISTIC_PER_SAMPLE:
                u = _deterministic_hash_prob(f"{case_id}::{template_id}::{RANDOM_SEED}")
                local_seed = int(u * 10_000_000) + 12345
                local_rng = random.Random(local_seed)
            else:
                local_rng = rng

            desc = pick_random_template_description(
                template_dict=template_dict,
                template_id=template_id,
                entity=entity,
                rng=local_rng,
            )

            template_index = TEMPLATE_TO_INDEX.get(template_id, None)

            answer_dict = {
                "id": f"{case_id}__{template_id}",
                "case_id": case_id,

                "template_id": template_id,
                "template_index": template_index,
                "question_type": template_meta.get("question_type", None),

                "entity_id": entity.get("entity_id", None),
                "entity_type": entity.get("entity_type", None),
                "belongs_to": entity.get("belongs_to", None),

                "selected_slices": kept_slices,
                "volume_shape": [int(Z), int(H), int(W)],
                "canon_size_xy": [512, 512],

                # reward paths: keep RELATIVE inside root_npy
                "image_rel_path": str(Path(case_id) / "image.npy"),
                "mask_rel_path": str(Path(case_id) / Path(mask_npz.name)),

                # reward targets
                "gt_key_slice": gt_key_slice,
                "gt_bbox_2d_list_512": gt_bbox_2d_list_512,
                # compatibility fallback
                "bbox_list_512": gt_bbox_2d_list_512,

                "extra": {
                    "split_file": os.path.abspath(split_file),
                    "png_case_dir": str(Path(case_id) / "png_64slices"),
                    "text_json_path": str(text_json_path),
                    "allowed_template_ids": sorted(list(ALLOWED_TEMPLATE_IDS)) if ALLOWED_TEMPLATE_IDS is not None else None,
                },
            }

            yield {
                "images": images,
                "problem": str(desc),
                "answer": json.dumps(answer_dict, ensure_ascii=False),
            }
            emitted += 1

    print("========== KiTS23 parquet generation stats ==========")
    print(f"candidate_cases_total      : {total_candidates}")
    print(f"used_cases                : {used_cases}")
    print(f"skipped_missing_png_dir    : {skipped_missing_png_dir}")
    print(f"skipped_missing_npy_dir    : {skipped_missing_npy_dir}")
    print(f"skipped_missing_files      : {skipped_missing_files}")
    print(f"skipped_bad_selected       : {skipped_bad_selected}")
    print(f"skipped_missing_png        : {skipped_missing_png}")
    print(f"skipped_no_text_json       : {skipped_no_text_json}")
    print(f"skipped_no_entity          : {skipped_no_entity}")
    print(f"skipped_no_template_owner  : {skipped_no_template_owner}")
    print(f"skipped_no_template_info   : {skipped_no_template_info}")
    print(f"skipped_no_gt_bbox         : {skipped_no_gt_bbox}")
    print(f"skipped_key_not_in_selected: {skipped_key_not_in_selected}")
    print(f"total_samples_emitted      : {emitted}")


# =========================================================
# Main
# =========================================================
def main():
    Path(out_dir).mkdir(parents=True, exist_ok=True)

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

    if GENERATE_TRAIN:
        print("========== Generate TRAIN parquet ==========")
        ds_train = Dataset.from_generator(lambda: generate_data(train_cases)).cast_column(
            "images", Sequence(ImageData())
        )
        print(ds_train)

        out_path_train = os.path.join(out_dir, f"{dataset_name_train}.parquet")
        ds_train.to_parquet(out_path_train)
        print(f"[OK] Wrote: {out_path_train}")

    if GENERATE_TEST:
        print("\n========== Generate TEST parquet ==========")
        ds_test = Dataset.from_generator(lambda: generate_data(test_cases)).cast_column(
            "images", Sequence(ImageData())
        )
        print(ds_test)

        out_path_test = os.path.join(out_dir, f"{dataset_name_test}.parquet")
        ds_test.to_parquet(out_path_test)
        print(f"[OK] Wrote: {out_path_test}")


if __name__ == "__main__":
    main()
