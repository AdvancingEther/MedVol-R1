#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import traceback
from pathlib import Path

import numpy as np
from scipy import sparse


# =========================================================
# Global config
# =========================================================
KITS_NPY_ROOT = os.environ.get('MEDVOL_STEP0_CVT_TO_KITS_M3D_KITS_NPY_ROOT', 'data/kits23/kits23_npy')
# 如果你的实际目录是 dataset_npy，就改成：
# KITS_NPY_ROOT = "data/kits23/dataset_npy"

ENTITY_WITH_TEMPLATES_JSON = os.environ.get('MEDVOL_STEP0_CVT_TO_KITS_M3D_ENTITY_WITH_TEMPLATES_JSON', 'data/kits23/final_vqa_gen/entity_with_templates.json')

OUT_ROOT = os.environ.get('MEDVOL_STEP0_CVT_TO_KITS_M3D_OUT_ROOT', 'data/kits23/kits23_npy_m3d')

IMAGE_FILENAME = "imaging_wc40_ww400_uint8.npy"
CASE_PREFIX = "case_"
INSTANCE_DIRNAME = "instances"

OVERWRITE = False
VERBOSE = True
SKIP_IF_NO_TEMPLATE = False  # True: 没有任何 template 命中的 case 直接跳过；False: 仍然输出全零 mask


# =========================================================
# Fixed template vocabulary (14)
# =========================================================
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
NUM_TEMPLATES = len(TEMPLATE_ORDER)


# =========================================================
# Helpers
# =========================================================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def natural_case_sort_key(case_name: str):
    try:
        return int(case_name.split("_")[-1])
    except Exception:
        return case_name


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj):
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def find_case_dirs(root: Path):
    case_dirs = []
    if not root.exists():
        return case_dirs
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith(CASE_PREFIX):
            case_dirs.append(p)
    case_dirs.sort(key=lambda x: natural_case_sort_key(x.name))
    return case_dirs


def get_image_path(case_dir: Path) -> Path:
    return case_dir / IMAGE_FILENAME


def get_mask_path_from_entity(entity: dict) -> Path:
    return Path(entity["mask_file"])


def convert_image_to_m3d(img_3d: np.ndarray) -> np.ndarray:
    """
    Input:
      img_3d shape = (axis0, axis1, axis2)
      axis0 is slice axis (top->bottom)
    Output:
      shape = (1, H, W, Z) = (1, axis1, axis2, axis0)
    """
    assert img_3d.ndim == 3, f"Expected 3D image, got {img_3d.shape}"
    img_h_w_z = np.transpose(img_3d, (1, 2, 0))   # (H,W,Z)
    img_out = img_h_w_z[None, ...]                 # (1,H,W,Z)
    return img_out.astype(np.float32)


def convert_mask_to_hwz(mask_3d: np.ndarray) -> np.ndarray:
    """
    Input:
      mask_3d shape = (axis0, axis1, axis2)
    Output:
      shape = (H, W, Z) = (axis1, axis2, axis0)
    """
    assert mask_3d.ndim == 3, f"Expected 3D mask, got {mask_3d.shape}"
    return np.transpose(mask_3d, (1, 2, 0))


def case_has_any_template(case_data: dict) -> bool:
    for ent in case_data.get("entities", []):
        if len(ent.get("applicable_templates", [])) > 0:
            return True
    return False


# =========================================================
# Core per-case conversion
# =========================================================
def convert_one_case(case_dir: Path, case_data: dict, out_root: Path):
    case_id = case_dir.name
    out_case_dir = out_root / case_id

    if out_case_dir.exists() and (not OVERWRITE):
        return {
            "case_id": case_id,
            "case_dir": str(out_case_dir),
            "status": "skip_exists",
        }

    image_path = get_image_path(case_dir)
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image file: {image_path}")

    if SKIP_IF_NO_TEMPLATE and (not case_has_any_template(case_data)):
        return {
            "case_id": case_id,
            "case_dir": str(out_case_dir),
            "status": "skip_no_template",
        }

    # -----------------------------------------------------
    # load and convert image
    # -----------------------------------------------------
    img_3d = np.load(str(image_path))  # expected (axis0,axis1,axis2)
    img_out = convert_image_to_m3d(img_3d)  # (1,H,W,Z)
    _, H, W, Z = img_out.shape
    HWZ = H * W * Z

    # -----------------------------------------------------
    # init template mask channels
    # shape = (14, H, W, Z)
    # -----------------------------------------------------
    template_dense = np.zeros((NUM_TEMPLATES, H, W, Z), dtype=np.uint8)

    filled_templates = {}
    collisions = []

    # -----------------------------------------------------
    # fill template channels from entities
    # -----------------------------------------------------
    entities = case_data.get("entities", [])
    for ent in entities:
        mask_file = ent.get("mask_file")
        if mask_file is None:
            continue

        applicable_templates = ent.get("applicable_templates", [])
        if len(applicable_templates) == 0:
            continue

        mask_path = get_mask_path_from_entity(ent)
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing entity mask file: {mask_path}")

        mask_3d = np.load(str(mask_path))  # expected (axis0,axis1,axis2)
        mask_3d = (mask_3d > 0).astype(np.uint8)
        mask_hwz = convert_mask_to_hwz(mask_3d)  # (H,W,Z)

        if mask_hwz.shape != (H, W, Z):
            raise ValueError(
                f"Mask shape mismatch in {case_id}: "
                f"mask {mask_hwz.shape} vs image {(H, W, Z)} from {mask_path}"
            )

        for tp in applicable_templates:
            template_id = tp["template_id"]
            if template_id not in TEMPLATE_TO_INDEX:
                continue

            ch = TEMPLATE_TO_INDEX[template_id]

            if template_id in filled_templates:
                prev = filled_templates[template_id]
                collisions.append({
                    "template_id": template_id,
                    "previous_entity_id": prev,
                    "current_entity_id": ent.get("entity_id"),
                })
                template_dense[ch] = np.maximum(template_dense[ch], mask_hwz)
            else:
                template_dense[ch] = mask_hwz
                filled_templates[template_id] = ent.get("entity_id")

    # -----------------------------------------------------
    # convert dense -> sparse (14, HWZ)
    # flatten rule: z fastest because shape is (H,W,Z)
    # -----------------------------------------------------
    dense_2d = template_dense.reshape(NUM_TEMPLATES, HWZ)
    sparse_mask = sparse.csr_matrix(dense_2d, dtype=np.uint8)

    # -----------------------------------------------------
    # save outputs
    # -----------------------------------------------------
    ensure_dir(str(out_case_dir))

    out_img_path = out_case_dir / "image.npy"
    out_mask_path = out_case_dir / f"mask_({NUM_TEMPLATES},{H},{W},{Z}).npz"
    out_meta_path = out_case_dir / "meta.json"

    np.save(str(out_img_path), img_out)
    sparse.save_npz(str(out_mask_path), sparse_mask)

    meta = {
        "case_id": case_id,
        "source_case_dir": str(case_dir),
        "source_image": str(image_path),
        "output_image": str(out_img_path),
        "output_mask": str(out_mask_path),

        "image_shape": list(img_out.shape),            # (1,H,W,Z)
        "mask_sparse_shape": list(sparse_mask.shape),  # (14, HWZ)
        "mask_dense_shape": [NUM_TEMPLATES, H, W, Z],
        "mask_nnz": int(sparse_mask.nnz),

        "template_order": TEMPLATE_ORDER,
        "template_to_index": TEMPLATE_TO_INDEX,

        "filled_templates": filled_templates,   # template_id -> entity_id
        "num_filled_templates": len(filled_templates),

        "collisions": collisions,
        "num_collisions": len(collisions),

        "note": (
            "Template-channel M3D-style conversion. "
            "Image shape is (1,H,W,Z). Sparse mask shape is (N_templates, H*W*Z). "
            "Flatten order uses z-fastest: linear=(y*W+x)*Z+z."
        ),
    }
    save_json(str(out_meta_path), meta)

    return {
        "case_id": case_id,
        "case_dir": str(out_case_dir),
        "status": "ok",
        "num_filled_templates": len(filled_templates),
        "num_collisions": len(collisions),
        "mask_nnz": int(sparse_mask.nnz),
    }


# =========================================================
# Main
# =========================================================
def main():
    kits_root = Path(KITS_NPY_ROOT)
    out_root = Path(OUT_ROOT)

    assert kits_root.exists(), f"KITS_NPY_ROOT not found: {kits_root}"
    assert os.path.exists(ENTITY_WITH_TEMPLATES_JSON), f"ENTITY_WITH_TEMPLATES_JSON not found: {ENTITY_WITH_TEMPLATES_JSON}"

    ensure_dir(str(out_root))

    entity_data = load_json(ENTITY_WITH_TEMPLATES_JSON)
    case_dirs = find_case_dirs(kits_root)

    print(f"[Info] Found {len(case_dirs)} case folders under: {kits_root}")
    print(f"[Info] Template channels: {NUM_TEMPLATES}")

    summary = {
        "kits_root": str(kits_root),
        "entity_with_templates_json": ENTITY_WITH_TEMPLATES_JSON,
        "out_root": str(out_root),
        "num_cases_scanned": len(case_dirs),
        "num_templates": NUM_TEMPLATES,
        "template_order": TEMPLATE_ORDER,
        "ok": 0,
        "skip_exists": 0,
        "skip_no_template": 0,
        "fail": 0,
        "items": [],
        "fails": [],
    }

    index_items = []

    for case_dir in case_dirs:
        case_id = case_dir.name

        if case_id not in entity_data:
            summary["fail"] += 1
            summary["fails"].append({
                "case_id": case_id,
                "error": f"{case_id} not found in entity_with_templates.json",
            })
            print(f"[Fail] {case_id}: missing in entity_with_templates.json")
            continue

        try:
            ret = convert_one_case(case_dir, entity_data[case_id], out_root)
            summary["items"].append(ret)

            if ret["status"] == "ok":
                summary["ok"] += 1
            elif ret["status"] == "skip_exists":
                summary["skip_exists"] += 1
            elif ret["status"] == "skip_no_template":
                summary["skip_no_template"] += 1

            index_items.append({
                "case_id": case_id,
                "case_dir": ret["case_dir"],
                "status": ret["status"],
            })

            if VERBOSE:
                print(f"[{ret['status']}] {case_id}")

        except Exception as e:
            summary["fail"] += 1
            err = {
                "case_id": case_id,
                "error": repr(e),
                "traceback": traceback.format_exc()[-4000:],
            }
            summary["fails"].append(err)
            print(f"[Fail] {case_id}: {repr(e)}")

    save_json(str(out_root / "index.json"), {
        "template_order": TEMPLATE_ORDER,
        "items": index_items,
    })
    save_json(str(out_root / "convert_summary.json"), summary)

    print("\n========== Done ==========")
    print(f"OK            : {summary['ok']}")
    print(f"Skip exists   : {summary['skip_exists']}")
    print(f"Skip no tpl   : {summary['skip_no_template']}")
    print(f"Fail          : {summary['fail']}")
    print(f"Out root      : {out_root}")
    print("==========================")


if __name__ == "__main__":
    main()
