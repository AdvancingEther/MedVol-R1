#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import SimpleITK as sitk


# =========================================================
# Global config
# =========================================================
DATASET_ROOT = Path(os.environ.get('MEDVOL_KITS23_RAW_ROOT', 'data/kits23/dataset'))
OUT_ROOT = Path(os.environ.get('MEDVOL_KITS23_INSTANCE_ROOT', 'data/kits23/kits23_npy'))

CASE_PREFIX = "case_"
INSTANCE_DIRNAME = "instances"

# 直接保存加窗后的 uint8
IMAGING_SAVE_MODE = "uint8_window"
WINDOW_CENTER = 40
WINDOW_WIDTH = 400

# 仅统一 in-plane 到固定大小，Z 严格不变
TARGET_INPLANE_SIZE = (512, 512)  # (axis0, axis1) after transpose -> (X, Y)

# 只保留 annotation=1
ONLY_ANNOTATION_ID = 1

INSTANCE_BINARIZE = True
SKIP_EMPTY_INSTANCE = False
OVERWRITE = False
WRITE_INDEX_JSON = True
VERBOSE = True

# 多线程
NUM_WORKERS = 8

PAT = re.compile(r"^(kidney|tumor|cyst)_instance-(\d+)_annotation-(\d+)\.nii\.gz$")


# =========================================================
# Helpers
# =========================================================
def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def natural_key(text: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def iter_cases(dataset_root: Path) -> List[Path]:
    cases = []
    if not dataset_root.exists():
        return cases
    for p in dataset_root.iterdir():
        if p.is_dir() and p.name.startswith(CASE_PREFIX):
            cases.append(p)
    cases.sort(key=lambda p: natural_key(p.name))
    return cases


def read_sitk_transposed(path: Path) -> np.ndarray:
    """
    Current internal convention in this script:
      sitk.GetArrayFromImage -> (Z, Y, X)
      transpose(2,1,0)      -> (X, Y, Z)
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    arr = np.transpose(arr, (2, 1, 0))
    return arr


def resize_2d_xy_slice(slice_xy: np.ndarray, target_hw: Tuple[int, int], is_mask: bool) -> np.ndarray:
    """
    Resize one 2D slice from (X, Y) -> (target_x, target_y)

    We explicitly do 2D resizing slice-by-slice so Z is guaranteed unchanged.
    """
    src_x, src_y = slice_xy.shape
    tgt_x, tgt_y = target_hw

    if (src_x, src_y) == (tgt_x, tgt_y):
        return slice_xy

    # SimpleITK 2D image expects numpy layout (Y, X)
    slice_yx = np.transpose(slice_xy, (1, 0))  # (Y, X)
    img2d = sitk.GetImageFromArray(slice_yx)

    old_size = list(img2d.GetSize())  # [X, Y]
    new_size = [int(tgt_x), int(tgt_y)]

    old_spacing = list(img2d.GetSpacing())
    new_spacing = [
        old_spacing[0] * old_size[0] / new_size[0],
        old_spacing[1] * old_size[1] / new_size[1],
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(img2d.GetOrigin())
    resampler.SetOutputDirection(img2d.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    out_img2d = resampler.Execute(img2d)
    out_yx = sitk.GetArrayFromImage(out_img2d)   # (Y, X)
    out_xy = np.transpose(out_yx, (1, 0))        # back to (X, Y)
    return out_xy


def resize_inplane_numpy(arr_xyz: np.ndarray, target_hw: Tuple[int, int], is_mask: bool) -> np.ndarray:
    """
    Resize only in-plane dimensions for array shaped (X, Y, Z).
    Z is guaranteed unchanged because we resize each slice independently.
    """
    assert arr_xyz.ndim == 3, f"Expected 3D array, got shape={arr_xyz.shape}"

    src_x, src_y, src_z = arr_xyz.shape
    tgt_x, tgt_y = target_hw

    if (src_x, src_y) == (tgt_x, tgt_y):
        return arr_xyz

    if is_mask:
        out = np.zeros((tgt_x, tgt_y, src_z), dtype=np.uint8)
    else:
        out = np.zeros((tgt_x, tgt_y, src_z), dtype=np.float32)

    for z in range(src_z):
        slice_xy = arr_xyz[:, :, z]
        out[:, :, z] = resize_2d_xy_slice(slice_xy, target_hw, is_mask=is_mask)

    return out


def apply_window(image: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    min_value = window_center - window_width / 2.0
    max_value = window_center + window_width / 2.0
    x = np.clip(image, min_value, max_value)
    x = (x - min_value) / (max_value - min_value + 1e-8)
    return x


def to_uint8(x01: np.ndarray) -> np.ndarray:
    return (np.clip(x01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def get_image_meta(path: Path):
    img = sitk.ReadImage(str(path))
    return {
        "size_xyz": list(img.GetSize()),
        "spacing_xyz": [float(v) for v in img.GetSpacing()],
        "origin_xyz": [float(v) for v in img.GetOrigin()],
        "direction_3x3_flat": [float(v) for v in img.GetDirection()],
        "sitk_pixel_id": int(img.GetPixelID()),
        "sitk_pixel_id_type": img.GetPixelIDTypeAsString(),
    }


def maybe_save_npy(path: Path, array: np.ndarray):
    if path.exists() and (not OVERWRITE):
        return
    np.save(path, array)


# =========================================================
# Per-case worker
# =========================================================
def process_case(case_dir: Path):
    case_name = case_dir.name
    imaging_path = case_dir / "imaging.nii.gz"
    instances_dir = case_dir / INSTANCE_DIRNAME

    out_case_dir = OUT_ROOT / case_name
    out_inst_dir = out_case_dir / INSTANCE_DIRNAME
    safe_mkdir(out_inst_dir)

    index = {
        "case": case_name,
        "convention": {
            "reader": "SimpleITK",
            "raw_shape_from_sitk_array": "(Z,Y,X)",
            "transpose_applied": [2, 1, 0],
            "final_shape": "(axis0,axis1,axis2)",
            "axis0_meaning": "top_to_bottom_confirmed_by_user",
        },
        "resize": {
            "enabled": True,
            "target_inplane_axis0_axis1": list(TARGET_INPLANE_SIZE),
            "z_axis_unchanged": True,
            "resize_mode": "slice_wise_2d",
            "image_interpolator": "linear",
            "mask_interpolator": "nearest",
        },
        "annotation_filter": {
            "enabled": True,
            "only_annotation_id": ONLY_ANNOTATION_ID,
        },
        "imaging": {},
        "instances": [],
    }

    try:
        # -----------------------------
        # imaging
        # -----------------------------
        if not imaging_path.exists():
            if VERBOSE:
                print(f"[Skip] {case_name}: missing imaging.nii.gz")
            return "skip", case_name

        img_arr = read_sitk_transposed(imaging_path).astype(np.float32)
        raw_img_shape = list(img_arr.shape)

        img_arr = resize_inplane_numpy(
            img_arr,
            target_hw=TARGET_INPLANE_SIZE,
            is_mask=False,
        )
        img01 = apply_window(img_arr, WINDOW_CENTER, WINDOW_WIDTH)
        img_save = to_uint8(img01)

        out_img_name = f"imaging_wc{int(WINDOW_CENTER)}_ww{int(WINDOW_WIDTH)}_uint8.npy"
        out_img_path = out_case_dir / out_img_name
        maybe_save_npy(out_img_path, img_save)

        img_meta = get_image_meta(imaging_path)
        index["imaging"] = {
            "src": str(imaging_path),
            "dst": str(out_img_path),
            "raw_shape_axis0_axis1_axis2": raw_img_shape,
            "dst_dtype": str(img_save.dtype),
            "dst_shape_axis0_axis1_axis2": list(img_save.shape),
            "save_mode": IMAGING_SAVE_MODE,
            "window_center": WINDOW_CENTER,
            "window_width": WINDOW_WIDTH,
            "sitk_meta": img_meta,
        }

        if VERBOSE:
            print(
                f"[OK] {case_name}: imaging -> {out_img_name}  "
                f"raw_shape={tuple(raw_img_shape)} resized_shape={img_save.shape} dtype={img_save.dtype}"
            )

        # -----------------------------
        # instances
        # -----------------------------
        if instances_dir.exists():
            inst_files = []
            for p in instances_dir.iterdir():
                if not (p.is_file() and p.name.endswith(".nii.gz")):
                    continue

                m = PAT.match(p.name)
                if m is None:
                    continue

                anno_id = int(m.group(3))
                if anno_id != ONLY_ANNOTATION_ID:
                    continue

                inst_files.append(p)

            inst_files.sort(key=lambda p: natural_key(p.name))

            for inst_path in inst_files:
                m = PAT.match(inst_path.name)
                typ = m.group(1)
                inst_id = int(m.group(2))
                anno_id = int(m.group(3))

                mask_arr = read_sitk_transposed(inst_path)
                raw_mask_shape = list(mask_arr.shape)

                mask_arr = resize_inplane_numpy(
                    mask_arr,
                    target_hw=TARGET_INPLANE_SIZE,
                    is_mask=True,
                )

                if INSTANCE_BINARIZE:
                    mask_save = (mask_arr > 0).astype(np.uint8)
                else:
                    mask_save = mask_arr.astype(np.uint8) if mask_arr.dtype != np.uint8 else mask_arr

                if SKIP_EMPTY_INSTANCE and int(mask_save.sum()) == 0:
                    continue

                out_mask_name = inst_path.name.replace(".nii.gz", ".npy")
                out_mask_path = out_inst_dir / out_mask_name
                maybe_save_npy(out_mask_path, mask_save)

                index["instances"].append({
                    "type": typ,
                    "instance_id": inst_id,
                    "annotation_id": anno_id,
                    "src": str(inst_path),
                    "dst": str(out_mask_path),
                    "raw_shape_axis0_axis1_axis2": raw_mask_shape,
                    "dst_dtype": str(mask_save.dtype),
                    "dst_shape_axis0_axis1_axis2": list(mask_save.shape),
                    "nonzero_voxels": int(mask_save.sum()),
                })

            if VERBOSE:
                print(f"     instances(annotation={ONLY_ANNOTATION_ID}) -> {len(index['instances'])} npy files")
        else:
            if VERBOSE:
                print("     instances: missing")

        # -----------------------------
        # index
        # -----------------------------
        if WRITE_INDEX_JSON:
            with open(out_case_dir / "index.json", "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, ensure_ascii=False)

        return "ok", case_name

    except Exception as e:
        print(f"[Error] {case_name}: {repr(e)}")

        if WRITE_INDEX_JSON:
            index["error"] = repr(e)
            try:
                with open(out_case_dir / "index.json", "w", encoding="utf-8") as f:
                    json.dump(index, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        return "err", case_name


# =========================================================
# Main
# =========================================================
def main():
    assert DATASET_ROOT.exists(), f"Missing DATASET_ROOT: {DATASET_ROOT}"
    safe_mkdir(OUT_ROOT)

    case_dirs = iter_cases(DATASET_ROOT)
    print(f"[Info] Found {len(case_dirs)} case folders under: {DATASET_ROOT}")
    print(f"[Info] NUM_WORKERS = {NUM_WORKERS}")
    print(f"[Info] TARGET_INPLANE_SIZE = {TARGET_INPLANE_SIZE} (Z unchanged)")
    print(f"[Info] ONLY_ANNOTATION_ID = {ONLY_ANNOTATION_ID}")

    num_ok = 0
    num_skip = 0
    num_err = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = [ex.submit(process_case, case_dir) for case_dir in case_dirs]

        for fut in as_completed(futures):
            status, case_name = fut.result()

            if status == "ok":
                num_ok += 1
            elif status == "skip":
                num_skip += 1
            else:
                num_err += 1

    print("\n========== Done ==========")
    print(f"[Summary] ok={num_ok}, skip={num_skip}, err={num_err}")
    print(f"[Out] {OUT_ROOT}")


if __name__ == "__main__":
    main()
