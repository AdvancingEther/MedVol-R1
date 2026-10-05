#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a large info_stats.json from KiTS23 dataset_npy.

Rules:
- Only use annotation-1 files
- Only use:
    kidney_instance-*_annotation-1.npy
    tumor_instance-*_annotation-1.npy
    cyst_instance-*_annotation-1.npy
- For each case:
    - build kidney_1 / kidney_2
    - assign each tumor/cyst to the kidney with larger overlap
    - compute voxel count
    - compute a single relative_height in assigned kidney

relative_height:
- computed along axis0
- axis0 increases from top -> bottom (confirmed by user)
- 0.0 means near upper end of assigned kidney
- 1.0 means near lower end of assigned kidney
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm


# =========================================================
# Global config
# =========================================================
npy_path = os.environ.get('MEDVOL_STEP2_STATS_TUMOR_CYST_NPY_PATH', 'data/kits23/kits23_npy')
output_json = os.environ.get('MEDVOL_STEP2_STATS_TUMOR_CYST_OUTPUT_JSON', 'data/kits23/final_vqa_gen/info_stats.json')

CASE_PREFIX = "case_"
INSTANCE_DIRNAME = "instances"

# 多进程数
NUM_WORKERS = 8

# 如果和两个 kidney overlap 都为 0，是否用质心距离兜底
USE_CENTROID_FALLBACK = True

# 只使用 annotation-1
TARGET_ANNOTATION_ID = 1

# 是否打印详细日志（开了进度条建议 False）
VERBOSE = False

PAT = re.compile(r"^(kidney|tumor|cyst)_instance-(\d+)_annotation-(\d+)\.npy$")


# =========================================================
# Helpers
# =========================================================
def natural_key(text: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def load_mask(path: Path) -> np.ndarray:
    arr = np.load(path)
    return arr > 0


def get_nonzero_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask)


def compute_voxels(mask: np.ndarray) -> int:
    return int(mask.sum())


def compute_centroid(mask: np.ndarray) -> Optional[List[float]]:
    coords = get_nonzero_coords(mask)
    if coords.shape[0] == 0:
        return None
    c = coords.mean(axis=0)
    return [float(v) for v in c]


def compute_bbox(mask: np.ndarray) -> Optional[Dict[str, List[int]]]:
    coords = get_nonzero_coords(mask)
    if coords.shape[0] == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return {
        "axis0": [int(mins[0]), int(maxs[0])],
        "axis1": [int(mins[1]), int(maxs[1])],
        "axis2": [int(mins[2]), int(maxs[2])],
    }


def compute_overlap(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.logical_and(a, b).sum())


def euclidean_distance(p1: List[float], p2: List[float]) -> float:
    a = np.array(p1, dtype=np.float64)
    b = np.array(p2, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def compute_relative_height(lesion_mask: np.ndarray, kidney_mask: np.ndarray) -> float:
    """
    axis0 increases from top -> bottom.

    Return:
      0.0 ~ 1.0
      0.0 = near upper end of assigned kidney
      1.0 = near lower end of assigned kidney
    """
    lesion_coords = get_nonzero_coords(lesion_mask)
    kidney_coords = get_nonzero_coords(kidney_mask)

    if lesion_coords.shape[0] == 0 or kidney_coords.shape[0] == 0:
        return -1.0

    lesion_c0 = float(lesion_coords[:, 0].mean())
    kidney_min0 = float(kidney_coords[:, 0].min())
    kidney_max0 = float(kidney_coords[:, 0].max())

    if kidney_max0 <= kidney_min0:
        return 0.5

    ratio = (lesion_c0 - kidney_min0) / (kidney_max0 - kidney_min0)
    ratio = max(0.0, min(1.0, ratio))
    return float(ratio)


def list_case_dirs(root: Path) -> List[Path]:
    case_dirs = []
    if not root.exists():
        return case_dirs
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith(CASE_PREFIX):
            case_dirs.append(p)
    case_dirs.sort(key=lambda p: natural_key(p.name))
    return case_dirs


def list_annotation1_instance_files(inst_dir: Path, instance_type: str) -> List[Path]:
    files = []
    if not inst_dir.exists():
        return files

    for p in inst_dir.iterdir():
        if not p.is_file():
            continue

        m = PAT.match(p.name)
        if m is None:
            continue

        typ = m.group(1)
        anno_id = int(m.group(3))

        if typ == instance_type and anno_id == TARGET_ANNOTATION_ID:
            files.append(p)

    files.sort(key=lambda p: natural_key(p.name))
    return files


def extract_instance_id(path: Path) -> int:
    m = PAT.match(path.name)
    if m is None:
        return -1
    return int(m.group(2))


def extract_annotation_id(path: Path) -> int:
    m = PAT.match(path.name)
    if m is None:
        return -1
    return int(m.group(3))


# =========================================================
# Core per-case function
# =========================================================
def build_case_info(case_dir_str: str) -> Tuple[str, Dict]:
    case_dir = Path(case_dir_str)
    case_name = case_dir.name
    inst_dir = case_dir / INSTANCE_DIRNAME

    kidney_files = list_annotation1_instance_files(inst_dir, "kidney")
    tumor_files = list_annotation1_instance_files(inst_dir, "tumor")
    cyst_files = list_annotation1_instance_files(inst_dir, "cyst")

    kidneys = []
    for i, kf in enumerate(kidney_files, start=1):
        kmask = load_mask(kf)
        kcent = compute_centroid(kmask)
        kbbox = compute_bbox(kmask)

        kidneys.append({
            "name": f"kidney_{i}",
            "file": str(kf),
            "instance_id": extract_instance_id(kf),
            "annotation_id": extract_annotation_id(kf),
            "mask": kmask,
            "voxels": compute_voxels(kmask),
            "centroid": kcent,
            "bbox": kbbox,
        })

    result = {}
    for k in kidneys:
        axis0_range = k["bbox"]["axis0"] if k["bbox"] is not None else [-1, -1]
        result[k["name"]] = {
            "file": k["file"],
            "instance_id": k["instance_id"],
            "annotation_id": k["annotation_id"],
            "voxels": k["voxels"],
            "centroid_axis0_axis1_axis2": k["centroid"],
            "axis0_range": axis0_range,
            "tumors": [],
            "cysts": [],
        }

    def assign_one_lesion(path: Path) -> Tuple[Optional[str], Dict]:
        lmask = load_mask(path)
        lvox = compute_voxels(lmask)
        lcent = compute_centroid(lmask)
        lbbox = compute_bbox(lmask)

        info = {
            "file": str(path),
            "instance_id": extract_instance_id(path),
            "annotation_id": extract_annotation_id(path),
            "voxels": lvox,
            "centroid_axis0_axis1_axis2": lcent,
            "bbox_axis0_axis1_axis2": lbbox,
        }

        if len(kidneys) == 0:
            info["assigned_kidney"] = None
            info["assignment_method"] = "no_kidney_found"
            info["overlap_with_assigned_kidney"] = 0
            info["relative_height"] = -1.0
            return None, info

        overlaps = [compute_overlap(lmask, k["mask"]) for k in kidneys]
        best_idx = int(np.argmax(overlaps))
        best_overlap = int(overlaps[best_idx])
        method = "overlap"

        if best_overlap == 0 and USE_CENTROID_FALLBACK:
            if lcent is None:
                chosen_idx = best_idx
            else:
                dists = []
                for k in kidneys:
                    if k["centroid"] is None:
                        dists.append(1e18)
                    else:
                        dists.append(euclidean_distance(lcent, k["centroid"]))
                chosen_idx = int(np.argmin(dists))
            best_idx = chosen_idx
            method = "centroid_fallback"

        chosen = kidneys[best_idx]
        rel_h = compute_relative_height(lmask, chosen["mask"])

        info["assigned_kidney"] = chosen["name"]
        info["assignment_method"] = method
        info["overlap_with_assigned_kidney"] = best_overlap if method == "overlap" else 0
        info["relative_height"] = rel_h

        return chosen["name"], info

    for tf in tumor_files:
        kname, info = assign_one_lesion(tf)
        if kname is not None:
            result[kname]["tumors"].append(info)

    for cf in cyst_files:
        kname, info = assign_one_lesion(cf)
        if kname is not None:
            result[kname]["cysts"].append(info)

    # 排序，方便阅读
    for kname in list(result.keys()):
        result[kname]["tumors"].sort(key=lambda x: (x["instance_id"], x["annotation_id"]))
        result[kname]["cysts"].sort(key=lambda x: (x["instance_id"], x["annotation_id"]))

    result["_summary"] = {
        "num_kidneys_annotation1": len(kidney_files),
        "num_tumors_annotation1": len(tumor_files),
        "num_cysts_annotation1": len(cyst_files),
    }

    return case_name, result


# =========================================================
# Main
# =========================================================
def main():
    root = Path(npy_path)
    assert root.exists(), f"npy_path not found: {root}"

    case_dirs = list_case_dirs(root)
    print(f"[Info] Found {len(case_dirs)} case folders under: {root}")
    print(f"[Info] NUM_WORKERS = {NUM_WORKERS}")

    all_info = {}
    ok = 0
    err = 0

    case_dir_strs = [str(p) for p in case_dirs]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(build_case_info, p) for p in case_dir_strs]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Building info_stats"):
            try:
                case_name, case_result = future.result()
                all_info[case_name] = case_result
                ok += 1
                if VERBOSE:
                    print(f"[OK] {case_name}")
            except Exception as e:
                err += 1
                # 尽量保留 case 名信息（可能拿不到）
                err_key = f"_error_case_{err:04d}"
                all_info[err_key] = {
                    "_error": repr(e)
                }
                if VERBOSE:
                    print(f"[Error] {repr(e)}")

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_info, f, indent=2, ensure_ascii=False)

    print("\n========== Done ==========")
    print(f"[Summary] ok={ok}, err={err}")
    print(f"[Saved] {output_json}")


if __name__ == "__main__":
    main()
