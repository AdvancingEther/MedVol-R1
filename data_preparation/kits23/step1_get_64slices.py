#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse
from PIL import Image, ImageDraw
from tqdm import tqdm


# =========================================================
# Global config
# =========================================================
ROOT_0012 = os.environ.get('MEDVOL_STEP1_GET_64SLICES_ROOT_0012', 'data/kits23/kits23_npy_m3d')
OUT_ROOT  = os.environ.get('MEDVOL_STEP1_GET_64SLICES_OUT_ROOT', 'data/kits23/kits23_png_m3d')

NUM_SLICES = 64
TOPK_KEYSLICE = 10   # 取 topK 面积 slice 的 z 中位数做 key_slice

# PNG 生成：robust per-slice scaling
P_LOW = 1.0
P_HIGH = 99.0

SKIP_IF_EXISTS = True
SAVE_TEMPLATE_VIS = True   # 是否保存每个 present template 的 bbox 叠加图

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


# =========================================================
# Utils
# =========================================================
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def autoscale_to_uint8(x: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.uint8)
    x01 = (x - lo) / (hi - lo)
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0 + 0.5).astype(np.uint8)


def uniform_select_slices(Z: int, k: int):
    if k >= Z:
        return list(range(Z))
    idx = np.linspace(0, Z - 1, k)
    idx = np.round(idx).astype(np.int32)
    idx = np.unique(idx)
    if idx.size < k:
        chosen = set(idx.tolist())
        cand = np.round(np.linspace(0, Z - 1, k * 5)).astype(np.int32)
        for c in cand:
            if int(c) not in chosen:
                chosen.add(int(c))
                if len(chosen) >= k:
                    break
        idx = np.array(sorted(chosen), dtype=np.int32)
        if idx.size > k:
            idx = idx[:k]
    return idx.tolist()


def bbox_from_mask(mask2d: np.ndarray):
    ys, xs = np.where(mask2d)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def median_of_z_upper(z_list):
    zs = sorted([int(z) for z in z_list])
    return int(zs[len(zs) // 2])


def choose_key_slice_median_topk_nonzero(selected_slices, areas, topk: int):
    if not selected_slices:
        return None, [], []

    a = np.asarray(areas, dtype=np.int64)
    s = np.asarray(selected_slices, dtype=np.int64)

    nz = np.nonzero(a > 0)[0]
    if nz.size == 0:
        return None, [], []

    k = int(min(max(1, topk), nz.size))
    order_local = np.argsort(a[nz], kind="mergesort")[::-1][:k]
    top_idx = nz[order_local]

    top_slices = [int(s[i]) for i in top_idx.tolist()]
    top_areas = [int(a[i]) for i in top_idx.tolist()]
    key_slice = median_of_z_upper(top_slices)
    return int(key_slice), top_slices, top_areas


# =========================================================
# Sparse helpers
# =========================================================
def build_sparse_all_indices_from_loaded_sparse(sp, H: int, W: int, Z: int, C: int):
    """
    Supports sparse shape:
      (C, H*W*Z)   or   (H*W*Z, C)

    Flatten rule:
      linear = (y*W + x)*Z + z    (z fastest)
    """
    nnz_total = int(sp.nnz)
    r, c = sp.shape
    HWZ = H * W * Z

    if r == C and c == HWZ:
        mode = "C_by_HWZ"
        sp = sp.tocsr()
        per_class = []
        for ci in range(C):
            row = sp.getrow(ci)
            idx = row.indices.astype(np.int64)
            z_arr = (idx % Z).astype(np.int32)
            yx_arr = (idx // Z).astype(np.int64)
            order = np.argsort(z_arr, kind="mergesort")
            per_class.append((z_arr[order], yx_arr[order]))
        return mode, per_class, nnz_total

    if c == C and r == HWZ:
        mode = "HWZ_by_C"
        sp = sp.tocsc()
        per_class = []
        for ci in range(C):
            col = sp.getcol(ci).tocoo()
            idx = col.row.astype(np.int64)
            z_arr = (idx % Z).astype(np.int32)
            yx_arr = (idx // Z).astype(np.int64)
            order = np.argsort(z_arr, kind="mergesort")
            per_class.append((z_arr[order], yx_arr[order]))
        return mode, per_class, nnz_total

    raise ValueError(
        f"[Sparse mask] shape={sp.shape} mismatch expected ({C},{HWZ}) or ({HWZ},{C}) "
        f"with (H,W,Z)=({H},{W},{Z})"
    )


def slice_area_from_sorted(z_sorted: np.ndarray, z: int):
    lo = np.searchsorted(z_sorted, z, side="left")
    hi = np.searchsorted(z_sorted, z, side="right")
    return lo, hi, int(hi - lo)


def get_mask2d_from_sorted(z_sorted, yx_sorted, H, W, Z, z):
    lo = np.searchsorted(z_sorted, z, side="left")
    hi = np.searchsorted(z_sorted, z, side="right")
    m = np.zeros((H, W), dtype=bool)
    if hi <= lo:
        return m
    yx = yx_sorted[lo:hi]
    y = (yx // W).astype(np.int32)
    x = (yx % W).astype(np.int32)
    m[y, x] = True
    return m


# =========================================================
# Visualization helpers
# =========================================================
def draw_bbox_on_gray(gray_u8: np.ndarray, bbox_list, color=(255, 0, 0), width=2):
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    for box in bbox_list:
        x1, y1, x2, y2 = box
        for t in range(width):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)

    return img


# =========================================================
# Per-case processing
# =========================================================
def process_case(case_dir: Path, image_npy: Path, mask_npz: Path, out_case_dir: Path):
    """
    out_case_dir will contain:
      - png_64slices/
          - slice_####.png
          - text.json
          - template_vis/   (optional)
    """
    png_dir = out_case_dir / "png_64slices"
    vis_dir = png_dir / "template_vis"
    ensure_dir(str(png_dir))
    if SAVE_TEMPLATE_VIS:
        ensure_dir(str(vis_dir))

    # load image
    img = np.load(str(image_npy))  # expected (1,H,W,Z)
    if img.ndim != 4 or img.shape[0] != 1:
        raise ValueError(f"Unexpected image shape: {img.shape} at {image_npy}")

    _, H, W, Z = img.shape

    # select 64 slices
    selected_slices = uniform_select_slices(Z, NUM_SLICES)
    # if len(selected_slices) != NUM_SLICES:
    #     raise RuntimeError(f"selected_slices size != {NUM_SLICES} for {case_dir}")

    # export 64 grayscale slices
    for z in selected_slices:
        out_png = png_dir / f"slice_{int(z):04d}.png"
        if SKIP_IF_EXISTS and out_png.exists():
            continue
        sl = img[0, :, :, int(z)]
        gray = autoscale_to_uint8(sl, p_low=P_LOW, p_high=P_HIGH)
        Image.fromarray(gray, mode="L").save(str(out_png))

    # load sparse mask
    sp = scipy.sparse.load_npz(str(mask_npz))
    C = len(TEMPLATE_ORDER)
    mode, per_class, nnz_total = build_sparse_all_indices_from_loaded_sparse(sp, H, W, Z, C=C)

    # per-template entries
    template_entries = []
    for ci, template_name in enumerate(TEMPLATE_ORDER):
        z_sorted, yx_sorted = per_class[ci]

        # areas on selected_64
        mask_pixels_per_slice = []
        for z in selected_slices:
            _, _, area = slice_area_from_sorted(z_sorted, int(z))
            mask_pixels_per_slice.append(int(area))

        max_area = int(max(mask_pixels_per_slice)) if mask_pixels_per_slice else 0
        key_slice, topk_slices, topk_areas = choose_key_slice_median_topk_nonzero(
            selected_slices, mask_pixels_per_slice, topk=TOPK_KEYSLICE
        )

        if key_slice is None:
            entry = {
                "template_name": template_name,
                "present": False,
                "key_slice": None,
                "bbox_2d_list": [],
                "mask_pixels_list": [],
                "mask_pixels_per_slice": mask_pixels_per_slice,
                "max_area_on_selected": 0,
                "bbox_mode": "none",
                "key_slice_strategy": f"median_z_of_top{TOPK_KEYSLICE}_by_area",
                "topk_slices": [],
                "topk_areas": [],
            }
            template_entries.append(entry)
            continue

        # single bbox on key_slice
        mask2d = get_mask2d_from_sorted(z_sorted, yx_sorted, H, W, Z, int(key_slice))
        box = bbox_from_mask(mask2d)
        bbox_list = [] if box is None else [box]
        pix_list = [] if box is None else [int(mask2d.sum())]
        bbox_mode = "single"

        present = (max_area > 0) and (len(bbox_list) > 0)

        entry = {
            "template_name": template_name,
            "present": bool(present),
            "key_slice": int(key_slice),
            "bbox_2d_list": bbox_list,
            "mask_pixels_list": pix_list,
            "mask_pixels_per_slice": mask_pixels_per_slice,
            "max_area_on_selected": int(max_area),
            "bbox_mode": bbox_mode,
            "key_slice_strategy": f"median_z_of_top{TOPK_KEYSLICE}_by_area",
            "topk_slices": [int(z) for z in topk_slices],
            "topk_areas": [int(a) for a in topk_areas],
        }
        template_entries.append(entry)

        # save overlay visualization for present templates
        if SAVE_TEMPLATE_VIS and present:
            out_vis = vis_dir / f"{ci:02d}_{template_name}_z{int(key_slice):04d}.png"
            if not (SKIP_IF_EXISTS and out_vis.exists()):
                sl = img[0, :, :, int(key_slice)]
                gray = autoscale_to_uint8(sl, p_low=P_LOW, p_high=P_HIGH)
                vis = draw_bbox_on_gray(gray, bbox_list, color=(255, 0, 0), width=2)
                vis.save(str(out_vis))

    meta = {
        "case_dir": str(case_dir),
        "image_path": str(image_npy),
        "mask_path": str(mask_npz),
        "mask_sparse_mode": mode,
        "mask_nnz_total": int(nnz_total),
        "templates": template_entries,
        "template_order": TEMPLATE_ORDER,
        "selected_slices": [int(z) for z in selected_slices],
        "volume_shape": [int(Z), int(H), int(W)],
        "key_slice_strategy": f"median_z_of_top{TOPK_KEYSLICE}_by_area_within_selected64",
        "note": (
            "Batch step2_pro for KiTS23 M3D-style template channels: first uniformly select 64 slices. "
            "For each template channel, compute mask area on these 64 slices. "
            f"Pick key_slice as median-z among top-{TOPK_KEYSLICE} slices ranked by area. "
            "On key_slice extract a single tight bbox."
        )
    }

    # IMPORTANT: text.json under png_64slices
    out_json = png_dir / "text.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "case": str(case_dir),
        "out": str(png_dir),
        "Z": int(Z),
        "status": "ok",
    }


# =========================================================
# Find cases
# =========================================================
def find_cases(root: Path):
    """
    Recursively find folders containing image.npy and one mask_(14,*,*,*).npz.
    Return list of tuples: (case_dir, image_path, mask_path)
    """
    cases = []
    for img_path in root.rglob("image.npy"):
        case_dir = img_path.parent

        masks = sorted(case_dir.glob("mask_(14,*)*.npz"))
        if not masks:
            masks = sorted(case_dir.glob("mask_*.npz"))
        if not masks:
            continue

        chosen = None
        for m in masks:
            if "mask_(14," in m.name:
                chosen = m
                break
        if chosen is None:
            chosen = masks[0]

        cases.append((case_dir, img_path, chosen))

    # de-dup by case_dir
    uniq = {}
    for cd, ip, mp in cases:
        uniq[str(cd)] = (cd, ip, mp)
    out = list(uniq.values())
    out.sort(key=lambda x: x[0].name)
    return out


# =========================================================
# Main
# =========================================================
def main():
    root = Path(ROOT_0012)
    out_root = Path(OUT_ROOT)
    ensure_dir(str(out_root))

    cases = find_cases(root)
    print(f"[Scan] Found {len(cases)} cases under: {root}")

    summary = {
        "root": ROOT_0012,
        "out_root": OUT_ROOT,
        "num_cases": len(cases),
        "ok": 0,
        "fail": 0,
        "fails": [],
    }

    for case_dir, img_path, mask_path in tqdm(cases, desc="Process cases"):
        rel = case_dir.relative_to(root).as_posix().strip("/")
        out_case_dir = out_root / rel

        try:
            ensure_dir(str(out_case_dir))
            _ = process_case(case_dir, img_path, mask_path, out_case_dir)
            summary["ok"] += 1
        except Exception as e:
            summary["fail"] += 1
            err = {
                "case_dir": str(case_dir),
                "image": str(img_path),
                "mask": str(mask_path),
                "error": repr(e),
                "traceback": traceback.format_exc()[-3000:],
            }
            summary["fails"].append(err)
            print(f"\n[Fail] {case_dir} -> {repr(e)}\n")

    summary_path = out_root / "step2_pro_batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== Done ==========")
    print(f"OK   : {summary['ok']}")
    print(f"Fail : {summary['fail']}")
    print("Summary:", str(summary_path))
    print("==========================")


if __name__ == "__main__":
    main()
