#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

# =========================================================
# Config (edit here)  ——  按你的偏好：全局变量配置，不用 argparse
# =========================================================
ROOT_0012 = os.environ.get('MEDVOL_SELECT_SLICES_ROOT_0012', 'data/ctorg/ct_org_npy')
OUT_ROOT  = os.environ.get('MEDVOL_SELECT_SLICES_OUT_ROOT', 'data/ctorg/ct_org_png')

NUM_SLICES = 64
TOPK_KEYSLICE = 10  # 取 topK 面积 slice 的 z 中位数做 key_slice

# mask channel order: (6,H,W,Z)
LABELS_ORDER = ["Liver", "Bladder", "Lungs", "Kidneys", "Bone", "Brain"]

MULTI_BBOX_LABELS = {"Lungs", "Kidneys"}
BONE_SINGLE_BBOX = True

MIN_COMPONENT_PIXELS = 200
MAX_BBOXES = {
    "Kidneys": 2,
    "Lungs": 2,
}

# PNG 生成：robust per-slice scaling
P_LOW = 1.0
P_HIGH = 99.0

SKIP_IF_EXISTS = True

# =========================================================
# Utils
# =========================================================
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def safe_relname(p: Path, root: Path) -> str:
    """Make a stable relative folder name for output."""
    rel = p.relative_to(root).as_posix()
    rel = rel.strip("/")
    # keep slashes as folders; also sanitize each part lightly
    parts = []
    for s in rel.split("/"):
        s2 = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s).strip("_")
        parts.append(s2 if s2 else "item")
    return "/".join(parts) if parts else "case"

def autoscale_to_uint8(x: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(x)); hi = float(np.max(x))
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

def median_of_z(z_list):
    zs = sorted([int(z) for z in z_list])
    return int(zs[len(zs) // 2])

def bbox_from_mask(mask2d: np.ndarray):
    ys, xs = np.where(mask2d)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def connected_components_bboxes(mask2d: np.ndarray, min_pixels: int, max_boxes: int | None):
    try:
        from scipy import ndimage
    except Exception as e:
        raise RuntimeError("需要 scipy.ndimage 做连通域：pip install scipy") from e

    if not mask2d.any():
        return [], []

    labeled, num = ndimage.label(mask2d.astype(np.uint8))
    if num <= 0:
        return [], []

    bboxes, pixels = [], []
    for cid in range(1, num + 1):
        comp = (labeled == cid)
        area = int(comp.sum())
        if area < min_pixels:
            continue
        box = bbox_from_mask(comp)
        if box is None:
            continue
        bboxes.append(box)
        pixels.append(area)

    if len(bboxes) > 1:
        order = np.argsort(np.array(pixels))[::-1]
        bboxes = [bboxes[i] for i in order]
        pixels = [pixels[i] for i in order]

    if max_boxes is not None and len(bboxes) > max_boxes:
        bboxes = bboxes[:max_boxes]
        pixels = pixels[:max_boxes]

    return bboxes, pixels

# =========================================================
# Sparse mask helpers
# =========================================================
def build_sparse_all_indices(mask_npz_path: str, H: int, W: int, Z: int, C: int):
    """
    Load scipy sparse mask once and prepare per-label indices sorted by z.
    Supports sp shape: (C, H*W*Z) or (H*W*Z, C)
    Flatten: linear = (y*W + x)*Z + z (z fastest)
    Returns: mode, per_class(list[(z_sorted,yx_sorted)]), nnz_total
    """
    try:
        from scipy import sparse
    except Exception as e:
        raise RuntimeError("需要安装 scipy 才能读取稀疏 mask：pip install scipy") from e

    sp = sparse.load_npz(mask_npz_path)
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
    x = (yx %  W).astype(np.int32)
    m[y, x] = True
    return m

def median_of_z_upper(z_list):
    """upper median: even count -> pick the larger middle."""
    zs = sorted([int(z) for z in z_list])
    return int(zs[len(zs) // 2])

def choose_key_slice_median_topk_nonzero(selected_slices, areas, topk: int):
    """
    只在 area>0 的 slice 里选 key_slice：
      - 先过滤 area>0
      - 在非零里按 area 排序取 topK
      - 取这些 topK 的 z 的 upper-median
    """
    if not selected_slices:
        return None, [], []

    a = np.asarray(areas, dtype=np.int64)
    s = np.asarray(selected_slices, dtype=np.int64)

    nz = np.nonzero(a > 0)[0]
    if nz.size == 0:
        return None, [], []

    # 在非零里取 topK（按 area 降序）
    k = int(min(max(1, topk), nz.size))
    # argsort on a[nz] descending, stable
    order_local = np.argsort(a[nz], kind="mergesort")[::-1][:k]
    top_idx = nz[order_local]

    top_slices = [int(s[i]) for i in top_idx.tolist()]
    top_areas  = [int(a[i]) for i in top_idx.tolist()]
    key_slice = median_of_z_upper(top_slices)
    return int(key_slice), top_slices, top_areas

# =========================================================
# Per-case processing (step2_pro)
# =========================================================
def process_case(case_dir: Path, image_npy: Path, mask_npz: Path, out_case_dir: Path):
    """
    out_case_dir will contain:
      - png_64slices/slice_####.png
      - png_64slices/text.json
    """
    png_dir = out_case_dir / "png_64slices"
    ensure_dir(str(png_dir))

    # load image
    img = np.load(str(image_npy))  # (1,H,W,Z)
    if img.ndim != 4 or img.shape[0] != 1:
        raise ValueError(f"Unexpected image shape: {img.shape} at {image_npy}")
    _, H, W, Z = img.shape

    # uniform 64
    selected_slices = uniform_select_slices(Z, NUM_SLICES)
    if len(selected_slices) != NUM_SLICES:
        raise RuntimeError(f"selected_slices size != {NUM_SLICES} for {case_dir}")

    # export 64 pngs
    for z in selected_slices:
        out_png = png_dir / f"slice_{int(z):04d}.png"
        if SKIP_IF_EXISTS and out_png.exists():
            continue
        sl = img[0, :, :, int(z)]
        gray = autoscale_to_uint8(sl, p_low=P_LOW, p_high=P_HIGH)
        Image.fromarray(gray, mode="L").save(str(out_png))

    # sparse load once
    mode, per_class, nnz_total = build_sparse_all_indices(str(mask_npz), H, W, Z, C=len(LABELS_ORDER))

    # per label entries
    label_entries = []
    for ci, lab in enumerate(LABELS_ORDER):
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
                "label_name": lab,
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
            label_entries.append(entry)
            continue

        # bbox on key_slice
        mask2d = get_mask2d_from_sorted(z_sorted, yx_sorted, H, W, Z, int(key_slice))

        if (lab in MULTI_BBOX_LABELS) and not (BONE_SINGLE_BBOX and lab == "Bone"):
            max_boxes = MAX_BBOXES.get(lab, None)
            bboxes, pix_list = connected_components_bboxes(
                mask2d, min_pixels=MIN_COMPONENT_PIXELS, max_boxes=max_boxes
            )
            bbox_mode = "components"
            if len(bboxes) == 0:
                box = bbox_from_mask(mask2d)
                if box is None:
                    bboxes, pix_list = [], []
                    bbox_mode = "none"
                else:
                    bboxes, pix_list = [box], [int(mask2d.sum())]
                    bbox_mode = "single_fallback"
        else:
            box = bbox_from_mask(mask2d)
            bboxes = [] if box is None else [box]
            pix_list = [] if box is None else [int(mask2d.sum())]
            bbox_mode = "single"

        present = (max_area > 0) and (len(bboxes) > 0)

        entry = {
            "label_name": lab,
            "present": bool(present),
            "key_slice": int(key_slice),
            "bbox_2d_list": bboxes,
            "mask_pixels_list": pix_list,
            "mask_pixels_per_slice": mask_pixels_per_slice,
            "max_area_on_selected": int(max_area),
            "bbox_mode": bbox_mode,
            "min_component_pixels": int(MIN_COMPONENT_PIXELS),
            "key_slice_strategy": f"median_z_of_top{TOPK_KEYSLICE}_by_area",
            "topk_slices": [int(z) for z in topk_slices],
            "topk_areas": [int(a) for a in topk_areas],
        }
        label_entries.append(entry)

    meta = {
        "case_dir": str(case_dir),
        "image_path": str(image_npy),
        "mask_path": str(mask_npz),
        "mask_sparse_mode": mode,
        "mask_nnz_total": int(nnz_total),
        "labels": label_entries,
        "selected_slices": [int(z) for z in selected_slices],
        "volume_shape": [int(Z), int(H), int(W)],
        "key_slice_strategy": f"median_z_of_top{TOPK_KEYSLICE}_by_area_within_selected64",
        "note": (
            "Batch step2_pro: first uniformly select 64 slices. For each label, compute mask area on these 64 slices. "
            f"Pick key_slice as median-z among top-{TOPK_KEYSLICE} slices ranked by area. "
            "On key_slice extract bbox(es): Lungs/Kidneys use connected components; Bone kept single bbox; others single bbox."
        )
    }

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
    Recursively find folders containing image.npy and one mask_(6,512,512,*.npz).
    Return list of tuples: (case_dir, image_path, mask_path)
    """
    cases = []
    for img_path in root.rglob("image.npy"):
        case_dir = img_path.parent
        # find mask_(6,512,512,*.npz) in same folder
        masks = sorted(case_dir.glob("mask_(6,512,512,*)*.npz"))
        if not masks:
            # sometimes naming like mask_(6,512,512,501).npz exact
            masks = sorted(case_dir.glob("mask_*.npz"))
        if not masks:
            continue

        # choose the one that contains "(6,512,512," and endswith .npz if possible
        chosen = None
        for m in masks:
            if "mask_(6,512,512" in m.name:
                chosen = m
                break
        if chosen is None:
            # fallback: first npz
            chosen = masks[0]

        cases.append((case_dir, img_path, chosen))
    # de-dup by case_dir
    uniq = {}
    for cd, ip, mp in cases:
        uniq[str(cd)] = (cd, ip, mp)
    return list(uniq.values())

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
        rel = safe_relname(case_dir, root)
        out_case_dir = out_root / rel

        try:
            ensure_dir(str(out_case_dir))
            ret = process_case(case_dir, img_path, mask_path, out_case_dir)
            summary["ok"] += 1
        except Exception as e:
            summary["fail"] += 1
            err = {
                "case_dir": str(case_dir),
                "image": str(img_path),
                "mask": str(mask_path),
                "error": repr(e),
                "traceback": traceback.format_exc()[-3000:],  # truncate
            }
            summary["fails"].append(err)
            print(f"\n[Fail] {case_dir} -> {repr(e)}\n")

    # write summary
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
