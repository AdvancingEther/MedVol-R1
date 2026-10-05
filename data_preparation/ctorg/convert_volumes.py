#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
from scipy import sparse
from scipy.ndimage import zoom

# =========================================================
# Config (edit here) —— 全量处理，按你的偏好：全局变量配置，不用 argparse
# =========================================================
CTORG_ROOT = os.environ.get('MEDVOL_CONVERT_VOLUMES_CTORG_ROOT', 'data/ctorg/raw')
VOLUMES_DIR = os.path.join(CTORG_ROOT, "volumes")
LABELS_DIR  = os.path.join(CTORG_ROOT, "labels")

OUT_ROOT = os.environ.get('MEDVOL_CONVERT_VOLUMES_OUT_ROOT', 'data/ctorg/ct_org_npy')

# window (WW, WC)
WC = 40.0
WW = 400.0

OUT_H = 512
OUT_W = 512

# 可视化：均匀选 64 张
NUM_VIS = 64

# 强制重做（True: 删除旧目录重做；False: 若检测到完整输出就跳过）
FORCE_REMAKE = False

# 固定 6 通道，对应 label 像素值 1..6
LABELS_ORDER = ["Liver", "Bladder", "Lungs", "Kidneys", "Bone", "Brain"]
LABEL_IDS = [1, 2, 3, 4, 5, 6]  # must align with LABELS_ORDER
C_FIXED = 6

# 保存可视化 png（如果你想关掉就设 False）
SAVE_VIS_64 = False

# =========================
# 并行配置（推荐多进程）
# =========================
# 建议运行前在 shell 里设置（避免“进程x线程”爆炸）：
#   export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NUM_WORKERS = 6  # 4~8 通常较合适（看 CPU/内存/IO）

# 是否在输出时用临时文件+原子替换（更稳，避免中途崩导致半成品被当成完成）
ATOMIC_WRITE = True

# =========================================================
# Utils
# =========================================================
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def reset_out_dir(out_dir: str):
    if os.path.exists(out_dir) and FORCE_REMAKE:
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

def is_case_complete(out_dir: str) -> bool:
    """
    判断一个 case 是否已经完整产出（用于 FORCE_REMAKE=False 的 skip）
    """
    image_out = os.path.join(out_dir, "image.npy")
    # mask 文件名里带 Z，不固定；用 glob 方式
    mask_npz = list(Path(out_dir).glob("mask_(6,512,512,*)*.npz"))
    vis_dir = os.path.join(out_dir, "vis_64")
    vis_ok = True
    if SAVE_VIS_64:
        vis_ok = os.path.isdir(vis_dir) and (len(list(Path(vis_dir).glob("slice_*.png"))) >= min(NUM_VIS, 4))
    return os.path.isfile(image_out) and (len(mask_npz) > 0) and vis_ok

def _stem_nii(fn: str) -> str:
    if fn.endswith(".nii.gz"):
        return fn[:-7]
    if fn.endswith(".nii"):
        return fn[:-4]
    return os.path.splitext(fn)[0]

def parse_case_id_from_stem(stem: str):
    """
    volume-136 -> 136
    labels-136 -> 136
    兼容 volume_136 / labels_136 等少数变体
    """
    m = re.search(r"(\d+)$", stem)
    if not m:
        return None
    return m.group(1)

def load_ct_zyx_float32(path: str) -> np.ndarray:
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)  # often (X,Y,Z)
    arr = np.transpose(arr, (2, 1, 0))     # -> (Z,Y,X)
    return arr

def load_label_zyx_int16(path: str) -> np.ndarray:
    img = nib.load(path)
    arr = np.asanyarray(img.dataobj)
    arr = np.transpose(arr, (2, 1, 0))     # -> (Z,Y,X)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = np.rint(arr).astype(np.int16)
    else:
        arr = arr.astype(np.int16, copy=False)
    return arr

def window_norm01(ct_zyx: np.ndarray, wc: float, ww: float) -> np.ndarray:
    lower = wc - ww / 2.0
    upper = wc + ww / 2.0
    ct = np.clip(ct_zyx, lower, upper)
    ct = (ct - lower) / max((upper - lower), 1e-8)
    return np.clip(ct, 0.0, 1.0).astype(np.float32)

def resize_zyx_to_512(ct_zyx: np.ndarray, lbl_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize only in Y,X to (OUT_H, OUT_W), keep Z unchanged.
    CT: linear, Label: nearest.
    """
    z, y, x = ct_zyx.shape
    if (y, x) == (OUT_H, OUT_W):
        return ct_zyx, lbl_zyx

    zoom_y = OUT_H / float(y)
    zoom_x = OUT_W / float(x)

    ct_rs  = zoom(ct_zyx,  zoom=(1.0, zoom_y, zoom_x), order=1)  # linear
    lbl_rs = zoom(lbl_zyx, zoom=(1.0, zoom_y, zoom_x), order=0)  # nearest

    ct_rs  = ct_rs.astype(np.float32, copy=False)
    lbl_rs = lbl_rs.astype(np.int16,  copy=False)
    return ct_rs, lbl_rs

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

def save_vis_64_pngs(ct_1hwz: np.ndarray, out_dir: str):
    """
    ct_1hwz: (1,H,W,Z) float32 0..1
    """
    _, H, W, Z = ct_1hwz.shape
    z_list = uniform_select_slices(Z, NUM_VIS)

    ensure_dir(out_dir)
    for z in z_list:
        sl = ct_1hwz[0, :, :, int(z)]  # (H,W) 0..1
        u8 = (np.clip(sl, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(u8, mode="L").save(os.path.join(out_dir, f"slice_{int(z):04d}.png"))

def save_fixed6_onehot_csr_from_label_hwz(lbl_hwz: np.ndarray, out_npz: str):
    """
    固定 6 通道：
      ch i <-> label == LABEL_IDS[i] (1..6)

    flatten on (H,W,Z) so that linear = (y*W + x)*Z + z  (z fastest)
    output CSR shape: (6, H*W*Z) dtype=uint8
    """
    H, W, Z = lbl_hwz.shape
    N = H * W * Z
    flat = lbl_hwz.reshape(-1)

    rows = []
    for lab in LABEL_IDS:
        idx = np.flatnonzero(flat == int(lab)).astype(np.int64, copy=False)
        if idx.size == 0:
            rows.append(sparse.csr_matrix((1, N), dtype=np.uint8))
            continue
        data = np.ones(idx.shape[0], dtype=np.uint8)
        indptr = np.array([0, idx.size], dtype=np.int64)
        row = sparse.csr_matrix((data, idx, indptr), shape=(1, N), dtype=np.uint8)
        rows.append(row)

    mat = sparse.vstack(rows, format="csr")  # (6, N)
    sparse.save_npz(out_npz, mat)

def write_fixed_mapping_txt(out_dir: str):
    fp = os.path.join(out_dir, "mask_labels_fixed.txt")
    with open(fp, "w", encoding="utf-8") as f:
        for i, (name, lab) in enumerate(zip(LABELS_ORDER, LABEL_IDS)):
            f.write(f"channel {i}: label_id={lab} name={name}\n")
    return fp

def atomic_save_npy(path: str, arr: np.ndarray):
    """
    np.save() 会在文件名不以 .npy 结尾时自动追加 .npy
    所以 tmp 必须以 .npy 结尾，否则 os.replace 会找不到 tmp。
    """
    if not ATOMIC_WRITE:
        np.save(path, arr)
        return
    tmp = path + ".tmp.npy"   # 关键：保证以 .npy 结尾
    np.save(tmp, arr)
    os.replace(tmp, path)

def atomic_save_npz(path: str, saver_fn):
    """
    sparse.save_npz() 同理：若不是以 .npz 结尾，会自动追加 .npz
    所以 tmp 必须以 .npz 结尾。
    saver_fn(tmp_path) 需要把 npz 写到 tmp_path。
    """
    if not ATOMIC_WRITE:
        saver_fn(path)
        return
    tmp = path + ".tmp.npz"   # 关键：保证以 .npz 结尾
    saver_fn(tmp)
    os.replace(tmp, path)

# =========================================================
# Pairing scan
# =========================================================
def scan_cases(volumes_dir: str, labels_dir: str):
    """
    返回匹配到的 (case_id, volume_path, label_path) 列表
    规则：volume-XXX(.nii/.nii.gz) 对应 labels-XXX(.nii/.nii.gz)
    """
    vol_map = {}
    lab_map = {}

    for p in sorted(Path(volumes_dir).glob("*.nii*")):
        stem = _stem_nii(p.name)
        cid = parse_case_id_from_stem(stem)
        if cid is None:
            continue
        # 同一 cid 可能有 .nii / .nii.gz，优先 .nii.gz（文件通常更小）
        key = cid
        prev = vol_map.get(key, None)
        if prev is None:
            vol_map[key] = str(p)
        else:
            if str(p).endswith(".nii.gz") and (not str(prev).endswith(".nii.gz")):
                vol_map[key] = str(p)

    for p in sorted(Path(labels_dir).glob("*.nii*")):
        stem = _stem_nii(p.name)
        cid = parse_case_id_from_stem(stem)
        if cid is None:
            continue
        key = cid
        prev = lab_map.get(key, None)
        if prev is None:
            lab_map[key] = str(p)
        else:
            if str(p).endswith(".nii.gz") and (not str(prev).endswith(".nii.gz")):
                lab_map[key] = str(p)

    # intersect
    case_ids = sorted(set(vol_map.keys()) & set(lab_map.keys()), key=lambda x: int(x))
    pairs = []
    for cid in case_ids:
        pairs.append((cid, vol_map[cid], lab_map[cid]))

    missing_vol = sorted(set(lab_map.keys()) - set(vol_map.keys()), key=lambda x: int(x))
    missing_lab = sorted(set(vol_map.keys()) - set(lab_map.keys()), key=lambda x: int(x))
    return pairs, missing_vol, missing_lab

# =========================================================
# Process one case
# =========================================================
def process_one_case(case_id: str, volume_path: str, label_path: str):
    out_dir = os.path.join(OUT_ROOT, f"volume-{case_id}")
    if (not FORCE_REMAKE) and is_case_complete(out_dir):
        return True, "skip (complete)", out_dir

    reset_out_dir(out_dir)

    # 1) load
    ct_zyx  = load_ct_zyx_float32(volume_path)
    lbl_zyx = load_label_zyx_int16(label_path)

    if ct_zyx.shape != lbl_zyx.shape:
        return False, f"shape mismatch: ct{ct_zyx.shape} vs lbl{lbl_zyx.shape}", out_dir

    # 2) flips（背部在下 + 头->脚）
    ct_zyx  = ct_zyx[:, ::-1, :]   # flip Y
    lbl_zyx = lbl_zyx[:, ::-1, :]
    ct_zyx  = ct_zyx[::-1, :, :]   # flip Z
    lbl_zyx = lbl_zyx[::-1, :, :]

    # 3) resize to 512x512
    ct_zyx, lbl_zyx = resize_zyx_to_512(ct_zyx, lbl_zyx)
    z, y, x = ct_zyx.shape
    if (y, x) != (OUT_H, OUT_W):
        return False, f"resize failed -> got {(y,x)} expected {(OUT_H,OUT_W)}", out_dir

    # 4) window + norm [0,1]
    ct_zyx = window_norm01(ct_zyx, wc=WC, ww=WW)  # (Z,512,512)

    # 5) convert CT to (1,H,W,Z) and save
    ct_hwz  = np.transpose(ct_zyx, (1, 2, 0))      # (H,W,Z)
    ct_1hwz = ct_hwz[None, ...].astype(np.float32) # (1,H,W,Z)
    image_out = os.path.join(out_dir, "image.npy")
    atomic_save_npy(image_out, ct_1hwz)

    # 6) label -> (H,W,Z) and save fixed6 CSR
    lbl_hwz = np.transpose(lbl_zyx, (1, 2, 0)).astype(np.int16, copy=False)  # (H,W,Z)

    mask_out = os.path.join(out_dir, f"mask_({C_FIXED},{OUT_H},{OUT_W},{z}).npz")

    def _save_mask(pth):
        save_fixed6_onehot_csr_from_label_hwz(lbl_hwz, pth)

    atomic_save_npz(mask_out, _save_mask)

    # mapping txt
    write_fixed_mapping_txt(out_dir)

    # 7) vis
    if SAVE_VIS_64:
        vis_dir = os.path.join(out_dir, "vis_64")
        save_vis_64_pngs(ct_1hwz, vis_dir)

    # basic info
    uniq = np.unique(lbl_hwz)
    msg = f"ok | Z={z} uniq={uniq.tolist()}"
    return True, msg, out_dir

# =========================================================
# Parallel worker wrapper (must be top-level for pickle)
# =========================================================
def worker_run_one(args):
    cid, vpath, lpath = args
    out_dir = os.path.join(OUT_ROOT, f"volume-{cid}")
    try:
        ok, msg, out_dir = process_one_case(cid, vpath, lpath)
        return {
            "case_id": cid,
            "ok": bool(ok),
            "msg": str(msg),
            "out_dir": out_dir,
            "volume": vpath,
            "label": lpath,
        }
    except Exception as e:
        return {
            "case_id": cid,
            "ok": False,
            "msg": f"exception: {repr(e)}",
            "out_dir": out_dir,
            "volume": vpath,
            "label": lpath,
        }

# =========================================================
# Main
# =========================================================
def main():
    ensure_dir(OUT_ROOT)
    if not os.path.isdir(VOLUMES_DIR):
        raise RuntimeError(f"VOLUMES_DIR not found: {VOLUMES_DIR}")
    if not os.path.isdir(LABELS_DIR):
        raise RuntimeError(f"LABELS_DIR not found: {LABELS_DIR}")

    pairs, missing_vol, missing_lab = scan_cases(VOLUMES_DIR, LABELS_DIR)

    print("========== Scan CT-Org ==========")
    print(f"VOLUMES_DIR: {VOLUMES_DIR}")
    print(f"LABELS_DIR : {LABELS_DIR}")
    print(f"Found pairs: {len(pairs)}")
    if missing_vol:
        print(f"[Warn] labels exist but volume missing: {len(missing_vol)}  e.g. {missing_vol[:10]}")
    if missing_lab:
        print(f"[Warn] volumes exist but label missing: {len(missing_lab)}  e.g. {missing_lab[:10]}")
    print("=================================\n")

    tasks = [(cid, vpath, lpath) for (cid, vpath, lpath) in pairs]

    ok_cnt = 0
    fail_cnt = 0
    skip_cnt = 0
    fails = []

    if len(tasks) == 0:
        print("[Info] No paired cases found.")
    elif NUM_WORKERS <= 1 or len(tasks) == 1:
        # serial fallback
        for t in tqdm(tasks, desc="Process CT-Org cases"):
            res = worker_run_one(t)
            if res["ok"] and res["msg"].startswith("skip"):
                skip_cnt += 1
            elif res["ok"]:
                ok_cnt += 1
            else:
                fail_cnt += 1
                fails.append({
                    "case_id": res["case_id"],
                    "volume": res["volume"],
                    "label": res["label"],
                    "out_dir": res["out_dir"],
                    "error": res["msg"],
                })
    else:
        print(f"[Parallel] ProcessPoolExecutor workers={NUM_WORKERS} tasks={len(tasks)}")
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = [ex.submit(worker_run_one, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Process CT-Org cases"):
                res = fut.result()
                if res["ok"] and res["msg"].startswith("skip"):
                    skip_cnt += 1
                elif res["ok"]:
                    ok_cnt += 1
                else:
                    fail_cnt += 1
                    fails.append({
                        "case_id": res["case_id"],
                        "volume": res["volume"],
                        "label": res["label"],
                        "out_dir": res["out_dir"],
                        "error": res["msg"],
                    })

    summary = {
        "ctorg_root": CTORG_ROOT,
        "volumes_dir": VOLUMES_DIR,
        "labels_dir": LABELS_DIR,
        "out_root": OUT_ROOT,
        "window": {"WC": WC, "WW": WW},
        "target_hw": [OUT_H, OUT_W],
        "fixed_labels": {
            "order": LABELS_ORDER,
            "ids": LABEL_IDS,
            "channels": C_FIXED,
            "flatten": "on (H,W,Z), linear=(y*W+x)*Z+z (z fastest)",
        },
        "save_vis_64": bool(SAVE_VIS_64),
        "num_vis": int(NUM_VIS),
        "force_remake": bool(FORCE_REMAKE),
        "atomic_write": bool(ATOMIC_WRITE),
        "parallel": {
            "backend": "process",
            "num_workers": int(NUM_WORKERS),
            "note": "建议设置 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 防止过度并发",
        },
        "pairs": len(pairs),
        "ok": ok_cnt,
        "skip": skip_cnt,
        "fail": fail_cnt,
        "fails": fails[:200],  # 防止太大
    }
    summary_path = os.path.join(OUT_ROOT, "ct_org_build_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== Summary ==========")
    print(f"OK   : {ok_cnt}")
    print(f"Skip : {skip_cnt}")
    print(f"Fail : {fail_cnt}")
    print("Saved summary:", summary_path)
    if fail_cnt > 0:
        print("First fails:", fails[:3])
    print("=============================\n")

if __name__ == "__main__":
    main()
