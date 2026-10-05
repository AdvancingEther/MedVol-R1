#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CT-Org -> Verl/Easy-R1 parquet generator.

Key changes vs Abd-1k:
1) Filter cases by candidate CSV (column: "case")
2) Oversample "brain" by duplicating 2 extra times (total 3 copies)
3) Description is sampled from term_dictionary.json each time (so oversampling won't force repeats)
4) Robust selected_slices retrieval:
   - Prefer <png_case_dir>/text.json["selected_slices"]
   - Else scan png folder for slice_*.png
5) Robust label existence check:
   - Use mask_*.npz CSR with optional mask_labels.json mapping to detect which labels exist in this case

NEW (for space reward):
6) Read per-label bbox from <png_case_dir>/text.json["labels"][...]["bbox_2d_list"]
   and write into answer as:
     - gt_bbox_2d_list_512 (and also bbox_list_512 for compatibility)
     - gt_key_slice (optional but helpful)
"""

import os
import json
import re
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, Sequence
from datasets import Image as ImageData
from scipy import sparse

# =========================================================
# Config (edit here)
# =========================================================
train_64_slices_folder = os.environ.get('MEDVOL_GENERATE_RL_TRAIN_64_SLICES_FOLDER', 'data/ctorg/ct_org_png')
train_npy_folder = os.environ.get('MEDVOL_GENERATE_RL_TRAIN_NPY_FOLDER', 'data/ctorg/ct_org_npy')

out_dir = os.environ.get('MEDVOL_GENERATE_RL_OUT_DIR', 'data/ctorg/verl_parquet')
dataset_name = "ctorg_refseg_verl_train"

TERM_DICT_JSON = os.environ.get('MEDVOL_GENERATE_RL_TERM_DICT_JSON', 'data_preparation/ctorg/term_dictionary.json')
RANDOM_SEED = 42

STRICT_REQUIRE_ALL_PNG = True
REQUIRE_K_IMAGES: Optional[int] = 64   # CTOrg 通常就是 64；若你想允许变长，改成 None
LIMIT = -1  # -1 means no limit

LABEL_NAME_TO_ID = {
    "liver": 1,
    "bladder": 2,
    "lungs": 3,
    "kidneys": 4,
    "bone": 5,
    "brain": 6,
}
ID_TO_LABEL_NAME = {v: k for k, v in LABEL_NAME_TO_ID.items()}

# candidate list
candidate_train_volume_csv = os.environ.get('MEDVOL_GENERATE_RL_CANDIDATE_TRAIN_VOLUME_CSV', 'data/ctorg/summary_all_train.csv')
CANDIDATE_CASE_KEY = "case"

# oversampling
OVERSAMPLE_LABEL = "brain"
OVERSAMPLE_EXTRA_COPIES = 2   # extra copies, total copies = 1 + 2 = 3

# deterministic per-case label emission order
DETERMINISTIC_PER_SAMPLE = True

# If true, skip emitting samples whose text.json doesn't provide bbox for that label
SKIP_IF_NO_GT_BBOX = True

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
    if len(cands) == 1:
        return cands[0]
    for c in cands:
        if c.name.startswith("mask_("):
            return c
    return cands[0]

def _infer_volume_shape_from_image_npy(image_npy: Path) -> Optional[Tuple[int, int, int]]:
    """
    image.npy could be (1,Z,H,W) or (Z,H,W) or (H,W,Z).
    Return (Z,H,W).
    """
    if not image_npy.exists():
        return None
    arr = np.load(str(image_npy), mmap_mode="r")
    arr = np.asarray(arr)

    if arr.ndim == 4:
        # (1,Z,H,W)
        return (int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3]))
    if arr.ndim == 3:
        # common: (Z,H,W)
        if arr.shape[1] == 512 and arr.shape[2] == 512:
            return (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
        # sometimes: (H,W,Z)
        if arr.shape[0] == 512 and arr.shape[1] == 512:
            return (int(arr.shape[2]), int(arr.shape[0]), int(arr.shape[1]))
        # fallback
        return (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
    return None

def load_term_dictionary(p: str) -> Dict[str, List[str]]:
    if not p or (not os.path.exists(p)):
        return {}
    d = _load_json(p)
    out: Dict[str, List[str]] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(k, str) and isinstance(v, list):
                out[k.strip().lower()] = [str(x).strip() for x in v if str(x).strip()]
    return out

def pick_random_description(term_dict: Dict[str, List[str]], label_name: str, rng: random.Random) -> str:
    key = str(label_name).strip().lower()
    if key in term_dict and len(term_dict[key]) > 0:
        return rng.choice(term_dict[key]).strip()
    return key

def _scan_selected_slices_from_png(case_png_dir: Path) -> List[int]:
    """
    If no text.json selected_slices, scan for slice_###.png
    """
    out = []
    for p in case_png_dir.glob("slice_*.png"):
        m = re.search(r"slice_(\d+)\.png$", p.name)
        if m:
            out.append(int(m.group(1)))
    out = sorted(list(set(out)))
    return out

def _resolve_png_case_dir(root_png: Path, case_id: str) -> Optional[Path]:
    """
    Return the directory that actually contains slice_*.png.
    Tries:
      1) root_png/case_id/png_64slices
      2) any subdir under root_png/case_id that contains slice_*.png
      3) root_png/case_id itself (if contains slice_*.png)
    """
    case_root = root_png / case_id
    if not case_root.exists():
        return None

    # common convention
    cand = case_root / "png_64slices"
    if cand.exists() and any(cand.glob("slice_*.png")):
        return cand

    # scan children
    for d in sorted([p for p in case_root.iterdir() if p.is_dir()]):
        if any(d.glob("slice_*.png")):
            return d

    # fallback
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

    # fallback: scan png filenames
    out = []
    for p in case_png_dir.glob("slice_*.png"):
        m = re.search(r"slice_(\d+)\.png$", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(list(set(out)))

def _load_channel_mapping(case_npy_dir: Path) -> Optional[List[int]]:
    """
    Optional: mask_labels.json contains channel_to_label_id list.
    """
    p = case_npy_dir / "mask_labels.json"
    if not p.exists():
        return None
    try:
        obj = _load_json(str(p))
        m = obj.get("channel_to_label_id", None)
        if isinstance(m, list):
            return [int(x) for x in m]
    except Exception:
        return None
    return None

def _infer_ZHW_from_csr(mat: sparse.csr_matrix) -> Tuple[int, int, int]:
    C, N = mat.shape
    HW = 512 * 512
    if N % HW != 0:
        raise ValueError(f"CSR N not divisible by 512*512: shape={mat.shape}")
    Z = N // HW
    return int(Z), 512, 512

def _label_to_channel_index(
    label_id: int,
    C: int,
    channel_to_label_id: Optional[List[int]],
) -> Optional[int]:
    """
    Map desired label_id to channel index in CSR.
    If mapping exists, invert it.
    Else assume channel i corresponds to label_id=i+1 (legacy).
    """
    if channel_to_label_id is not None and len(channel_to_label_id) == C:
        for ch, lid in enumerate(channel_to_label_id):
            if int(lid) == int(label_id):
                return int(ch)
        return None
    # fallback: label_id 1..C
    ch = int(label_id) - 1
    if 0 <= ch < C:
        return ch
    return None

def _case_has_label(
    mat: sparse.csr_matrix,
    label_id: int,
    channel_to_label_id: Optional[List[int]],
) -> bool:
    C, _ = mat.shape
    ch = _label_to_channel_index(label_id, C, channel_to_label_id)
    if ch is None:
        return False
    row = mat.getrow(ch)
    return row.nnz > 0

def _deterministic_hash_prob(key: str) -> float:
    """
    Stable hash -> [0,1).
    """
    h = 0
    for ch in key:
        h = (h * 131 + ord(ch)) % 1000003
    return (h % 1000000) / 1000000.0

def _load_candidate_cases(csv_path: str, key: str) -> List[str]:
    df = pd.read_csv(csv_path)
    if key not in df.columns:
        raise ValueError(f"candidate csv missing column '{key}'. got columns={list(df.columns)[:20]}")
    out = []
    for v in df[key].tolist():
        s = str(v).strip()
        if s:
            out.append(s)
    out = sorted(list(set(out)))
    return out

# ---------- NEW: read per-label bbox/key_slice from text.json ----------
def _norm_label_name(s: str) -> str:
    return str(s).strip().lower()

def _pick_text_json_path(case_png_dir: Path, case_root_dir: Path) -> Optional[Path]:
    for tj in [case_png_dir / "text.json", case_root_dir / "text.json"]:
        if tj.exists():
            return tj
    return None

def _read_label_bbox_from_text_json(
    text_json_path: Optional[Path],
    label_name: str,
) -> Tuple[Optional[int], List[List[int]]]:
    """
    From your provided schema:
      obj["labels"][i]["label_name"] ~ "Liver" / "Kidneys" ...
      obj["labels"][i]["present"] bool
      obj["labels"][i]["key_slice"] int
      obj["labels"][i]["bbox_2d_list"] list of [x1,y1,x2,y2] in 512 coords

    Return:
      gt_key_slice, gt_bbox_2d_list_512
    """
    if text_json_path is None or (not text_json_path.exists()):
        return None, []
    try:
        obj = _load_json(str(text_json_path))
    except Exception:
        return None, []

    labels = obj.get("labels", None)
    if not isinstance(labels, list):
        return None, []

    target = _norm_label_name(label_name)
    for it in labels:
        if not isinstance(it, dict):
            continue
        ln = _norm_label_name(it.get("label_name", ""))
        if ln != target:
            continue

        if it.get("present", True) is False:
            return None, []

        ks = it.get("key_slice", None)
        try:
            ks = int(ks) if ks is not None else None
        except Exception:
            ks = None

        bbl = it.get("bbox_2d_list", [])
        if not isinstance(bbl, list):
            bbl = []

        out_boxes: List[List[int]] = []
        for b in bbl:
            if not (isinstance(b, list) and len(b) == 4):
                continue
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in b]
            except Exception:
                continue

            # clamp to [0,511] and enforce x1<x2, y1<y2
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

        return ks, out_boxes

    return None, []

# =========================================================
# Generator
# =========================================================
def generate_data() -> Iterator[Dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)

    root_png = Path(train_64_slices_folder)
    root_npy = Path(train_npy_folder)
    term_dict = load_term_dictionary(TERM_DICT_JSON)

    # candidate list
    candidate_cases = _load_candidate_cases(candidate_train_volume_csv, CANDIDATE_CASE_KEY)

    # stats
    total_candidates = len(candidate_cases)
    used_cases = 0
    skipped_missing_png_dir = 0
    skipped_missing_npy_dir = 0
    skipped_missing_files = 0
    skipped_bad_selected = 0
    skipped_missing_png = 0
    skipped_no_text_json = 0
    skipped_no_gt_bbox = 0

    emitted = 0

    for case_id in candidate_cases:
        if LIMIT > 0 and emitted >= LIMIT:
            break

        case_root_png_dir = root_png / case_id
        case_npy_dir = root_npy / case_id
        case_png_dir = _resolve_png_case_dir(root_png, case_id)

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

        # selected_slices
        selected_slices = _load_selected_slices(case_png_dir, case_root_png_dir)
        if len(selected_slices) == 0:
            skipped_bad_selected += 1
            continue
        if REQUIRE_K_IMAGES is not None and len(selected_slices) != int(REQUIRE_K_IMAGES):
            skipped_bad_selected += 1
            continue

        # png paths
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

        # pick text.json for bbox/key_slice
        text_json_path = _pick_text_json_path(case_png_dir, case_root_png_dir)
        if text_json_path is None:
            skipped_no_text_json += 1
            # 你也可以选择 continue；这里我们继续也行，但后面会因无 bbox 被 skip（若 SKIP_IF_NO_GT_BBOX）
            # continue

        # load mask CSR and mapping
        mat = sparse.load_npz(str(mask_npz)).tocsr()
        channel_map = _load_channel_mapping(case_npy_dir)
        C, _ = mat.shape
        z2, _, _ = _infer_ZHW_from_csr(mat)
        if z2 != Z:
            Z = z2

        # which labels exist in this case
        present_labels: List[str] = []
        for lab_name, lab_id in LABEL_NAME_TO_ID.items():
            if _case_has_label(mat, int(lab_id), channel_map):
                present_labels.append(lab_name)

        if len(present_labels) == 0:
            continue

        used_cases += 1

        # emit per (case,label)
        for label_name in present_labels:
            label_id = int(LABEL_NAME_TO_ID[label_name])

            # NEW: read gt bbox/key_slice from text.json
            gt_key_slice, gt_bbox_2d_list_512 = _read_label_bbox_from_text_json(
                text_json_path=text_json_path,
                label_name=label_name,
            )
            if SKIP_IF_NO_GT_BBOX and len(gt_bbox_2d_list_512) == 0:
                skipped_no_gt_bbox += 1
                continue

            # oversample brain: total copies = 1 + extra
            copies = 1
            if label_name == OVERSAMPLE_LABEL:
                copies = 1 + int(OVERSAMPLE_EXTRA_COPIES)

            for copy_idx in range(copies):
                # per-copy rng that is deterministic w.r.t (case,label,copy) if desired
                if DETERMINISTIC_PER_SAMPLE:
                    u = _deterministic_hash_prob(f"{case_id}::{label_name}::{copy_idx}::{RANDOM_SEED}")
                    local_seed = int(u * 10_000_000) + 12345
                    local_rng = random.Random(local_seed)
                else:
                    local_rng = rng

                desc = pick_random_description(term_dict, label_name, local_rng)

                answer_dict = {
                    "id": f"{case_id}__{label_name}__copy{copy_idx}",
                    "case_id": case_id,
                    "label_name": label_name,
                    "label_id": label_id,
                    "selected_slices": kept_slices,
                    "volume_shape": [int(Z), int(H), int(W)],
                    "canon_size_xy": [512, 512],

                    # reward paths: keep RELATIVE inside case_npy_dir
                    "image_rel_path": str(Path(case_id) / "image.npy"),
                    "mask_rel_path": str(Path(case_id) / Path(mask_npz.name)),

                    # NEW keys for reward:
                    "gt_key_slice": gt_key_slice,
                    "gt_bbox_2d_list_512": gt_bbox_2d_list_512,
                    # compatibility (你的 reward 会 fallback 读 bbox_list_512)
                    "bbox_list_512": gt_bbox_2d_list_512,

                    "extra": {
                        "candidate_csv": os.path.abspath(candidate_train_volume_csv),
                        "oversample": {
                            "label": OVERSAMPLE_LABEL,
                            "extra_copies": int(OVERSAMPLE_EXTRA_COPIES),
                            "copy_idx": int(copy_idx),
                        },
                        "png_case_dir": str(Path(case_id)),
                        "text_json_path": str(text_json_path) if text_json_path is not None else None,
                    },
                }

                yield {
                    "images": images,
                    "problem": str(desc),
                    "answer": json.dumps(answer_dict, ensure_ascii=False),
                }
                emitted += 1

    print("========== CTOrg parquet generation stats ==========")
    print(f"candidate_cases_total      : {total_candidates}")
    print(f"used_cases                : {used_cases}")
    print(f"skipped_missing_png_dir    : {skipped_missing_png_dir}")
    print(f"skipped_missing_npy_dir    : {skipped_missing_npy_dir}")
    print(f"skipped_missing_files      : {skipped_missing_files}")
    print(f"skipped_bad_selected       : {skipped_bad_selected}")
    print(f"skipped_missing_png        : {skipped_missing_png}")
    print(f"skipped_no_text_json       : {skipped_no_text_json}")
    print(f"skipped_no_gt_bbox         : {skipped_no_gt_bbox}")
    print(f"total_samples_emitted      : {emitted}")
    print("Note: 'brain' is oversampled by (1 + OVERSAMPLE_EXTRA_COPIES).")

# =========================================================
# Main
# =========================================================
def main():
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ds = Dataset.from_generator(generate_data).cast_column("images", Sequence(ImageData()))
    print(ds)

    out_path = os.path.join(out_dir, f"{dataset_name}.parquet")
    ds.to_parquet(out_path)
    print(f"[OK] Wrote: {out_path}")

if __name__ == "__main__":
    main()
