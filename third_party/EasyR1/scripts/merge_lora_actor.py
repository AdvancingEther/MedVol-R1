#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge GRPO actor FSDP shards + export LoRA adapter / merged full model.

What it does:
1) Merge model_world_size_*_rank_*.pt under ACTOR_DIR into a single CPU state_dict (bf16).
2) Detect whether the merged state_dict looks like a PEFT-wrapped model (keys with 'base_model.').
3) Export a clean LoRA adapter directory (adapter_config.json + adapter_model.safetensors).
   - Prefer using existing lora_adapter/ files if present.
   - Otherwise, try to extract LoRA weights from merged actor state_dict.
4) Optionally merge LoRA into base model and save a full HF model (for vLLM stability).

Dependencies:
- torch, numpy
- transformers
- peft (only needed if DO_MERGE_INTO_BASE=True)
- safetensors

Notes:
- This script assumes FSDP-only (or DDP+FSDP replicate) sharding, no TP.
"""

import os
import re
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.distributed._tensor import DTensor, Placement, Shard

from safetensors.torch import save_file as save_safetensors
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer,AutoProcessor

from peft import PeftModel

# =========================================================
# Config (edit here)
# =========================================================
# 你的 global_step_660/actor 目录（里面有 model_world_size_*_rank_*.pt 和 lora_adapter/）
ACTOR_DIR = os.environ.get('MEDVOL_MERGE_LORA_ACTOR_ACTOR_DIR', 'checkpoints/rl/qwen3_grounding_grpo_ctorg/qwen3_vl_4b_ctorg_grounding_grpo_slice64_bs4_rollout_4_full_reward_w_o_space_epoch3_medsam_2411/global_step_300/actor')

# base 模型（用于可选 merge into base）
BASE_MODEL_DIR = os.environ.get('MEDVOL_MERGE_LORA_ACTOR_BASE_MODEL_DIR', 'outputs/merged/qwen3vl_CTOrg_sft_lora_64_lr_2e-5_epoch_2_slice_64_res_336_ckpt_912')

# 输出目录
OUT_DIR = os.environ.get('MEDVOL_MERGE_LORA_ACTOR_OUT_DIR', 'outputs/merged/qwen3_grounding_grpo_ctorg/global_step_300')
# 输出的 adapter 子目录名
OUT_ADAPTER_DIRNAME = "lora_adapter_exported"
# 输出 merged full model 子目录名
OUT_MERGED_MODEL_DIRNAME = "merged_full_model"

# 是否把 LoRA merge 进 base，输出完整 HF 模型（vLLM 最稳）
DO_MERGE_INTO_BASE = True

# 合并出来的权重 dtype（推荐 bf16）
TARGET_DTYPE = torch.bfloat16

# Thread workers for loading shards
MAX_WORKERS = 16
# =========================================================


def merge_by_placement(tensors: list[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def find_world_size_and_rank0(actor_dir: str):
    world_size = None
    for fn in os.listdir(actor_dir):
        m = re.match(r"model_world_size_(\d+)_rank_0\.pt", fn)
        if m:
            world_size = int(m.group(1))
            break
    if world_size is None:
        raise FileNotFoundError("Cannot find model_world_size_*_rank_0.pt under ACTOR_DIR")
    rank0_path = os.path.join(actor_dir, f"model_world_size_{world_size}_rank_0.pt")
    return world_size, rank0_path


def load_shard(actor_dir: str, world_size: int, rank: int):
    p = os.path.join(actor_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    sd = torch.load(p, map_location="cpu", weights_only=False)
    return sd


def merge_fsdp_shards(actor_dir: str):
    world_size, rank0_path = find_world_size_and_rank0(actor_dir)
    print(f"[Info] world_size={world_size}")
    sd0 = torch.load(rank0_path, map_location="cpu", weights_only=False)

    pivot_key = sorted(sd0.keys())[0]
    w = sd0[pivot_key]
    if isinstance(w, DTensor):
        mesh = w.device_mesh.mesh
        mesh_dim_names = w.device_mesh.mesh_dim_names
    else:
        mesh = np.array([world_size], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    print(f"[Info] device mesh={mesh}, mesh_dim_names={mesh_dim_names}")
    assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names={mesh_dim_names}"

    # FSDP-only supported (no TP)
    total_shards = int(mesh.shape[-1])
    print(f"[Info] total_shards={total_shards}")

    model_state_dict_lst = [sd0] + [None] * (total_shards - 1)

    def _worker(r):
        model_state_dict_lst[r] = load_shard(actor_dir, world_size, r)

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, os.cpu_count() or 8)) as ex:
        for r in range(1, total_shards):
            ex.submit(_worker, r)

    # collect tensors per key
    merged_lists = {}
    param_placements = {}

    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        merged_lists[key] = []
        for r, msd in enumerate(model_state_dict_lst):
            if msd is None:
                raise RuntimeError(f"Shard rank {r} not loaded?")
            if key not in msd:
                raise KeyError(f"Missing key {key} in shard rank {r}")
            t = msd.pop(key)
            if isinstance(t, DTensor):
                merged_lists[key].append(t._local_tensor.to(dtype=TARGET_DTYPE))
                placements = tuple(t.placements)
                if mesh_dim_names[0] == "ddp":
                    placements = placements[1:]  # drop ddp replicate dim
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                merged_lists[key].append(t.to(dtype=TARGET_DTYPE))

    del model_state_dict_lst

    # merge per key
    out_sd = {}
    for key in sorted(merged_lists.keys()):
        shards = merged_lists[key]
        if key in param_placements:
            placements = param_placements[key]
            assert len(placements) == 1, "Only FSDP 1-D sharding supported"
            out_sd[key] = merge_by_placement(shards, placements[0])
        else:
            # fallback: concat dim0
            out_sd[key] = torch.cat(shards, dim=0).contiguous()

    print("[Info] FSDP merge completed.")
    return out_sd


def copy_adapter_dir(src_adapter_dir: str, dst_adapter_dir: str):
    os.makedirs(dst_adapter_dir, exist_ok=True)
    # copy adapter_config.json and any adapter_model*.safetensors/bin
    for fn in os.listdir(src_adapter_dir):
        if fn.startswith("adapter_") or fn in ("README.md", "special_tokens_map.json"):
            shutil.copy2(os.path.join(src_adapter_dir, fn), os.path.join(dst_adapter_dir, fn))
        if fn.endswith(".safetensors") or fn.endswith(".bin") or fn.endswith(".json"):
            # keep typical PEFT artifacts
            if fn in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
                shutil.copy2(os.path.join(src_adapter_dir, fn), os.path.join(dst_adapter_dir, fn))


def extract_lora_tensors_from_state_dict(state_dict: dict):
    """
    Try to extract LoRA tensors from a PEFT-wrapped state dict.
    We keep only keys that contain '.lora_A.' or '.lora_B.' (and sometimes 'lora_embedding').
    """
    lora_keys = [k for k in state_dict.keys() if ("lora_A" in k or "lora_B" in k or "lora_embedding" in k)]
    if not lora_keys:
        return None

    adapter_tensors = {k: state_dict[k].cpu() for k in lora_keys}
    return adapter_tensors


def save_adapter_from_extracted(adapter_tensors: dict, src_adapter_dir: str, dst_adapter_dir: str):
    """
    Save adapter_config.json from src_adapter_dir and adapter_model.safetensors built from adapter_tensors.
    """
    os.makedirs(dst_adapter_dir, exist_ok=True)

    cfg_src = os.path.join(src_adapter_dir, "adapter_config.json")
    if not os.path.isfile(cfg_src):
        raise FileNotFoundError(f"Missing adapter_config.json in {src_adapter_dir}")
    shutil.copy2(cfg_src, os.path.join(dst_adapter_dir, "adapter_config.json"))

    # some toolchains also expect tokenizer-related json; optional copy
    for fn in ("special_tokens_map.json", "tokenizer_config.json"):
        p = os.path.join(src_adapter_dir, fn)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst_adapter_dir, fn))

    out_path = os.path.join(dst_adapter_dir, "adapter_model.safetensors")
    save_safetensors(adapter_tensors, out_path)
    print(f"[OK] Saved extracted adapter_model.safetensors with {len(adapter_tensors)} tensors.")


def merge_adapter_into_base(base_model_dir: str, adapter_dir: str, out_model_dir: str):
    os.makedirs(out_model_dir, exist_ok=True)

    # ✅ 关键：processor 也要保存（里面会写 preprocessor_config.json）
    processor = AutoProcessor.from_pretrained(base_model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)

    base = AutoModelForImageTextToText.from_pretrained(
        base_model_dir, trust_remote_code=True, torch_dtype="auto", device_map="cpu"
    )
    m = PeftModel.from_pretrained(base, adapter_dir)
    m = m.merge_and_unload()

    m.save_pretrained(out_model_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_model_dir)
    processor.save_pretrained(out_model_dir)   # ✅ 会生成 preprocessor_config.json / processor_config.json 等


def main():
    actor_dir = ACTOR_DIR
    src_adapter_dir = os.path.join(actor_dir, "lora_adapter")
    out_dir = OUT_DIR
    out_adapter_dir = os.path.join(out_dir, OUT_ADAPTER_DIRNAME)
    out_merged_model_dir = os.path.join(out_dir, OUT_MERGED_MODEL_DIRNAME)

    os.makedirs(out_dir, exist_ok=True)

    # Step 1: merge actor shards (optional, but useful for debugging / fallback extraction)
    merged_sd = None
    if any(re.match(r"model_world_size_(\d+)_rank_0\.pt", fn) for fn in os.listdir(actor_dir)):
        merged_sd = merge_fsdp_shards(actor_dir)

    # Step 2: export adapter
    if os.path.isfile(os.path.join(src_adapter_dir, "adapter_config.json")):
        # Prefer direct copy if adapter_model exists
        has_adapter_weights = (
            os.path.isfile(os.path.join(src_adapter_dir, "adapter_model.safetensors"))
            or os.path.isfile(os.path.join(src_adapter_dir, "adapter_model.bin"))
        )
        if has_adapter_weights:
            print("[Info] Found lora_adapter/ with adapter_config + adapter_model. Copying as exported adapter...")
            if os.path.exists(out_adapter_dir):
                shutil.rmtree(out_adapter_dir)
            os.makedirs(out_adapter_dir, exist_ok=True)
            # copy all files in adapter dir (safe)
            for fn in os.listdir(src_adapter_dir):
                shutil.copy2(os.path.join(src_adapter_dir, fn), os.path.join(out_adapter_dir, fn))
        else:
            # No adapter_model file; try extracting from merged state dict
            if merged_sd is None:
                raise RuntimeError("No merged state_dict available to extract LoRA tensors.")
            adapter_tensors = extract_lora_tensors_from_state_dict(merged_sd)
            if adapter_tensors is None:
                raise RuntimeError("Cannot find LoRA tensors in merged actor state_dict.")
            print("[Info] lora_adapter exists but no adapter_model.* found; extracted from merged actor state_dict.")
            if os.path.exists(out_adapter_dir):
                shutil.rmtree(out_adapter_dir)
            save_adapter_from_extracted(adapter_tensors, src_adapter_dir, out_adapter_dir)
    else:
        raise FileNotFoundError(f"Missing {src_adapter_dir}/adapter_config.json")

    print(f"[OK] Exported adapter dir: {out_adapter_dir}")

    # Step 3: optionally merge into base
    if DO_MERGE_INTO_BASE:
        merge_adapter_into_base(BASE_MODEL_DIR, out_adapter_dir, out_merged_model_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
