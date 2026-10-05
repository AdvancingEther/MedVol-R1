#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0

import os
import gc
import json
from typing import Optional, Any, List

import fire
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Seq2SeqTrainingArguments, AutoModelForImageTextToText

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


def _mkdir_parent(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _safe_load_pil(img_obj: Any) -> Image.Image:
    """
    img_obj can be:
      - PIL.Image
      - str path
    """
    if isinstance(img_obj, Image.Image):
        return img_obj.convert("RGB")
    if isinstance(img_obj, str):
        with Image.open(img_obj) as im:
            return im.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img_obj)}")


def _resize_to_bounds_square(
    im: Image.Image, image_max_pixels: int, image_min_pixels: int, multiple: int = 16
) -> Image.Image:
    """
    For stability, resize to a square side ~= sqrt(image_max_pixels), aligned to `multiple`.
    Example: image_max_pixels=262144 -> side~=512.
    """
    side = int(round((image_max_pixels ** 0.5)))
    side = max(multiple, (side // multiple) * multiple)
    if side <= 0:
        side = 512
    if im.size != (side, side):
        im = im.resize((side, side), Image.BICUBIC)
    return im


@torch.inference_mode()
def hf_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,  # keep interface; merged ckpt can leave None
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    max_new_tokens: int = 128,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    default_system: Optional[str] = None,
    enable_thinking: bool = True,
    seed: Optional[int] = None,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    batch_size: int = 4,
    limit_mm_image: int = 16,
):
    """
    HF transformers batch inference replacement for vllm_infer.py.

    Output jsonl lines:
      {"prompt": ..., "predict": ..., "label": ...}

    Key fix vs your error:
      - Qwen3-VL processor MUST receive `text=` together with `images=`.
        Calling processor(images=...) alone will crash with:
        TypeError: argument of type 'NoneType' is not iterable
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Build args via LLaMA-Factory helper (same behavior/flags style as vllm script)
    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=4,  # dataset tokenization workers
            default_system=default_system,
            enable_thinking=enable_thinking,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module.get("processor", None)
    if processor is None:
        raise ValueError(
            "Processor was not found. Please make sure your model folder contains the correct "
            "processor/preprocessor configs for Qwen3-VL (e.g., preprocessor_config.json)."
        )

    # For causal LM generation, left padding is usually safer when batching variable length prompts
    tokenizer.padding_side = "left"
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"

    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Load dataset (this step includes tokenization & mm processing in LLaMA-Factory)
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, stage="ppo", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]
    if max_samples is not None:
        train_dataset = train_dataset.select(range(min(int(max_samples), len(train_dataset))))

    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        dtype="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Pad token
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    _mkdir_parent(save_name)

    all_prompts, all_preds, all_labels = [], [], []

    # Generation kwargs
    gen_kwargs = dict(
        max_new_tokens=int(generating_args.max_new_tokens),
        repetition_penalty=float(generating_args.repetition_penalty or 1.0),
        do_sample=(float(generating_args.temperature) > 1e-8),
        temperature=float(generating_args.temperature),
        top_p=float(generating_args.top_p or 1.0),
        top_k=int(generating_args.top_k or 0),
        pad_token_id=pad_token_id,
    )

    for i in tqdm(range(0, len(train_dataset), batch_size), desc="HF batched inference"):
        batch = train_dataset[i : min(i + batch_size, len(train_dataset))]

        mm_texts: List[str] = []          # IMPORTANT: keep <image> tokens
        save_prompts: List[str] = []      # readable prompt for jsonl
        save_labels: List[str] = []
        images_batch: List[List[Image.Image]] = []

        B = len(batch["input_ids"])
        for j in range(B):
            # Build text for processor: MUST keep <image> tokens, so skip_special_tokens=False
            ids = batch["input_ids"][j]
            if isinstance(ids, torch.Tensor):
                ids_list = ids.tolist()
            else:
                ids_list = list(ids)

            mm_text = tokenizer.decode(ids_list, skip_special_tokens=False)
            mm_texts.append(mm_text)

            save_prompts.append(tokenizer.decode(ids_list, skip_special_tokens=skip_special_tokens))

            # Label text (optional)
            if "labels" in batch and batch["labels"][j] is not None:
                lab = batch["labels"][j]
                if isinstance(lab, torch.Tensor):
                    lab = lab.tolist()
                lab = list(filter(lambda x: x != IGNORE_INDEX, lab))
                save_labels.append(tokenizer.decode(lab, skip_special_tokens=skip_special_tokens))
            else:
                save_labels.append("")

            # Images: take from batch["images"][j]
            imgs_field = batch["images"][j] if "images" in batch else None
            if imgs_field is None:
                images_batch.append([])
                continue

            # LLaMA-Factory mm_plugin regularization (handles paths/PIL and size constraints)
            # NOTE: imgs_field is often list[str] (paths) or list[PIL]
            if not isinstance(imgs_field, (list, tuple)):
                imgs_field = [imgs_field]
            imgs_field = list(imgs_field)[: int(limit_mm_image)]

            reg = template_obj.mm_plugin._regularize_images(
                imgs_field, image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels
            )
            pil_images = [_resize_to_bounds_square(_safe_load_pil(x), image_max_pixels, image_min_pixels) for x in reg["images"]]
            images_batch.append(pil_images)

        # ====== CRITICAL FIX: call processor with BOTH text and images ======
        # This avoids: TypeError: argument of type 'NoneType' is not iterable
        proc_kwargs = dict(
            text=mm_texts,
            images=images_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(cutoff_len),
            do_resize=False,
        )


        def _count_substr(s: str, sub: str) -> int:
            return s.count(sub)

        def _remove_extra_image_tokens(text: str, image_token: str, keep: int) -> str:
            """
            Keep only the first `keep` occurrences of `image_token`, remove the rest.
            """
            if keep < 0:
                keep = 0
            parts = text.split(image_token)
            occ = len(parts) - 1
            if occ <= keep:
                return text
            # Re-join: keep first `keep` tokens, then concatenate the rest without tokens
            return image_token.join(parts[: keep + 1]) + "".join(parts[keep + 1 :])

        # ---- right before calling processor ----
        image_token = getattr(processor, "image_token", "<|image_pad|>")

        # per-sample align (recommended)
        for j in range(len(mm_texts)):
            n_tok = _count_substr(mm_texts[j], image_token)
            n_img = len(images_batch[j])
            if n_tok != n_img:
                k = min(n_tok, n_img)
                if n_tok > k:
                    mm_texts[j] = _remove_extra_image_tokens(mm_texts[j], image_token, k)
                if n_img > k:
                    images_batch[j] = images_batch[j][:k]

        # batch-level sanity check (this is what prevents your IndexError)
        total_tok = sum(_count_substr(t, image_token) for t in mm_texts)
        total_img = sum(len(imgs) for imgs in images_batch)
        assert total_tok == total_img, f"image_token({total_tok}) != images({total_img})"















        inputs = processor(**proc_kwargs)

        # Move to device (for sharded models, putting inputs on the first param device is typical)
        device = next(model.parameters()).device
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                # pixel_values should match model dtype
                if k == "pixel_values":
                    inputs[k] = v.to(device=device, dtype=getattr(model, "dtype", v.dtype))
                else:
                    inputs[k] = v.to(device=device)

        # Some configs may include token_type_ids; Qwen3-VL generally doesn't need it
        inputs.pop("token_type_ids", None)

        # Generate
        outputs = model.generate(**inputs, **gen_kwargs)

        # Trim the prompt part
        prompt_len = inputs["input_ids"].shape[1]
        outputs_trim = outputs[:, prompt_len:]

        preds = tokenizer.batch_decode(outputs_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        all_prompts.extend(save_prompts)
        all_preds.extend(preds)
        all_labels.extend(save_labels)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write jsonl
    with open(save_name, "w", encoding="utf-8") as f:
        for p, pred, lab in zip(all_prompts, all_preds, all_labels):
            f.write(json.dumps({"prompt": p, "predict": pred, "label": lab}, ensure_ascii=False) + "\n")

    print("*" * 70)
    print(f"{len(all_prompts)} total generated results have been saved at {save_name}.")
    print("*" * 70)


if __name__ == "__main__":
    fire.Fire(hf_infer)
