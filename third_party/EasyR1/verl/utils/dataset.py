# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
"""
CTOrg-adapted RLHF dataset (Qwen-VL style):
- Build prompt from (problem + answer.selected_slices + images) ONLY.
- CTOrg prompt wording (not abdomen-only).
- IMPORTANT: slice_ids fallback MUST align with images:
  1) prefer answer.selected_slices if length matches images
  2) else parse slice ids from image file names: slice_XXX.png
  3) else fallback to range(n_imgs) (with warning)

Drop-in replacement for your previous file content.
"""

import math
import os
import json
import re
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF


# =========================================================
# Collate
# =========================================================
def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


# =========================================================
# Image / Video processing
# =========================================================
def process_image(
    image: Union[dict[str, Any], ImageObject, str, bytes],
    min_pixels: Optional[int],
    max_pixels: Optional[int],
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        # HF Image dict: {"path": "...", "bytes": None} or {"path": None, "bytes": ...}
        if image.get("bytes", None) is not None:
            image = Image.open(BytesIO(image["bytes"]))
        else:
            image = Image.open(image["path"])
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()

    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def process_video(
    video: str,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    video_fps: float,
    return_fps: bool = False,
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


# =========================================================
# Answer parsing
# =========================================================
def _safe_parse_answer(ans_raw: Any) -> dict[str, Any]:
    if isinstance(ans_raw, str):
        try:
            return json.loads(ans_raw)
        except Exception:
            return {}
    if isinstance(ans_raw, dict):
        return ans_raw
    return {}


# =========================================================
# Slice id inference / prompt
# =========================================================
_SLICE_RE = re.compile(r"slice_(\d+)\.(png|jpg|jpeg|bmp|webp)$", re.IGNORECASE)

def _infer_slice_ids_from_images(images: Any) -> list[int]:
    """
    images can be list of:
      - str paths
      - HF Image dict {"path": "...", "bytes": None}
    returns [] if cannot parse or images not list
    """
    if not isinstance(images, list):
        return []
    out: list[int] = []
    for it in images:
        p = None
        if isinstance(it, str):
            p = it
        elif isinstance(it, dict) and isinstance(it.get("path", None), str):
            p = it["path"]
        if not p:
            continue
        m = _SLICE_RE.search(os.path.basename(p))
        if m:
            out.append(int(m.group(1)))
    return out


def _build_prompt_from_problem_and_slices(description_text: str, slice_ids: list[int]) -> str:
    # slice tag + <image> placeholder
    slice_blocks = []
    for sid in slice_ids:
        slice_blocks.append(f"<slice {sid}>\n<image>\n")
    slices_prefix = "".join(slice_blocks)

    instruction = (
        "You are given multiple axial CT slices"
        "These slices may come from head, chest, or abdomen.\n"
        f"Referring description: {description_text}\n\n"
        "Task:\n"
        "1) Choose ONE key slice where the target anatomical structure is most clearly visible.\n"
        "2) On that key slice, output bounding box(es) for ALL visible target instances.\n\n"
        "Output format (STRICT):\n"
        "Return EXACTLY two blocks in this order and NOTHING else:\n"
        "1) one <think>...</think>\n"
        "2) one <answer>...</answer>\n\n"
        "<think>\n"
        "Briefly explain your choice (1-2 short sentences).\n"
        "</think>\n\n"
        "<answer>\n"
        "Output a JSON array with EXACTLY ONE object:\n"
        "[{\"slice\": N, \"bbox_2d_list\": [[x1,y1,x2,y2], ...]}]\n"
        "</answer>\n\n"
        "Rules:\n"
        "- N must be one of the provided <slice N> tags.\n"
        "- bbox_2d_list can contain one or multiple boxes.\n"
        "- If there are too many small fragments (e.g., bone), output ONE tight box covering the main visible region.\n"
        "- Coordinates are integers normalized to [0,1000].\n"
        "- For each box: x1<x2 and y1<y2.\n"
        "- Do NOT output any extra keys.\n\n"
        "Example (only):\n"
        "<think>Your reason here.</think>\n"
        "<answer>[{\"slice\": 123, \"bbox_2d_list\": [[100,200,300,400],[500,100,700,350]]}]</answer>\n"
    )
    return slices_prefix + instruction


# =========================================================
# RLHF Dataset
# =========================================================
class RLHFDataset(Dataset):
    """
    CTOrg-adapted:
    - Always build prompt from (problem + answer.selected_slices + images)
    - slice tag alignment priority:
        (1) answer.selected_slices if len matches images
        (2) parse slice ids from image filename slice_XXX.png
        (3) fallback range(n_imgs) + warn
    - multi_modal_data only returns {"images": images} or {"videos": videos}
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        # ✅ add optional debug
        debug_slice_alignment: bool = False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.debug_slice_alignment = bool(debug_slice_alignment)

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        description_text = str(example.get("problem", "")).strip()

        images = example.get(self.image_key, None)
        videos = example.get(self.video_key, None)

        if isinstance(images, list):
            n_imgs = len(images)
        elif isinstance(videos, list):
            n_imgs = len(videos)
        else:
            n_imgs = 0

        ans = _safe_parse_answer(example.get(self.answer_key, {}))
        slice_ids = ans.get("selected_slices", [])
        if not isinstance(slice_ids, list):
            slice_ids = []

        # to int
        cleaned: list[int] = []
        for x in slice_ids:
            try:
                cleaned.append(int(x))
            except Exception:
                pass
        slice_ids = cleaned

        # ✅ CTOrg alignment: prefer answer.selected_slices only if len matches images
        fallback_reason = None
        if self.image_key in example and isinstance(images, list):
            if len(slice_ids) != len(images):
                img_slice_ids = _infer_slice_ids_from_images(images)
                if len(img_slice_ids) == len(images):
                    slice_ids = img_slice_ids
                    fallback_reason = "use_slice_ids_from_image_paths"
                else:
                    slice_ids = list(range(len(images)))
                    fallback_reason = "fallback_range_n_imgs"
            else:
                fallback_reason = "use_answer_selected_slices"

            if self.debug_slice_alignment and fallback_reason in ("use_slice_ids_from_image_paths", "fallback_range_n_imgs"):
                case_id = None
                try:
                    case_id = ans.get("case_id", None)
                except Exception:
                    case_id = None
                print(
                    f"[SliceAlign][{fallback_reason}] case_id={case_id} "
                    f"len(images)={len(images)} len(answer.selected_slices)={len(cleaned)} "
                    f"parsed_from_paths={len(_infer_slice_ids_from_images(images))}",
                    flush=True,
                )

        # If not image mode (pure text / video), allow old fallback
        if (self.image_key not in example) and n_imgs > 0 and len(slice_ids) != n_imgs:
            slice_ids = list(range(n_imgs))

        prompt_str = _build_prompt_from_problem_and_slices(description_text=description_text, slice_ids=slice_ids)

        # optional external format_prompt wrapper
        if self.format_prompt:
            prompt_str = Template(self.format_prompt.strip()).render(content=prompt_str)

        if self.image_key in example:
            content_list = []
            for i, chunk in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})
                if chunk:
                    content_list.append({"type": "text", "text": chunk})
            return [{"role": "user", "content": content_list}]

        if self.video_key in example:
            content_list = []
            for i, chunk in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})
                if chunk:
                    content_list.append({"type": "text", "text": chunk})
            return [{"role": "user", "content": content_list}]

        return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length

        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length

        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)

        # keep old behavior: pop prompt_key even if exists (we don't use it)
        example.pop(self.prompt_key, None)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

            # ✅ multi_modal_data only keeps "images"
            example["multi_modal_data"] = {"images": images}

        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt")
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}

        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        # Qwen-VL mrope logic (keep original)
        if (
            self.processor is not None
            and hasattr(self.processor, "image_processor")
            and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        ):
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids

        example["ground_truth"] = example.pop(self.answer_key)
        return example
