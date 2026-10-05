#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TASK="${1:?Usage: infer.sh TASK MODEL_PATH [--option value ...]}"
MODEL="${2:?Provide a merged Hugging Face model path}"
[[ "$TASK" == kits23 || "$TASK" == ctorg ]] || { echo "Unknown task: $TASK" >&2; exit 2; }
shift 2
DATASET=kits23_frames_refseg_test
[[ "$TASK" != ctorg ]] || DATASET=ct_org_frames_refseg_test
mkdir -p "outputs/$TASK"
exec python third_party/LLaMA-Factory/scripts/vllm_infer.py   --model_name_or_path "$MODEL" --dataset "$DATASET"   --dataset_dir "data/$TASK" --template qwen3_vl   --cutoff_len 24576 --max_new_tokens 256   --image_max_pixels 262144 --image_min_pixels 4096 --batch_size 4   --save_name "outputs/$TASK/predictions.jsonl"   --vllm_config '{"enforce_eager":true,"disable_custom_all_reduce":true,"limit_mm_per_prompt":{"image":64}}' "$@"
