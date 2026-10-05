#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:?Usage: export_rl.sh {full|lora} ACTOR_DIR [BASE_MODEL OUTPUT_DIR]}"
ACTOR="${2:?Provide the actor checkpoint directory}"
[[ -d "$ACTOR" ]] || { echo "Missing actor directory: $ACTOR" >&2; exit 2; }
case "$MODE" in
  full)
    exec python third_party/EasyR1/scripts/model_merger.py --local_dir "$ACTOR"
    ;;
  lora)
    export MEDVOL_MERGE_LORA_ACTOR_ACTOR_DIR="$ACTOR"
    export MEDVOL_MERGE_LORA_ACTOR_BASE_MODEL_DIR="${3:?Provide the base SFT model}"
    export MEDVOL_MERGE_LORA_ACTOR_OUT_DIR="${4:?Provide a new output directory}"
    [[ ! -e "$MEDVOL_MERGE_LORA_ACTOR_OUT_DIR" ]] || { echo "Output already exists; use a new directory." >&2; exit 2; }
    exec python third_party/EasyR1/scripts/merge_lora_actor.py
    ;;
  *) echo "Mode must be full or lora" >&2; exit 2 ;;
esac
