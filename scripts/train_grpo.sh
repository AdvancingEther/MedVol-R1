#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TASK="${1:-kits23}"
[[ "$TASK" == kits23 || "$TASK" == ctorg ]] || { echo "Usage: $0 {kits23|ctorg} [key=value ...]" >&2; exit 2; }
if (( $# )); then shift; fi
export PYTHONPATH="$ROOT/third_party/EasyR1:$ROOT/third_party/MedSAM2${PYTHONPATH:+:$PYTHONPATH}"
export CTORG_NPY_ROOT="${CTORG_NPY_ROOT:-$ROOT/data/ctorg/ct_org_npy}"
export KITS23_NPY_ROOT="${KITS23_NPY_ROOT:-$ROOT/data/kits23/kits23_npy_m3d}"
export MEDSAM2_CHECKPOINT="${MEDSAM2_CHECKPOINT:-$ROOT/checkpoints/MedSAM2_2411.pt}"
export RAY_DISABLE_DASHBOARD=1
exec python -m verl.trainer.main "config=configs/grpo_${TASK}.yaml" "$@"
