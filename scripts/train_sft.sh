#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TASK="${1:-kits23}"
[[ "$TASK" == kits23 || "$TASK" == ctorg ]] || { echo "Usage: $0 {kits23|ctorg} [key=value ...]" >&2; exit 2; }
if (( $# )); then shift; fi
exec llamafactory-cli train "configs/sft_${TASK}.yaml" "$@"
