#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-cpu}"
MAX_ITERS="${2:-200}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

cd "$(dirname "$0")/../nanoGPT"
"$PYTHON_BIN" train.py config/train_shakespeare_char.py --device="$DEVICE" --compile=False --max_iters="$MAX_ITERS"
