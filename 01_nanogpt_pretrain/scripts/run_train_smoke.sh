#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-cpu}"
MAX_ITERS="${2:-200}"

cd "$(dirname "$0")/../nanoGPT"
python train.py config/train_shakespeare_char.py --device="$DEVICE" --compile=False --max_iters="$MAX_ITERS"
