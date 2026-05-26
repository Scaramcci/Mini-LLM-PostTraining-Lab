#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-cpu}"
OUT_DIR="${2:-out-shakespeare-char}"

cd "$(dirname "$0")/../nanoGPT"
python sample.py --out_dir="$OUT_DIR" --device="$DEVICE"
