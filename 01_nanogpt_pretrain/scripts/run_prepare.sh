#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../nanoGPT"
python data/shakespeare/prepare.py
