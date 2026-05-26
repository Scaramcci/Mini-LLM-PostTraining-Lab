"""Split a JSONL dataset into train and validation files."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--val-out", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise SystemExit("--val-ratio must be between 0 and 1")

    lines = [line for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise SystemExit("need at least 2 examples to split")

    rng = random.Random(args.seed)
    rng.shuffle(lines)

    val_size = max(1, round(len(lines) * args.val_ratio))
    val_lines = lines[:val_size]
    train_lines = lines[val_size:]

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    args.train_out.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    args.val_out.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    print(f"input: {len(lines)}")
    print(f"train: {len(train_lines)} -> {args.train_out}")
    print(f"val: {len(val_lines)} -> {args.val_out}")


if __name__ == "__main__":
    main()
