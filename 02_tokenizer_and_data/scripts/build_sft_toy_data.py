"""Build a tiny SFT dataset for tool-call style training."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_examples" / "sft_toy_10.jsonl"


def calculator_item(idx: int, expression: str, answer: str) -> dict:
    return {
        "id": f"sft_calc_{idx:03d}",
        "messages": [
            {"role": "system", "content": "You are a careful tool-use assistant."},
            {"role": "user", "content": f"Use a tool to calculate: {expression}"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {"tool": "calculator", "arguments": {"expression": expression}},
                    ensure_ascii=False,
                ),
            },
        ],
        "meta": {"task": "calculator", "answer": answer},
    }


def json_validator_item(idx: int, text: str) -> dict:
    return {
        "id": f"sft_json_{idx:03d}",
        "messages": [
            {"role": "system", "content": "You are a careful tool-use assistant."},
            {"role": "user", "content": f"Validate whether this text is JSON: {text}"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {"tool": "json_validator", "arguments": {"text": text}},
                    ensure_ascii=False,
                ),
            },
        ],
        "meta": {"task": "json_validator"},
    }


def main() -> None:
    items = [
        calculator_item(1, "17 * 23", "391"),
        calculator_item(2, "(48 + 12) / 5", "12"),
        calculator_item(3, "144 ** 0.5", "12"),
        calculator_item(4, "99 - 37", "62"),
        calculator_item(5, "8 * (7 + 6)", "104"),
        calculator_item(6, "1000 / 25", "40"),
        json_validator_item(1, '{"name": "Ada", "age": 20}'),
        json_validator_item(2, '{"tool": "calculator", "arguments": {"expression": "2+2"}}'),
        json_validator_item(3, "{name: Ada, age: 20}"),
        json_validator_item(4, "[1, 2, 3]"),
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"wrote {len(items)} examples to {OUT}")


if __name__ == "__main__":
    main()
