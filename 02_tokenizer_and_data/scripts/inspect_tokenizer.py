"""Inspect how a tokenizer encodes text and renders chat messages."""

from __future__ import annotations

import argparse
import json


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_TEXTS = [
    "Hello, tokenizer!",
    "大模型到底是怎么处理中文的？",
    "Calculate 17 * 23 and return JSON.",
    '{"tool": "calculator", "arguments": {"expression": "17 * 23"}}',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text", action="append", help="Text to inspect. Can be passed multiple times.")
    parser.add_argument("--show-chat-template", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Please install dependencies first: pip install -r requirements.txt") from exc

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    texts = args.text or DEFAULT_TEXTS

    print(f"model: {args.model}")
    print(f"vocab_size: {getattr(tokenizer, 'vocab_size', 'unknown')}")
    print(f"bos_token: {tokenizer.bos_token!r} id={tokenizer.bos_token_id}")
    print(f"eos_token: {tokenizer.eos_token!r} id={tokenizer.eos_token_id}")
    print(f"pad_token: {tokenizer.pad_token!r} id={tokenizer.pad_token_id}")
    print()

    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False)
        token_ids = encoded["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        decoded = tokenizer.decode(token_ids)

        print("=" * 80)
        print("text:", text)
        print("num_tokens:", len(token_ids))
        print("token_ids:", token_ids)
        print("tokens:", tokens)
        print("decoded:", decoded)

    if args.show_chat_template:
        messages = [
            {"role": "system", "content": "You are a careful tool-use assistant."},
            {"role": "user", "content": "Calculate 17 * 23."},
            {
                "role": "assistant",
                "content": json.dumps(
                    {"tool": "calculator", "arguments": {"expression": "17 * 23"}},
                    ensure_ascii=False,
                ),
            },
        ]
        print("=" * 80)
        print("messages:")
        print(json.dumps(messages, ensure_ascii=False, indent=2))
        print()
        print("rendered chat template:")
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        print(rendered)


if __name__ == "__main__":
    main()
