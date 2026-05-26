"""Validate JSONL chat data for the toy SFT pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ALLOWED_ROLES = {"system", "user", "assistant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    return parser.parse_args()


def validate_item(item: dict, line_no: int) -> list[str]:
    errors: list[str] = []

    if not isinstance(item.get("id"), str) or not item["id"].strip():
        errors.append(f"line {line_no}: missing non-empty string id")

    messages = item.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append(f"line {line_no}: messages must be a non-empty list")
        return errors

    roles = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"line {line_no}: message {i} must be an object")
            continue
        role = message.get("role")
        content = message.get("content")
        roles.append(role)
        if role not in ALLOWED_ROLES:
            errors.append(f"line {line_no}: message {i} has invalid role {role!r}")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"line {line_no}: message {i} has empty content")

        if role == "assistant" and isinstance(content, str) and content.strip().startswith(("{", "[")):
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: assistant JSON is invalid: {exc}")

    if "user" not in roles:
        errors.append(f"line {line_no}: missing user message")
    if "assistant" not in roles:
        errors.append(f"line {line_no}: missing assistant message")

    return errors


def main() -> None:
    args = parse_args()
    total = 0
    all_errors: list[str] = []

    with args.path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                all_errors.append(f"line {line_no}: invalid JSONL: {exc}")
                continue
            if not isinstance(item, dict):
                all_errors.append(f"line {line_no}: top-level value must be an object")
                continue
            all_errors.extend(validate_item(item, line_no))

    if all_errors:
        print(f"checked {total} examples: FAILED")
        for error in all_errors:
            print("-", error)
        raise SystemExit(1)

    print(f"checked {total} examples: OK")


if __name__ == "__main__":
    main()
