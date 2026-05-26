"""
validate_preference_data.py
校验 preference 数据集格式是否符合 TRL DPOTrainer 的要求。

检查项：
  1. 文件可读，每行是合法 JSON
  2. 每条数据包含 prompt / chosen / rejected 三个字段
  3. prompt 是 list of messages（role 为 system/user）
  4. chosen 是 list of messages（role 为 assistant）
  5. rejected 是 list of messages（role 为 assistant）
  6. chosen 和 rejected 不完全一致
  7. 统计 rejected 类型分布（可选，帮助了解数据多样性）

运行方式：
  python scripts/validate_preference_data.py
  python scripts/validate_preference_data.py --file data/preference_toy_val.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def validate_messages(field_name: str, messages, expected_roles: list) -> list[str]:
    """校验 messages 格式，返回错误列表。"""
    errors = []
    if not isinstance(messages, list):
        errors.append(f"  [{field_name}] 应为 list，实际为 {type(messages).__name__}")
        return errors
    if len(messages) == 0:
        errors.append(f"  [{field_name}] 不能为空 list")
        return errors
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"  [{field_name}][{i}] 应为 dict")
            continue
        if "role" not in msg:
            errors.append(f"  [{field_name}][{i}] 缺少 'role' 字段")
        elif msg["role"] not in (expected_roles + ["system", "user", "assistant"]):
            errors.append(f"  [{field_name}][{i}] role={msg['role']!r} 不在 {expected_roles}")
        if "content" not in msg:
            errors.append(f"  [{field_name}][{i}] 缺少 'content' 字段")
        elif not isinstance(msg["content"], str):
            errors.append(f"  [{field_name}][{i}] content 应为 str，实际为 {type(msg['content']).__name__}")
        elif len(msg["content"].strip()) == 0:
            errors.append(f"  [{field_name}][{i}] content 不能为空字符串")
    return errors


def classify_rejected(rejected_content: str) -> str:
    """简单判断 rejected 的错误类型（用于统计）。"""
    stripped = rejected_content.strip()
    if not stripped.startswith("{"):
        return "non_json"
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return "invalid_json"
    if "function" in obj or "params" in obj:
        return "wrong_key_name"
    if "arguments" in obj and obj.get("arguments") == {}:
        return "empty_arguments"
    if "tool" in obj:
        return "wrong_tool_name"
    if len(stripped) > 200:
        return "extra_text_pollution"
    return "other"


def validate_file(filepath: Path, verbose: bool = False):
    print(f"\n{'='*60}")
    print(f"Validating: {filepath}")
    print(f"{'='*60}")

    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        return

    errors_total = 0
    warnings_total = 0
    n_examples = 0
    rejected_type_counter = defaultdict(int)

    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            n_examples += 1

            # ── 1. JSON 解析 ──
            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Line {line_no}] JSON parse error: {e}")
                errors_total += 1
                continue

            line_errors = []

            # ── 2. 必需字段检查 ──
            for field in ("prompt", "chosen", "rejected"):
                if field not in example:
                    line_errors.append(f"  缺少必需字段: '{field}'")

            if line_errors:
                errors_total += len(line_errors)
                if verbose:
                    print(f"[Line {line_no}] 错误:")
                    for e in line_errors:
                        print(e)
                continue

            # ── 3. messages 格式校验 ──
            prompt_errors = validate_messages("prompt",   example["prompt"],   ["system", "user"])
            chosen_errors = validate_messages("chosen",   example["chosen"],   ["assistant"])
            rejected_errors = validate_messages("rejected", example["rejected"], ["assistant"])

            all_line_errors = prompt_errors + chosen_errors + rejected_errors

            # ── 4. chosen != rejected ──
            chosen_content   = example["chosen"][-1].get("content", "")
            rejected_content = example["rejected"][-1].get("content", "")
            if chosen_content == rejected_content:
                all_line_errors.append("  [warning] chosen 和 rejected 内容完全一致")
                warnings_total += 1

            # ── 5. 统计 rejected 类型 ──
            rtype = classify_rejected(rejected_content)
            rejected_type_counter[rtype] += 1

            if all_line_errors:
                errors_total += len(all_line_errors)
                if verbose:
                    print(f"[Line {line_no}] 问题:")
                    for e in all_line_errors:
                        print(e)

    # ── 汇总报告 ──
    print(f"\n总样例数：{n_examples}")
    print(f"发现错误：{errors_total}")
    print(f"发现警告：{warnings_total}")

    print("\nRejected 类型分布：")
    for rtype, count in sorted(rejected_type_counter.items(), key=lambda x: -x[1]):
        bar = "█" * (count * 20 // max(rejected_type_counter.values(), default=1))
        print(f"  {rtype:<25} {count:>4}  {bar}")

    if errors_total == 0:
        print("\n✓ 数据格式校验通过！")
    else:
        print(f"\n✗ 发现 {errors_total} 个错误，请检查后重新生成。")
        print("  提示：使用 --verbose 参数可查看每条错误的详情。")


def main():
    parser = argparse.ArgumentParser(description="Validate DPO preference dataset format.")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to the JSONL file to validate. "
             "If not specified, validates both train and val files."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-line error details."
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent / "data"

    if args.file:
        validate_file(Path(args.file), verbose=args.verbose)
    else:
        # 默认校验 train 和 val
        for filename in ("preference_toy_train.jsonl", "preference_toy_val.jsonl"):
            validate_file(base_dir / filename, verbose=args.verbose)


if __name__ == "__main__":
    main()
