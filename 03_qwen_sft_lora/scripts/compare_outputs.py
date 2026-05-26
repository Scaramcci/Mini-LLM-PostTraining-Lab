"""
Stage 03: Base vs SFT 输出对比脚本

用法：
    python scripts/compare_outputs.py \
        --base_output results/base_outputs.md \
        --sft_output results/sft_outputs.md

功能：
    读取 base 和 SFT 模型的输出，并排对比。
    可选：自动计算格式准确率（如 JSON 格式是否正确、工具调用是否正确）。

输出：
    对比表保存到 results/sft_metrics.md
"""

import argparse
import json

# TODO: 学习阶段请逐步实现


def check_json_format(output: str) -> bool:
    """检查输出是否为合法 JSON"""
    try:
        json.loads(output)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def check_tool_call_format(output: str) -> bool:
    """
    检查输出是否包含工具调用格式
    这里只是示例，你需要根据自己定义的格式修改
    """
    # 示例：检查是否包含 tool_name 和 arguments
    try:
        parsed = json.loads(output)
        return "tool_name" in parsed and "arguments" in parsed
    except (json.JSONDecodeError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description="Compare base vs SFT outputs")
    parser.add_argument("--base_output", type=str, default="results/base_outputs.md")
    parser.add_argument("--sft_output", type=str, default="results/sft_outputs.md")
    parser.add_argument("--output", type=str, default="results/sft_metrics.md")
    args = parser.parse_args()

    # ============================================================
    # TODO: 实现以下步骤
    # ============================================================

    # Step 1: 读取两个输出文件
    # ...

    # Step 2: 逐条对比
    # 对比维度：
    #   - 是否回答了问题
    #   - 格式是否正确（JSON / tool call）
    #   - 输出长度
    #   - 是否有重复内容
    # ...

    # Step 3: 计算格式准确率
    # format_accuracy = correct / total
    # ...

    # Step 4: 生成对比 markdown 表格
    # ...

    print("TODO: 请按照学习计划逐步实现上面注释中的代码")
    print()
    print("对比维度建议：")
    print("  1. 是否回答了问题")
    print("  2. 格式是否正确（JSON / tool call）")
    print("  3. 输出长度")
    print("  4. 是否有重复内容")
    print("  5. 整体格式准确率")


if __name__ == "__main__":
    main()
