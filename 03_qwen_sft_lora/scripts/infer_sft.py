"""
Stage 03: SFT 模型推理脚本

用法：
    python scripts/infer_sft.py --base_model Qwen/Qwen3-0.6B --adapter results/qwen3_0_6b_sft

功能：
    加载 base model + LoRA adapter，用固定 prompt 生成输出。
    用于和 base model 做对比，验证 SFT 效果。

输出：
    结果保存到 results/sft_outputs.md
"""

import argparse
import json

# TODO: 学习阶段请逐步实现
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
# import torch


# 和 infer_base.py 使用相同的测试 prompt
DEFAULT_PROMPTS = [
    {
        "messages": [
            {"role": "user", "content": "Calculate 17 * 23."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Write a Python function that checks if a number is prime."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that responds in JSON format."},
            {"role": "user", "content": "List 3 fruits with their colors."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Translate 'Hello, how are you?' to Chinese."}
        ]
    },
]


def main():
    parser = argparse.ArgumentParser(description="SFT model inference")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--prompts", type=str, default=None, help="Optional jsonl file with prompts")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output", type=str, default="results/sft_outputs.md")
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        with open(args.prompts, "r") as f:
            prompts = [json.loads(line) for line in f if line.strip()]

    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter}")
    print(f"Number of prompts: {len(prompts)}")
    print()

    # ============================================================
    # TODO: 实现以下步骤
    # ============================================================

    # Step 1: 加载 tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Step 2: 加载 base model
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     args.base_model, torch_dtype=torch.bfloat16
    # )

    # Step 3: 加载 LoRA adapter
    # model = PeftModel.from_pretrained(base_model, args.adapter)
    # model.eval()

    # Step 4: 对每个 prompt 生成输出（同 infer_base.py）
    # ...

    # Step 5: 保存到 markdown 文件
    # ...

    print("TODO: 请按照学习计划逐步实现上面注释中的代码")


if __name__ == "__main__":
    main()
