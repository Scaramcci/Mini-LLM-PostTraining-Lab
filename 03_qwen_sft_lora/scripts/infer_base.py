"""
Stage 03: Base 模型推理脚本

用法：
    python scripts/infer_base.py --model Qwen/Qwen3-0.6B --prompts prompts.jsonl

功能：
    加载未经 SFT 的 base model，用固定 prompt 生成输出。
    用于和 SFT 后的模型做对比。

输出：
    结果保存到 results/base_outputs.md
"""

import argparse
import json

# TODO: 学习阶段请逐步实现
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch


# 固定测试 prompt（你也可以从文件加载）
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
    parser = argparse.ArgumentParser(description="Base model inference")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompts", type=str, default=None, help="Optional jsonl file with prompts")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output", type=str, default="results/base_outputs.md")
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        with open(args.prompts, "r") as f:
            prompts = [json.loads(line) for line in f if line.strip()]

    print(f"Model: {args.model}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print()

    # ============================================================
    # TODO: 实现以下步骤
    # ============================================================

    # Step 1: 加载 tokenizer 和 model
    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    # model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    # model.eval()

    # Step 2: 对每个 prompt 生成输出
    # for i, prompt in enumerate(prompts):
    #     input_text = tokenizer.apply_chat_template(
    #         prompt["messages"], tokenize=False, add_generation_prompt=True
    #     )
    #     inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    #     with torch.no_grad():
    #         outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    #     response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    #     print(f"--- Prompt {i+1} ---")
    #     print(f"Input: {prompt['messages'][-1]['content']}")
    #     print(f"Output: {response}")
    #     print()

    # Step 3: 保存到 markdown 文件
    # 参考 results/base_outputs.md 的格式

    print("TODO: 请按照学习计划逐步实现上面注释中的代码")


if __name__ == "__main__":
    main()
