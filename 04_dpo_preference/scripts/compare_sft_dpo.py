"""
compare_sft_dpo.py
对比 SFT 模型和 DPO 模型在相同 prompt 上的输出差异。

用法：
  python scripts/compare_sft_dpo.py \
      --sft_model_path /path/to/sft_model_or_adapter \
      --dpo_model_path results/dpo_output \
      --base_model Qwen/Qwen3-1.7B

输出：
  results/sft_vs_dpo_outputs.md

功能：
  - 用固定 prompt 集合分别调用 SFT 模型和 DPO 模型
  - 对比输出格式正确率、tool_name 正确率
  - 输出 Markdown 对比表
"""

import argparse
import json
import sys
from pathlib import Path


# ─────────────────────────────────────────────
# 固定评估 prompt（用于复现对比）
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful assistant that outputs only valid JSON tool calls. "
    "Do not include any explanation or extra text. "
    "Always respond with a single JSON object in the format: "
    '{\"tool\": \"<tool_name>\", \"arguments\": {<key>: <value>}}'
)

EVAL_PROMPTS = [
    # (user_prompt, expected_tool, expected_arg_key)
    ("Calculate 17 * 23.",                          "calculator",   "expression"),
    ("What is the weather in Tokyo?",               "weather",      "city"),
    ("Translate 'Good morning' to French.",         "translate",    "text"),
    ("Search for the latest news on AI.",           "search",       "query"),
    ("Convert 100 kilometers to miles.",            "unit_convert", "value"),
    ("What is 144 / 12?",                           "calculator",   "expression"),
    ("Get the weather for London.",                 "weather",      "city"),
    ("Translate 'Thank you' to Japanese.",          "translate",    "text"),
    ("Find papers on large language models.",       "search",       "query"),
    ("Convert 70 kilograms to pounds.",             "unit_convert", "value"),
]


# ─────────────────────────────────────────────
# 输出质量评估函数
# ─────────────────────────────────────────────

def evaluate_output(output: str, expected_tool: str) -> dict:
    """评估一条模型输出的质量。"""
    result = {
        "is_valid_json":    False,
        "has_tool_key":     False,
        "tool_correct":     False,
        "has_arguments":    False,
    }
    try:
        obj = json.loads(output.strip())
        result["is_valid_json"] = True
        if "tool" in obj:
            result["has_tool_key"] = True
            result["tool_correct"] = (obj["tool"] == expected_tool)
        if "arguments" in obj and isinstance(obj["arguments"], dict):
            result["has_arguments"] = len(obj["arguments"]) > 0
    except (json.JSONDecodeError, Exception):
        pass
    return result


def score(eval_result: dict) -> float:
    """简单打分：合法 JSON 0.4 + tool key 存在 0.2 + tool 正确 0.3 + arguments 非空 0.1"""
    s = 0.0
    if eval_result["is_valid_json"]:    s += 0.4
    if eval_result["has_tool_key"]:     s += 0.2
    if eval_result["tool_correct"]:     s += 0.3
    if eval_result["has_arguments"]:    s += 0.1
    return round(s, 2)


# ─────────────────────────────────────────────
# 推理函数
# ─────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str, base_model: str = None):
    """加载模型（支持完整模型或 LoRA adapter）。"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(f"[ERROR] 缺少依赖：{e}")
        sys.exit(1)

    path = Path(model_path)

    # 判断是否是 PEFT adapter（含 adapter_config.json）
    is_adapter = (path / "adapter_config.json").exists()

    if is_adapter:
        if base_model is None:
            print("[ERROR] 检测到 LoRA adapter，请通过 --base_model 指定 base model 路径。")
            sys.exit(1)
        print(f"  [Adapter] base_model = {base_model}")
        print(f"  [Adapter] adapter    = {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        print(f"  [Full model] path = {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, user_prompt: str, max_new_tokens: int = 128) -> str:
    """生成一条回答。"""
    import torch

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy decoding，确保复现性
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 只取新生成的 token
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare SFT vs DPO model outputs.")
    parser.add_argument("--sft_model_path", type=str, required=True,
                        help="Path to SFT model or adapter directory.")
    parser.add_argument("--dpo_model_path", type=str, required=True,
                        help="Path to DPO-trained model or adapter directory.")
    parser.add_argument("--base_model",     type=str, default=None,
                        help="Base model name/path (required if SFT/DPO paths are adapters).")
    parser.add_argument("--output_file",    type=str,
                        default=str(Path(__file__).parent.parent / "results/sft_vs_dpo_outputs.md"))
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    # ── 加载 SFT 模型 ──
    print("\n[SFT] Loading model...")
    sft_model, sft_tokenizer = load_model_and_tokenizer(args.sft_model_path, args.base_model)

    # ── 加载 DPO 模型 ──
    print("\n[DPO] Loading model...")
    dpo_model, dpo_tokenizer = load_model_and_tokenizer(args.dpo_model_path, args.base_model)

    # ── 推理 & 评估 ──
    print("\n[Eval] Running inference on evaluation prompts...")
    rows = []
    sft_scores = []
    dpo_scores = []

    for i, (user_prompt, expected_tool, _) in enumerate(EVAL_PROMPTS):
        print(f"  [{i+1}/{len(EVAL_PROMPTS)}] {user_prompt[:50]}...")
        sft_out = generate(sft_model, sft_tokenizer, user_prompt, args.max_new_tokens)
        dpo_out = generate(dpo_model, dpo_tokenizer, user_prompt, args.max_new_tokens)

        sft_eval = evaluate_output(sft_out, expected_tool)
        dpo_eval = evaluate_output(dpo_out, expected_tool)
        sft_s = score(sft_eval)
        dpo_s = score(dpo_eval)
        sft_scores.append(sft_s)
        dpo_scores.append(dpo_s)

        rows.append({
            "prompt":       user_prompt,
            "expected_tool": expected_tool,
            "sft_output":   sft_out,
            "dpo_output":   dpo_out,
            "sft_eval":     sft_eval,
            "dpo_eval":     dpo_eval,
            "sft_score":    sft_s,
            "dpo_score":    dpo_s,
        })

    # ── 生成 Markdown 报告 ──
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sft_avg = round(sum(sft_scores) / len(sft_scores), 3)
    dpo_avg = round(sum(dpo_scores) / len(dpo_scores), 3)

    lines = [
        "# SFT vs DPO 输出对比",
        "",
        "## 评分说明",
        "",
        "| 评估项 | 分值 |",
        "|--------|------|",
        "| is_valid_json | 0.4 |",
        "| has_tool_key | 0.2 |",
        "| tool_correct | 0.3 |",
        "| has_arguments (非空) | 0.1 |",
        "",
        "## 汇总结果",
        "",
        f"| 模型 | 平均分 |",
        f"|------|--------|",
        f"| SFT  | {sft_avg} |",
        f"| DPO  | {dpo_avg} |",
        "",
        "**结论（填写你的观察）：**",
        "",
        "> TODO: DPO 是否提高了格式准确率？tool_correct 有没有变化？有没有意外变差的样例？",
        "",
        "---",
        "",
        "## 详细对比",
        "",
    ]

    for i, row in enumerate(rows):
        sft_e = row["sft_eval"]
        dpo_e = row["dpo_eval"]
        lines += [
            f"### {i+1}. {row['prompt']}",
            "",
            f"**Expected tool**: `{row['expected_tool']}`",
            "",
            "| | SFT | DPO |",
            "|-|-|-|",
            f"| is_valid_json | {'✓' if sft_e['is_valid_json'] else '✗'} | {'✓' if dpo_e['is_valid_json'] else '✗'} |",
            f"| has_tool_key  | {'✓' if sft_e['has_tool_key'] else '✗'} | {'✓' if dpo_e['has_tool_key'] else '✗'} |",
            f"| tool_correct  | {'✓' if sft_e['tool_correct'] else '✗'} | {'✓' if dpo_e['tool_correct'] else '✗'} |",
            f"| has_arguments | {'✓' if sft_e['has_arguments'] else '✗'} | {'✓' if dpo_e['has_arguments'] else '✗'} |",
            f"| **score**     | **{row['sft_score']}** | **{row['dpo_score']}** |",
            "",
            "**SFT output:**",
            "```",
            row["sft_output"],
            "```",
            "",
            "**DPO output:**",
            "```",
            row["dpo_output"],
            "```",
            "",
            "**Observation:** (填写你对这条样例的观察)",
            "",
            "---",
            "",
        ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✓ Comparison saved to: {output_path}")
    print(f"\n  SFT avg score: {sft_avg}")
    print(f"  DPO avg score: {dpo_avg}")
    diff = round(dpo_avg - sft_avg, 3)
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
    print(f"  Difference:    {diff:+.3f} {arrow}")


if __name__ == "__main__":
    main()
