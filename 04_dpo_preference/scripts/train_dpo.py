"""
train_dpo.py
使用 TRL DPOTrainer 对 Qwen3 模型进行 DPO 偏好对齐训练。

依赖：
  pip install trl transformers datasets peft accelerate bitsandbytes

运行方式（本地 smoke test，CPU，仅测试流程）：
  python scripts/train_dpo.py --smoke_test

运行方式（GPU 正式训练，Qwen3-0.6B）：
  python scripts/train_dpo.py \
      --model_name Qwen/Qwen3-0.6B \
      --output_dir results/dpo_qwen3_0_6b \
      --num_train_epochs 2

运行方式（GPU 正式训练，Qwen3-1.7B）：
  python scripts/train_dpo.py \
      --model_name Qwen/Qwen3-1.7B \
      --output_dir results/dpo_qwen3_1_7b \
      --num_train_epochs 3 \
      --bf16

注意：
  - 需要 GPU 环境（4090 24GB 推荐）
  - smoke_test 模式在本地 CPU 验证流程，不保证效果
  - 训练完毕后 adapter 保存在 output_dir
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DPO training with TRL DPOTrainer.")

    # 模型
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B",
                        help="HuggingFace model name or path to SFT-merged model.")

    # 数据
    parser.add_argument("--train_file", type=str,
                        default=str(Path(__file__).parent.parent / "data/preference_toy_train.jsonl"))
    parser.add_argument("--eval_file",  type=str,
                        default=str(Path(__file__).parent.parent / "data/preference_toy_val.jsonl"))

    # 输出
    parser.add_argument("--output_dir", type=str,
                        default=str(Path(__file__).parent.parent / "results/dpo_output"))

    # DPO 超参
    parser.add_argument("--beta",            type=float, default=0.1)
    parser.add_argument("--loss_type",       type=str,   default="sigmoid")
    parser.add_argument("--max_length",      type=int,   default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)

    # 训练超参
    parser.add_argument("--learning_rate",   type=float, default=5e-7)
    parser.add_argument("--num_train_epochs", type=int,  default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size",  type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio",    type=float, default=0.1)

    # 精度
    parser.add_argument("--bf16",  action="store_true", help="Use bfloat16 precision.")
    parser.add_argument("--fp16",  action="store_true", help="Use float16 precision.")

    # 量化
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization.")

    # 显存优化
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # 日志与保存
    parser.add_argument("--logging_steps",  type=int, default=10)
    parser.add_argument("--save_steps",     type=int, default=200)
    parser.add_argument("--eval_steps",     type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)

    # Smoke test 模式（本地 CPU，仅验证流程）
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run a quick smoke test on CPU with tiny data.")

    return parser.parse_args()


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 导入（放在函数内，让 --help 不依赖重型库）
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOTrainer, DPOConfig
    except ImportError as e:
        print(f"[ERROR] 缺少依赖：{e}")
        print("请运行：pip install trl transformers datasets peft accelerate bitsandbytes")
        sys.exit(1)

    # ── Smoke test 覆盖参数 ──
    if args.smoke_test:
        print("=" * 60)
        print("SMOKE TEST MODE: 仅验证流程，使用 CPU + 极小配置")
        print("=" * 60)
        args.model_name = "Qwen/Qwen3-0.6B"
        args.num_train_epochs = 1
        args.per_device_train_batch_size = 1
        args.gradient_accumulation_steps = 1
        args.max_length = 256
        args.max_prompt_length = 128
        args.logging_steps = 1
        args.save_steps = 9999
        args.eval_steps = 9999
        args.load_in_4bit = False
        args.bf16 = False
        args.fp16 = False

    print(f"\n[Config]")
    print(f"  model:          {args.model_name}")
    print(f"  beta:           {args.beta}")
    print(f"  learning_rate:  {args.learning_rate}")
    print(f"  num_epochs:     {args.num_train_epochs}")
    print(f"  batch_size:     {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} = "
          f"{args.per_device_train_batch_size * args.gradient_accumulation_steps} (effective)")
    print(f"  max_length:     {args.max_length}")
    print(f"  output_dir:     {args.output_dir}")

    # ── 加载数据 ──
    def load_jsonl(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    print(f"\n[Data] Loading train: {args.train_file}")
    train_data = load_jsonl(args.train_file)
    print(f"[Data] Loading eval:  {args.eval_file}")
    eval_data  = load_jsonl(args.eval_file)

    if args.smoke_test:
        # smoke test 只用少量数据
        train_data = train_data[:8]
        eval_data  = eval_data[:4]

    print(f"[Data] train={len(train_data)}, eval={len(eval_data)}")

    train_dataset = Dataset.from_list(train_data)
    eval_dataset  = Dataset.from_list(eval_data)

    # ── 加载 Tokenizer ──
    print(f"\n[Model] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 加载模型 ──
    print(f"[Model] Loading model: {args.model_name}")

    if args.load_in_4bit:
        print("[Model] Using 4-bit quantization (bitsandbytes)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
    else:
        dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            print("[Model] Running on CPU (smoke test / no GPU)")

    # ── DPO 配置 ──
    # 注意：DPOTrainer 默认会自动从 model 克隆 ref_model，
    # 如果你想显式传入已保存的 SFT ref_model，可以在下面单独加载。
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,

        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.0,

        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,

        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        report_to="none",  # 不上传到 wandb。如有需要改为 "wandb"
    )

    # ── 初始化 DPOTrainer ──
    print("\n[Trainer] Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # None = 自动从 model 克隆 reference model
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # ── 训练 ──
    print("\n[Train] Starting DPO training...")
    trainer.train()

    # ── 保存 ──
    print(f"\n[Save] Saving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n✓ DPO training complete!")
    print(f"  Output saved to: {args.output_dir}")
    print("\n[Next Steps]")
    print("  1. 用 compare_sft_dpo.py 对比 SFT 和 DPO 输出")
    print("  2. 记录结果到 results/dpo_metrics.md")
    print("  3. 更新 reports/experiment_log.md 和 reports/eval_results.md")


if __name__ == "__main__":
    main()
