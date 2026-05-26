"""
Stage 03: Qwen LoRA SFT 训练脚本

用法：
    python scripts/train_sft.py --config configs/qwen3_0_6b_lora_sft.yaml

这个脚本的功能：
    1. 加载 Qwen base model
    2. 加载 LoRA 配置
    3. 加载 SFT 数据（jsonl 格式，messages 字段）
    4. 用 TRL SFTTrainer 训练
    5. 保存 LoRA adapter

注意：
    - 第一轮先用 0.6B 验证流程
    - 确认能跑通后再用 1.7B 做正式实验
    - 只保存 LoRA adapter，不保存完整 base model
"""

import argparse
import json
import yaml
from pathlib import Path

# TODO: 学习阶段请根据 TRL 文档逐步填写以下内容
# 参考文档：
#   https://huggingface.co/docs/trl/sft_trainer
#   https://huggingface.co/docs/peft/developer_guides/lora
#   https://huggingface.co/docs/transformers/model_doc/qwen3


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_dataset_from_jsonl(path: str) -> list:
    """
    加载 jsonl 格式数据集

    每条数据应包含 messages 字段：
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} examples from {path}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Qwen LoRA SFT Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    print("=== Config ===")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # ============================================================
    # TODO: 以下部分需要你在学习过程中逐步实现
    # ============================================================

    # Step 1: 加载 tokenizer
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])

    # Step 2: 加载 base model
    # from transformers import AutoModelForCausalLM
    # model = AutoModelForCausalLM.from_pretrained(
    #     config["model_name_or_path"],
    #     torch_dtype=torch.bfloat16 if config.get("bf16") else torch.float16,
    # )

    # Step 3: 配置 LoRA
    # from peft import LoraConfig, get_peft_model
    # lora_config = LoraConfig(
    #     r=config["lora_r"],
    #     lora_alpha=config["lora_alpha"],
    #     lora_dropout=config["lora_dropout"],
    #     target_modules=config["lora_target_modules"],
    #     task_type="CAUSAL_LM",
    # )
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()

    # Step 4: 加载数据
    # dataset = load_dataset_from_jsonl(config["dataset_path"])

    # Step 5: 配置 SFTTrainer
    # from trl import SFTTrainer, SFTConfig
    # training_args = SFTConfig(
    #     output_dir=config["output_dir"],
    #     num_train_epochs=config["num_train_epochs"],
    #     per_device_train_batch_size=config["per_device_train_batch_size"],
    #     gradient_accumulation_steps=config["gradient_accumulation_steps"],
    #     learning_rate=config["learning_rate"],
    #     warmup_ratio=config["warmup_ratio"],
    #     lr_scheduler_type=config["lr_scheduler_type"],
    #     max_seq_length=config["max_seq_length"],
    #     logging_steps=config["logging_steps"],
    #     save_steps=config["save_steps"],
    #     save_total_limit=config["save_total_limit"],
    #     bf16=config.get("bf16", False),
    #     gradient_checkpointing=config.get("gradient_checkpointing", False),
    #     seed=config.get("seed", 42),
    #     report_to=config.get("report_to", "none"),
    # )
    #
    # trainer = SFTTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset,
    #     tokenizer=tokenizer,
    # )

    # Step 6: 训练
    # trainer.train()

    # Step 7: 保存 LoRA adapter
    # trainer.save_model(config["output_dir"])
    # print(f"LoRA adapter saved to {config['output_dir']}")

    print("TODO: 请按照学习计划逐步实现上面注释中的代码")
    print("参考文档：")
    print("  - TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer")
    print("  - PEFT LoRA: https://huggingface.co/docs/peft/developer_guides/lora")


if __name__ == "__main__":
    main()
