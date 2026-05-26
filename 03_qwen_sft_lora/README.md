# Stage 03: Qwen LoRA SFT

## 目标

用 Qwen 小模型 + LoRA 做监督微调 (SFT)，让 base model 学会按照 instruction 格式回答问题。

## 前置要求

- 完成 Stage 01（nanoGPT 预训练）和 Stage 02（Tokenizer 和数据工程）
- Stage 02 产出的 toy SFT 数据集（jsonl 格式）
- 理解 chat template、tokenizer、input_ids / attention_mask / labels
- GPU 环境可用（推荐 4090 24GB）

## 核心概念

| 概念 | 说明 |
| --- | --- |
| SFT | Supervised Fine-Tuning，用 instruction-response 对来微调模型 |
| LoRA | Low-Rank Adaptation，只训练少量参数，省显存 |
| QLoRA | 量化 + LoRA，进一步省显存 |
| adapter | LoRA 训练出的小参数文件，可以和 base model 合并 |
| SFTTrainer | TRL 提供的高层训练 API |

## 目录结构

```text
03_qwen_sft_lora/
├── README.md              # 本文件
├── notes.md               # 学习笔记
├── stage03_plan.md        # 详细学习计划
├── configs/
│   ├── qwen3_0_6b_lora_sft.yaml   # 0.6B 模型配置
│   └── qwen3_1_7b_lora_sft.yaml   # 1.7B 模型配置
├── scripts/
│   ├── train_sft.py       # SFT 训练脚本
│   ├── infer_base.py      # base 模型推理
│   ├── infer_sft.py       # SFT 模型推理
│   └── compare_outputs.py # 对比 base vs SFT 输出
└── results/
    ├── base_outputs.md    # base 模型输出记录
    ├── sft_outputs.md     # SFT 模型输出记录
    ├── sft_metrics.md     # 训练指标记录
    └── stage03_summary.md # 阶段总结
```

## 模型选择

| 轮次 | 模型 | 用途 |
| --- | --- | --- |
| 第一轮 | Qwen/Qwen3-0.6B | 跑通代码、验证流程 |
| 第二轮 | Qwen/Qwen3-1.7B | 正式实验、记录结果 |
| 进阶 | Qwen/Qwen3-4B | 预算充足时可选 |

## 完成标准

- [ ] 有 base vs SFT 对比输出
- [ ] 有 LoRA adapter 保存和加载说明
- [ ] 有训练 loss 曲线记录
- [ ] 有格式准确率或任务准确率
- [ ] 能解释 LoRA 为什么省显存
- [ ] 能解释 SFT 改变了模型什么行为
- [ ] stage03_summary.md 已完成
