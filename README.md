# Mini-LLM Post-Training Lab

这是我的第一个长期大模型学习项目。目标不是一口气训练大模型，而是用大约一个月时间，从最小 GPT 原理开始，完整走一遍小型 LLM post-training pipeline。

## Goal

用“小而全”的方式完成：

1. 从零训练一个 mini GPT，理解 Transformer 和 causal LM。
2. 学 tokenizer、chat template 和数据工程。
3. 用 Qwen 小模型做 LoRA SFT。
4. 用 DPO 做偏好对齐。
5. 用 GRPO / verifiable reward 做小型后训练实验。
6. 用 vLLM 起 OpenAI-compatible 推理服务。
7. 做一个 tool-use agent。
8. 整理评测结果和最终报告。

## Current Focus

当前推进到 Stage 02：

```text
Stage 00: 项目初始化、日志、环境、GitHub 管理，已完成主体文件
Stage 01: nanoGPT 预训练和 Transformer 基础，已完成
Stage 02: Tokenizer 和数据工程，当前阶段
```

后续 Stage 03-08 的目录和文件会边学习边创建。

## Project Structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── docs/
│   ├── learning_log.md
│   ├── concept_checklist.md
│   ├── environment_setup.md
│   ├── gpu_rental_notes.md
│   └── reading_notes/
│       ├── transformer.md
│       └── tokenizer.md
├── data/
│   ├── README.md
│   └── samples/
├── models/
│   └── README.md
├── reports/
│   ├── experiment_log.md
│   ├── gpu_usage_log.md
│   ├── eval_results.md
│   └── final_report.md
├── scripts/
│   ├── check_env.py
│   ├── download_sample_data.py
│   └── clean_outputs.py
├── 01_nanogpt_pretrain/
│   ├── README.md
│   ├── notes.md
│   ├── stage01_plan.md
│   ├── configs/
│   ├── scripts/
│   └── results/
└── 02_tokenizer_and_data/
    ├── README.md
    ├── notes.md
    ├── stage02_plan.md
    ├── scripts/
    ├── data_examples/
    └── results/
```

## Progress

- [ ] Stage 00: project setup
- [x] Stage 01: nanoGPT pretraining
- [ ] Stage 02: tokenizer and data
- [ ] Stage 03: Qwen SFT LoRA
- [ ] Stage 04: DPO preference alignment
- [ ] Stage 05: GRPO verifiable reward
- [ ] Stage 06: vLLM serving
- [ ] Stage 07: Agent tool use
- [ ] Stage 08: eval and final report

## Learning Rule

每次学习或实验都要留下记录：

- 学习记录写到 `docs/learning_log.md`
- 环境配置写到 `docs/environment_setup.md`
- 实验记录写到 `reports/experiment_log.md`
- GPU 使用写到 `reports/gpu_usage_log.md`
- 阶段总结写到对应 stage 的 `results/stageXX_summary.md`

第一阶段先不要急着租 GPU。本地能跑通 tiny Shakespeare 和 nanoGPT 的最小流程后，再考虑按小时租 4090。
