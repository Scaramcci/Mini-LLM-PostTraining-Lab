# Stage 04：DPO 偏好对齐

## 本阶段目标

理解偏好对齐（Preference Alignment）的核心思想：给模型提供 chosen / rejected 对比数据，让模型学会更偏向好的回答，远离差的回答。

本阶段在 Stage 03（SFT LoRA）的基础上推进，将 SFT 模型作为起点，用 DPO 进一步对齐偏好。

## 核心概念

学完本阶段你需要能解释：

- **preference dataset** — 包含 prompt、chosen、rejected 三元组的偏好数据集
- **chosen / rejected** — 同一 prompt 的好回答 vs 差回答
- **policy model** — 正在训练的目标模型
- **reference model** — 用于计算 KL 散度的固定参考模型（通常是 SFT 模型）
- **DPO loss** — 不需要显式 reward 模型，直接从偏好数据优化策略
- **beta (β)** — 控制策略偏离 reference model 的程度
- **SFT → DPO** — DPO 通常在 SFT 之后做，SFT 模型作为 reference model

## 目录结构

```text
04_dpo_preference/
├── README.md                     ← 本文件
├── notes.md                      ← 概念学习笔记
├── configs/
│   └── qwen3_1_7b_dpo.yaml       ← DPO 训练配置
├── scripts/
│   ├── build_preference_data.py  ← 构造 toy preference 数据
│   ├── validate_preference_data.py ← 校验偏好数据格式
│   ├── train_dpo.py              ← DPO 训练主脚本
│   └── compare_sft_dpo.py        ← 对比 SFT 和 DPO 输出
└── results/
    ├── preference_examples.md    ← chosen/rejected 样例
    ├── dpo_metrics.md            ← 训练指标记录
    ├── sft_vs_dpo_outputs.md     ← SFT vs DPO 输出对比
    └── stage04_summary.md        ← 阶段总结（最后填写）
```

## 推荐资料

| 资料 | 链接 |
|------|------|
| TRL DPOTrainer 文档 | https://huggingface.co/docs/trl/main/dpo_trainer |
| DPO 原始论文 | https://arxiv.org/abs/2305.18290 |
| TRL preference 数据集 | https://huggingface.co/trl-lib |
| trl-lib/ultrafeedback_binarized | https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized |

## 本阶段进度

- [ ] 理解 DPO 的核心思想（读论文 + 读文档）
- [ ] 在 notes.md 中解释 DPO loss 和 beta
- [ ] 构造 toy preference 数据（chosen/rejected 对）
- [ ] 校验偏好数据格式
- [ ] 跑通 DPOTrainer（先用 0.6B）
- [ ] 正式用 1.7B 跑 DPO 训练
- [ ] 对比 SFT vs DPO 输出
- [ ] 记录失败样例和分析
- [ ] 更新 `reports/experiment_log.md`
- [ ] 更新 `reports/eval_results.md`
- [ ] 写完 `results/stage04_summary.md`

## GPU 建议

- 推荐机器：4090 24GB，按小时租
- 预计 GPU 时间：3–10 小时
- 先在本地完成数据构造、脚本调试，不需要 GPU
- 验证数据格式后再开机训练

## 完成标准

```text
✓ 有 preference 数据构造脚本
✓ 有 chosen/rejected 样例
✓ 有 SFT vs DPO 输出对比
✓ 能解释 DPO 和 SFT 的区别
✓ 知道 beta 大概控制什么
✓ 有 stage04_summary.md
```
