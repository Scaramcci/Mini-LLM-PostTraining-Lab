# DPO Training Metrics

记录 Stage 04 DPO 训练过程中的指标，方便和 SFT 阶段对比。

---

## 实验汇总表

| 实验编号 | 模型 | beta | lr | 数据量 | 训练步数 | 初始 loss | 最终 loss | 备注 |
|----------|------|------|-----|--------|----------|-----------|-----------|------|
| dpo-001  |      |      |     |        |          |           |           | smoke test |
| dpo-002  |      |      |     |        |          |           |           | 正式训练 |

---

## 实验 dpo-001（Smoke Test）

### 目标
验证 DPOTrainer 流程可跑通，不追求效果。

### 配置

```text
模型：
设备：
beta：
学习率：
batch_size (effective)：
数据量：
max_length：
训练步数：
```

### 训练日志摘要

```text
Step 1:   train_loss =
Step 10:  train_loss =
Step 20:  train_loss =
...
Final:    train_loss =
```

### 问题与解决

```text
问题：
解决：
```

---

## 实验 dpo-002（Qwen3-1.7B 正式训练）

### 目标
用 1.7B 模型做正式 DPO 训练，记录训练过程和评测结果。

### 配置

```text
模型：Qwen/Qwen3-1.7B
SFT adapter / 起点：（填写 Stage 03 的输出路径）
设备：GPU（填写型号）
beta：0.1
学习率：5e-7
batch_size：2 x 8 = 16（effective）
数据量：
max_length：1024
max_prompt_length：512
训练 epoch：3
bf16：true
gradient_checkpointing：true
```

### GPU 使用记录

```text
开始时间：
结束时间：
实际 GPU 时间：
费用估计：
Provider：
```

### 训练日志摘要

```text
Step 10:   train_loss =
Step 50:   train_loss =
Step 100:  train_loss =   eval_loss =
Step 200:  train_loss =   eval_loss =
Final:     train_loss =   eval_loss =
```

### 训练曲线

（截图或描述）

---

## 评测结果（从 compare_sft_dpo.py 输出填写）

| 评估项 | SFT | DPO | 变化 |
|--------|-----|-----|------|
| 平均分 | | | |
| is_valid_json 通过率 | | | |
| tool_correct 通过率  | | | |
| has_arguments 通过率 | | | |

### 分析结论

```text
TODO: DPO 是否提高了偏好行为？哪类错误减少了？哪类没变化？
```

---

## DPO 关键超参影响记录

（如果你做了 beta 的消融，记录在这里）

| beta | 现象 |
|------|------|
| 0.01 | |
| 0.1  | （默认） |
| 0.5  | |

---

## 失败案例记录

（DPO 后反而变差的样例，帮助你理解 DPO 的局限）

### 失败案例 1

```text
Prompt：
SFT 输出：
DPO 输出：
分析：
```
