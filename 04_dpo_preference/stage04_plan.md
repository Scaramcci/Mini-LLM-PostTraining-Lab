# Stage 04 学习与实验计划：DPO 偏好对齐

这一阶段的核心目标不是"把模型训练得多好"，而是第一次真正理解：

```text
偏好对齐是什么，为什么 SFT 之后还要做 DPO
policy model 和 reference model 的关系
DPO loss 的直觉含义
chosen/rejected 数据该怎么设计
TRL DPOTrainer 怎么用
DPO 真的改变了模型什么行为
```

建议用 **5–8 天** 完成。不要急着往 GRPO 走，Stage 04 的概念是后面所有强化学习内容的基础。

---

## 0. 本阶段最终交付

完成 Stage 04 时，你应该交付这些内容：

```text
04_dpo_preference/
├── notes.md                             ← 概念笔记，自己写，不能只复制粘贴
├── data/
│   ├── preference_toy_train.jsonl       ← 偏好训练集
│   └── preference_toy_val.jsonl         ← 偏好验证集
└── results/
    ├── preference_examples.md           ← chosen/rejected 样例说明
    ├── dpo_metrics.md                   ← 训练指标记录
    ├── sft_vs_dpo_outputs.md            ← SFT 和 DPO 输出对比
    └── stage04_summary.md               ← 阶段总结（最重要）

../reports/experiment_log.md             ← 更新实验记录
../reports/eval_results.md              ← 更新评测结果表
../docs/learning_log.md                 ← 每天更新
../docs/concept_checklist.md            ← 勾选掌握的概念
```

最终你需要能用自己的话讲清楚：

```text
1. 偏好对齐是什么？为什么 SFT 训练不能解决偏好问题？
2. chosen 和 rejected 分别代表什么？如何设计？
3. policy model 是什么？reference model 是什么？为什么需要 reference model？
4. DPO loss 的直觉是什么？（不需要背公式，但要能解释思路）
5. beta 控制什么？增大 vs 减小 beta 各有什么效果？
6. DPO 和 RLHF 的关系是什么？DPO 省去了什么？
7. 你的 DPO 实验结果说明了什么？SFT 和 DPO 的输出有什么具体区别？
8. 这个实验有哪些局限？
```

---

## 1. 学习顺序总览

推荐顺序：

```text
Day 1: 理解偏好对齐的动机，不看代码
Day 2: 读 DPO 论文摘要 + TRL DPOTrainer 文档
Day 3: 构造 toy preference 数据并验证格式
Day 4: smoke test — 在 CPU 上跑通 DPOTrainer 流程
Day 5: 租 GPU，用 Qwen3-0.6B 跑第一个 DPO 实验
Day 6: 用 Qwen3-1.7B 跑正式 DPO 实验
Day 7: 对比 SFT vs DPO 输出，分析差异
Day 8: 写 Stage 04 总结
```

时间宽松的话，每个 Day 可以拆成两天。

---

## 2. Day 1：理解偏好对齐的动机

### 2.1 今天要理解什么

先不看代码，先回答一个问题：

```text
SFT 已经可以让模型遵循指令了，为什么还需要偏好对齐？
```

你需要理解：

```text
SFT 只告诉模型"按照这个格式输出"
SFT 没有告诉模型"在同类型的输出里，哪种更好"

比如：
- 两个回答都格式正确，但一个更简洁、一个废话很多
- SFT 对这两种都一样高兴
- 偏好对齐告诉模型：简洁的那个更好
```

偏好对齐 pipeline 的经典形式（RLHF）：

```text
1. 训练 Reward Model（RM）
   → 学习"什么是好回答"
2. 用 PPO 优化 LLM
   → 最大化 RM 的打分，同时不偏离 SFT 太远
```

DPO 的改进：

```text
跳过单独的 RM 训练
直接从 chosen/rejected 对优化 LLM
数学上等价于在隐式 reward 下做 RLHF
更简单、更稳定
```

### 2.2 今天要看的资料

按顺序：

```text
1. TRL DPOTrainer 文档（先看概述部分，不用看所有参数）
   https://huggingface.co/docs/trl/main/dpo_trainer

2. DPO 论文摘要（只看 Abstract + Introduction，不用读全文）
   https://arxiv.org/abs/2305.18290

3. 可选：Anthropic 的 Constitutional AI 概念介绍
   （帮助理解为什么"偏好"很重要）
```

### 2.3 今天要记录什么

写到 `notes.md` 的"为什么需要偏好对齐"部分：

```markdown
## 为什么需要偏好对齐？

### SFT 的局限

### 偏好对齐的目标

### RLHF vs DPO

### 我的问题
```

写到 `../docs/learning_log.md`：

```markdown
## 2026-xx-xx

### Today
- 理解了偏好对齐的动机和 DPO 与 RLHF 的关系。

### Concepts
- preference alignment
- chosen / rejected
- RLHF
- DPO

### Questions
-

### Next
- 读 DPO 论文摘要和 TRL 文档，理解 policy/reference model。
```

### 2.4 完成标准

你能不看资料回答：

```text
为什么 SFT 之后还要做偏好对齐？
chosen 和 rejected 分别是什么？
DPO 省去了 RLHF 的哪一步？
```

---

## 3. Day 2：理解 DPO 机制和关键参数

### 3.1 今天要理解什么

重点理解以下概念（不需要背数学公式，需要理解直觉）：

```text
policy model（π_θ）
reference model（π_ref）
DPO loss 的直觉
beta（β）的作用
```

DPO loss 简化理解：

```text
目标：
  让 policy model 对 chosen 的"相对 reference model 的偏好"
  高于对 rejected 的"相对 reference model 的偏好"

关键约束：
  policy 不能离 reference model 太远（KL 散度约束）
  β 控制这个约束的强度

β 大 → KL 惩罚强 → 不敢偏离 reference → 训练保守
β 小 → KL 惩罚弱 → 可以自由变化 → 可能过拟合
```

为什么需要 reference model：

```text
如果没有 reference model：
  模型可能极度放大 chosen 的概率
  同时极度压低 rejected 的概率
  但可能生成完全不相关的文本（reward hacking）

reference model 作为锚点：
  保证 policy 不会偏离"能说人话"的基础太远
```

### 3.2 今天要看的资料

```text
1. TRL DPOConfig 文档（看 beta、loss_type 等参数）
   https://huggingface.co/docs/trl/main/dpo_trainer#trl.DPOConfig

2. TRL 提供的 preference 数据集格式说明
   https://huggingface.co/docs/trl/main/dataset_formats#preference

3. 可选：DPO 论文 Section 3（直接优化部分，读不懂公式没关系，看图）
```

### 3.3 今天要记录什么

写到 `notes.md` 的"DPO Loss"和"policy/reference model"部分。

更新 `../docs/concept_checklist.md`，勾选能解释的概念。

### 3.4 完成标准

你能回答：

```text
policy model 和 reference model 分别是什么？
为什么 DPO 需要 reference model，不能直接做 MLE？
β 增大 vs 减小各有什么影响？
TRL DPOTrainer 的数据格式要求是什么？
```

---

## 4. Day 3：构造 Toy Preference 数据

### 4.1 今天目标

从 Stage 03 的 JSON tool-call 任务延续，构造 chosen/rejected 数据。

不要用很大的公开偏好数据集。先自己手动设计，理解数据质量对 DPO 的影响。

### 4.2 数据设计原则

chosen 应该是：

```text
✓ 格式正确（合法 JSON）
✓ tool name 正确
✓ arguments 完整且正确
✓ 简洁，没有废话
```

rejected 应该覆盖常见错误类型：

```text
类型 1：直接给文字答案，不调用工具
类型 2：JSON 格式存在但 key 名称错（function/params 而不是 tool/arguments）
类型 3：tool name 错误（选错了工具）
类型 4：废话前缀 + 正确 JSON（输出不规范）
类型 5：arguments 为空 {}（参数缺失）
类型 6：JSON 截断/无法解析
```

### 4.3 数据格式（TRL messages 格式）

```json
{
  "prompt": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user",   "content": "Calculate 17 * 23."}
  ],
  "chosen": [
    {"role": "assistant", "content": "{\"tool\": \"calculator\", \"arguments\": {\"expression\": \"17 * 23\"}}"}
  ],
  "rejected": [
    {"role": "assistant", "content": "The answer is probably 400."}
  ]
}
```

### 4.4 运行脚本

```bash
# 进入项目根目录（Mini-LLM-PostTraining-Lab/）
python 04_dpo_preference/scripts/build_preference_data.py
```

输出：

```text
data/preference_toy_train.jsonl
data/preference_toy_val.jsonl
```

### 4.5 校验数据

```bash
python 04_dpo_preference/scripts/validate_preference_data.py
python 04_dpo_preference/scripts/validate_preference_data.py --verbose
```

确认：

```text
✓ 格式校验通过（0 errors）
✓ rejected 类型分布合理（不能全是同一种错误类型）
✓ 没有 chosen == rejected 的样例
```

### 4.6 今天要记录什么

更新 `results/preference_examples.md`：

```text
填写数据统计
贴 2-3 条有代表性的样例
记录每种 rejected 类型的设计意图
```

### 4.7 完成标准

```text
data/preference_toy_train.jsonl 可以读取，格式正确
validate_preference_data.py 显示 0 errors
你能解释每种 rejected 类型的设计意图
```

---

## 5. Day 4：本地 Smoke Test（无需 GPU）

### 5.1 今天目标

在本地 CPU 上跑通完整的 DPOTrainer 流程，不追求效果，只验证代码路径。

### 5.2 运行 smoke test

```bash
# 在项目根目录
python 04_dpo_preference/scripts/train_dpo.py --smoke_test
```

smoke test 会自动使用：

```text
模型：Qwen3-0.6B
设备：CPU
批量：极小（batch=1，no gradient accumulation）
步数：极少（几步就停止）
目标：验证数据加载、模型加载、DPO loss 计算、保存都没有报错
```

### 5.3 常见错误和解决思路

```text
ImportError: trl / transformers / peft 未安装
  → pip install trl transformers peft accelerate bitsandbytes

RuntimeError: out of memory
  → smoke test 模式不应该 OOM，如果出现，减小 max_length

ValueError: dataset format
  → 检查 jsonl 文件格式，运行 validate_preference_data.py

AttributeError: DPOConfig
  → trl 版本过低，pip install --upgrade trl
```

### 5.4 今天要记录什么

写到 `reports/experiment_log.md`：

```markdown
## exp-dpo-001: DPO smoke test（CPU）

### Date
### Device
### Command
### Config
- model:
- beta:
- batch_size:
- max_length:

### Result
- smoke test 是否通过：
- 报错（如有）：
- 解决方法：
```

### 5.5 完成标准

```text
train_dpo.py --smoke_test 能跑完，没有崩溃
experiment_log.md 有一条 smoke test 记录
你大概知道 DPOTrainer 在做什么（加载数据、init ref_model、计算 DPO loss）
```

---

## 6. Day 5：租 GPU，用 Qwen3-0.6B 跑第一个 DPO 实验

### 6.1 租 GPU 前检查清单

先在本地写好：

```text
□ 今天要跑哪个脚本？
□ 模型：Qwen/Qwen3-0.6B
□ 数据路径是否正确？
□ 输出目录是否有足够磁盘空间？
□ GPU 小时预计：1-3 小时
□ 跑完要保存哪些结果？
```

写到 `docs/gpu_rental_notes.md` 和 `reports/gpu_usage_log.md`。

### 6.2 在 GPU 服务器上的操作顺序

```bash
# 1. 克隆或上传项目
git clone <your_repo_url>
cd Mini-LLM-PostTraining-Lab

# 2. 安装依赖
pip install -r requirements.txt
pip install trl accelerate peft bitsandbytes

# 3. 下载模型（自动缓存）
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')"

# 4. 构造数据（如果没有预先生成）
python 04_dpo_preference/scripts/build_preference_data.py

# 5. 运行 DPO 训练
python 04_dpo_preference/scripts/train_dpo.py \
    --model_name Qwen/Qwen3-0.6B \
    --output_dir 04_dpo_preference/results/dpo_qwen3_0_6b \
    --num_train_epochs 2 \
    --bf16
```

### 6.3 训练时观察什么

```text
train_loss 是否在下降？
eval_loss 是否跟随下降（没有严重过拟合）？
有没有 OOM 或 CUDA error？
chosen_rewards 和 rejected_rewards 差距是否在扩大？（TRL 会输出这个）
```

### 6.4 完成标准

```text
0.6B 模型 DPO 训练跑完
adapter 保存到 results/dpo_qwen3_0_6b
experiment_log 有记录
gpu_usage_log 有记录
```

---

## 7. Day 6：用 Qwen3-1.7B 跑正式 DPO 实验

### 7.1 和 0.6B 的区别

```text
模型更大 → 效果通常更能看出差异
显存需求更大 → 注意 batch size 和 gradient checkpointing
训练更慢 → 提前估算时间，合理设置 save_steps
```

### 7.2 运行命令

```bash
python 04_dpo_preference/scripts/train_dpo.py \
    --model_name Qwen/Qwen3-1.7B \
    --output_dir 04_dpo_preference/results/dpo_qwen3_1_7b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --beta 0.1 \
    --learning_rate 5e-7 \
    --bf16 \
    --gradient_checkpointing
```

### 7.3 如果 OOM

```text
减小 max_length（512 → 256）
减小 per_device_train_batch_size（2 → 1）
增大 gradient_accumulation_steps（保持等效 batch）
开启 load_in_4bit 量化（加 --load_in_4bit 参数）
```

### 7.4 完成标准

```text
1.7B DPO 训练跑完
adapter 保存
experiment_log 有记录
dpo_metrics.md 有训练日志摘要
```

---

## 8. Day 7：对比 SFT vs DPO，分析差异

### 8.1 今天目标

用固定 prompt 集合，量化对比 SFT 和 DPO 的输出差异。

### 8.2 运行对比脚本

```bash
python 04_dpo_preference/scripts/compare_sft_dpo.py \
    --sft_model_path 03_qwen_sft_lora/results/sft_qwen3_1_7b \
    --dpo_model_path 04_dpo_preference/results/dpo_qwen3_1_7b \
    --base_model Qwen/Qwen3-1.7B
```

输出会保存到 `04_dpo_preference/results/sft_vs_dpo_outputs.md`。

### 8.3 人工审查

自动脚本只是辅助，你还需要：

```text
1. 每条 prompt，对比 SFT 和 DPO 的原始输出文本
2. 找出 DPO 改善的案例（格式更好、内容更对）
3. 找出 DPO 退化的案例（本来好的变差了）
4. 分析退化原因（数据质量？beta 太大/太小？）
```

### 8.4 更新评测结果

写到 `../reports/eval_results.md`，新增 DPO 一行。

### 8.5 完成标准

```text
sft_vs_dpo_outputs.md 有定量对比表
有人工补充的典型案例分析
eval_results.md 有 DPO 那一行
```

---

## 9. Day 8：写阶段总结

### 9.1 今天目标

把 Stage 04 收束成一个可以复盘的阶段成果。

写：

```text
04_dpo_preference/results/stage04_summary.md
```

### 9.2 总结应该包含什么

参考 `stage04_summary.md` 模板，填写：

```text
Goal
What I Did
Key Commands
Key Results（数据、训练、评测）
What I Learned（概念、代码、实验）
Failure Cases
Remaining Questions
Next Stage（GRPO 预告）
```

### 9.3 完成标准

```text
notes.md 有概念笔记（不能是 copy-paste，要用自己的话）
preference_examples.md 有样例说明
dpo_metrics.md 有训练日志
sft_vs_dpo_outputs.md 有对比结果
stage04_summary.md 能回答验收问题
eval_results.md 有 DPO 那一行
experiment_log.md 有实验记录
concept_checklist.md 勾选 Stage 04 相关概念
```

---

## 10. 参考资料清单

| 资料 | 链接 | 阅读优先级 |
|------|------|-----------|
| TRL DPOTrainer 文档 | https://huggingface.co/docs/trl/main/dpo_trainer | ★★★（必读） |
| TRL DPOConfig 参数 | https://huggingface.co/docs/trl/main/dpo_trainer#trl.DPOConfig | ★★★（必读） |
| TRL preference 数据格式 | https://huggingface.co/docs/trl/main/dataset_formats#preference | ★★★（必读） |
| DPO 原始论文 | https://arxiv.org/abs/2305.18290 | ★★（读 Abstract + Intro） |
| trl-lib/ultrafeedback_binarized | https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized | ★（可参考数据格式） |

---

## 11. GPU 费用控制

```text
Day 1-3：本地学习 + 数据准备，不需要 GPU
Day 4：smoke test 在本地 CPU，不需要 GPU
Day 5：Qwen3-0.6B DPO，预计 1-3 GPU 小时
Day 6：Qwen3-1.7B DPO，预计 3-8 GPU 小时
Day 7：compare_sft_dpo.py，用已保存模型，1-2 GPU 小时（或 CPU）
Day 8：写总结，不需要 GPU

合计预计：5-13 GPU 小时（4090 24GB）
```

每次开机前，先在本地写好命令，跑完立刻保存结果并关机。

---

## 12. 验收问题（完成前自查）

完成本阶段后，你应该能不看笔记回答：

1. 偏好对齐是什么？为什么 SFT 训练后还需要做？
2. DPO 的训练数据包含哪三个字段？每个字段是什么？
3. policy model 和 reference model 分别是什么？
4. 为什么 DPO 需要 reference model？不用它会有什么问题？
5. beta 控制什么？你在实验里感受到了什么？
6. TRL DPOTrainer 需要什么格式的数据？和 SFTTrainer 有什么不同？
7. 你的 DPO 实验结果说明了什么？SFT 和 DPO 输出有什么具体区别？
8. DPO 之后，有没有任何样例反而变差了？你怎么解释？
9. 这个实验的局限是什么？
10. Stage 05（GRPO）和 DPO 的核心区别是什么？
