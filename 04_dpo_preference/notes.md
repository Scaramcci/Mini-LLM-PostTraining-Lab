# Stage 04 学习笔记：DPO 偏好对齐

> 这里记录你在 Stage 04 阶段的概念理解、代码阅读和问题记录。
> 每天学习后更新，不要等到最后一次性补写。

---

## 概念理解

### 为什么需要偏好对齐？

SFT 之后，模型能按照指令格式输出，但不一定会输出"好的"回答。
比如：格式正确但内容冗余、有帮助但不够简洁、给出错误的 tool arguments。

偏好对齐（Preference Alignment）的目标：
- 给模型两种回答（好的 vs 差的）
- 让模型学会更偏向好的输出

---

### RLHF 与 DPO 的关系

```text
RLHF 流程：
1. 训练一个 Reward Model（RM）— 需要大量标注
2. 用 PPO 用 RM 的分数来更新 LLM — 实现复杂，不稳定

DPO（Direct Preference Optimization）：
- 跳过 Reward Model，直接从偏好数据对 LLM 优化
- 数学上等价于在 implicit reward 下做 RLHF
- 更简单、更稳定
```

**填写你的理解：**

```text
TODO: 用自己的话解释 DPO 为什么不需要单独的 Reward Model？
```

---

### DPO Loss 公式

DPO 的训练目标：让策略模型对 chosen 的对数概率 比 rejected 高，同时不离 reference model 太远。

Loss 形式（简化理解）：

```text
L_DPO = -log σ( β * (log π_θ(y_w|x) - log π_ref(y_w|x))
                  - β * (log π_θ(y_l|x) - log π_ref(y_l|x)) )

其中：
- π_θ  = 正在训练的 policy model
- π_ref = 固定的 reference model（通常是 SFT model）
- y_w  = chosen（好的回答）
- y_l  = rejected（差的回答）
- β    = 控制 KL 惩罚强度的超参数
- σ    = sigmoid 函数
```

**填写你的理解：**

```text
TODO: β 增大时，训练有什么变化？β 减小时呢？
```

---

### policy model 和 reference model

```text
policy model（π_θ）：
- 正在训练、参数在更新的模型
- 从 SFT 模型初始化

reference model（π_ref）：
- 参数固定、不更新
- 通常是 SFT 训练完成后的模型
- 作为"基准"，防止 policy 偏离太远
```

**填写你的理解：**

```text
TODO: 为什么需要 reference model？不用它会有什么问题？
```

---

### beta（β）的作用

```text
β 控制对 reference model 的 KL 惩罚强度：

β 大 → KL 惩罚强 → policy 不敢偏离 reference 太远 → 保守
β 小 → KL 惩罚弱 → policy 可以自由变化 → 可能过拟合偏好数据

TRL 默认 beta=0.1，通常在 0.01 ~ 0.5 之间调整。
```

**填写你的理解：**

```text
TODO: 你在实验中用了什么 beta？效果如何？
```

---

### SFT → DPO 的关系

```text
标准流程：
1. 先做 SFT（Stage 03）
   - 得到一个会遵循指令格式的 base SFT 模型
   - 这个模型同时作为 DPO 的 reference model

2. 再做 DPO（Stage 04）
   - policy model 从 SFT 模型初始化
   - reference model 固定为 SFT 模型
   - 用 chosen/rejected 数据训练

注意：DPO 不是 SFT 的替代，是 SFT 之后的进一步对齐。
```

---

## 数据格式理解

### TRL DPOTrainer 期望的数据格式

标准格式（messages 格式，更推荐）：

```json
{
  "prompt": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Return a JSON tool call for calculating 17 * 23."}
  ],
  "chosen": [
    {"role": "assistant", "content": "{\"tool\": \"calculator\", \"arguments\": {\"expression\": \"17 * 23\"}}"}
  ],
  "rejected": [
    {"role": "assistant", "content": "The answer is probably 400."}
  ]
}
```

简单字符串格式（也可以，但不推荐混用）：

```json
{
  "prompt": "Return a JSON tool call for calculating 17 * 23.",
  "chosen": "{\"tool\": \"calculator\", \"arguments\": {\"expression\": \"17 * 23\"}}",
  "rejected": "The answer is probably 400."
}
```

**填写你的理解：**

```text
TODO: 你最终选择了哪种格式？为什么？
```

---

### chosen 和 rejected 的设计原则

本阶段的任务延续 Stage 03 的 JSON tool-call 任务：

chosen（好的回答）应该：
- 格式正确（合法 JSON）
- tool name 正确
- arguments 正确
- 简洁，不废话

rejected（差的回答）应该覆盖常见错误类型：
- 格式错误（不是 JSON 或无法解析）
- tool name 错误（比如选错工具）
- arguments 错误（参数值错误或缺失）
- 废话太多（先解释再输出，而不是直接输出 JSON）
- 直接给出答案而不是调用工具

---

## TRL DPOTrainer 代码阅读笔记

**填写你阅读 TRL DPOTrainer 代码后的理解：**

### DPOTrainer 初始化

```text
TODO: model 参数传的是什么？
TODO: ref_model 参数传的是什么？（也可以不传，让 TRL 自动从 model 创建）
TODO: train_dataset 的格式要求？
```

### DPOConfig 关键参数

```text
beta:           KL 惩罚强度，默认 0.1
learning_rate:  学习率，DPO 通常用更小的 lr，比如 5e-7 ~ 1e-6
num_train_epochs:
per_device_train_batch_size:
gradient_accumulation_steps:
max_length:     输入序列最大长度
max_prompt_length:  prompt 最大长度
```

**填写你认为最重要的参数和原因：**

```text
TODO:
```

---

## 实验记录

### 实验 1（数据验证）

```text
日期：
目标：验证 preference 数据格式是否正确
数据量：
运行命令：
结果：
发现的问题：
```

### 实验 2（DPO smoke test）

```text
日期：
模型：Qwen3-0.6B
目标：跑通 DPOTrainer 完整流程
GPU：
beta：
学习率：
数据量：
训练步数：
训练 loss（初始/最终）：
问题与解决：
```

### 实验 3（正式 DPO）

```text
日期：
模型：Qwen3-1.7B
目标：正式 DPO 训练，记录结果
GPU：
beta：
学习率：
数据量：
训练步数：
训练 loss（初始/最终）：
SFT vs DPO 对比结论：
```

---

## 问题记录

在这里记录学习和实验过程中遇到的问题：

```text
问题 1：
解决方法：

问题 2：
解决方法：
```

---

## 概念自查清单

学完本阶段，你应该能不看资料回答以下问题：

- [ ] preference dataset 包含哪三个字段？
- [ ] policy model 和 reference model 的区别是什么？
- [ ] DPO 和 RLHF 的关系是什么？
- [ ] beta 大小各有什么影响？
- [ ] 为什么 DPO 需要 reference model？
- [ ] TRL DPOTrainer 需要什么格式的数据？
- [ ] 如何设计 chosen 和 rejected？
- [ ] SFT 完成后为什么还要做 DPO？
- [ ] DPO 训练完，如何和 SFT 模型对比？
