# Stage 03 学习与实验计划：Qwen LoRA SFT

这一阶段的核心目标不是"把模型训练得多好"，而是第一次真正理解：

```text
base model 和 SFT model 的行为差异
LoRA 如何用极少参数让模型学会新行为
SFTTrainer 的训练流程从数据到 adapter 的完整路径
如何设计一个可以自动评测的 instruction 任务
base vs SFT 的对比怎么做
```

建议用 7-10 天完成。Stage 03 是后面 DPO 和 GRPO 的基础，不要跳过。

## 0. 本阶段最终交付

完成 Stage 03 时，你应该交付这些内容：

```text
03_qwen_sft_lora/
├── notes.md                      # 概念和代码笔记
├── configs/
│   ├── qwen3_0_6b_lora_sft.yaml  # 0.6B smoke test 配置
│   └── qwen3_1_7b_lora_sft.yaml  # 1.7B 正式实验配置
├── scripts/
│   ├── train_sft.py              # SFT 训练脚本
│   ├── infer_base.py             # base 模型推理
│   ├── infer_sft.py              # SFT 模型推理
│   └── compare_outputs.py        # base vs SFT 对比
└── results/
    ├── base_outputs.md            # base 模型输出记录
    ├── sft_outputs.md             # SFT 模型输出记录
    ├── sft_metrics.md             # 训练指标记录
    └── stage03_summary.md         # 阶段总结

../reports/experiment_log.md       # 每次训练实验记录
../reports/eval_results.md         # SFT 准确率和对比结果
../reports/gpu_usage_log.md        # GPU 使用记录
../docs/learning_log.md            # 每天学习记录
../docs/concept_checklist.md       # 勾选已掌握概念
```

最终你需要能用自己的话讲清楚：

```text
1. SFT 是什么？base model 为什么需要 SFT？
2. LoRA 在哪里插入参数？为什么这样能省显存？
3. instruction tuning 的数据格式是什么？
4. SFTTrainer 如何处理 labels？为什么要 mask 掉 prompt 部分？
5. LoRA adapter 是什么？它保存了哪些参数？
6. 如何把 adapter 加载回来并推理？
7. base model 和 SFT model 的输出有哪些典型差别？
8. 格式准确率和任务准确率分别衡量什么？
9. 这次实验有什么局限？
10. SFT 之后下一步是什么？
```

## 1. 学习顺序总览

推荐顺序：

```text
Day 1: 理解 SFT 和 instruction tuning 的基本直觉
Day 2: 理解 LoRA 原理，能讲清楚为什么省显存
Day 3: 学 SFTTrainer API 和配置，读 TRL 文档
Day 4: 写 train_sft.py，在 0.6B 上跑 smoke test
Day 5: 查看训练日志，写 infer_base.py 和 infer_sft.py
Day 6: 用 compare_outputs.py 对比 base vs SFT，记录评测结果
Day 7: 正式跑 Qwen3-1.7B 实验
Day 8: 记录指标，填写 results/，写 stage03_summary.md
```

如果你时间更紧，Day 7-8 可以合并为一天。

## 2. Day 1：理解 SFT 和 instruction tuning

### 2.1 今天要理解什么

先不急着写代码。今天只回答一个问题：

```text
base model 和 SFT model 有什么区别？
```

Base model 只做了 next token prediction，它：

```text
会续写文本
不区分 system / user / assistant
收到 instruction 后往往当做文本继续写，而不是"回答"
```

SFT model 被人工标注对话训练过，它：

```text
学会识别 system / user / assistant 格式
会按照 instruction 给出回答
更可能按照约定格式输出（比如 JSON）
```

一个关键直觉：

```text
SFT 不是让模型"更聪明"，而是让模型学会"怎么说话"。
```

### 2.2 理解 instruction tuning 的数据格式

SFT 数据是对话格式，通常长这样：

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant that returns JSON tool calls."},
    {"role": "user", "content": "Calculate 17 * 23 and return a JSON tool call."},
    {"role": "assistant", "content": "{\"tool\": \"calculator\", \"arguments\": {\"expression\": \"17 * 23\"}}"}
  ]
}
```

SFTTrainer 会：

```text
1. 用 tokenizer.apply_chat_template 把 messages 转成 token ids
2. 只对 assistant 的 token 计算 loss（prompt 部分 label 设为 -100）
3. 用这些数据做监督训练
```

为什么要 mask prompt？

```text
我们只想让模型学"如何回答"，不想让它也学去"预测用户问了什么"。
```

### 2.3 今天要看的资料

```text
1. TRL SFTTrainer 文档入口
   https://huggingface.co/docs/trl/sft_trainer

2. Qwen3 Hugging Face 模型页（了解 tokenizer 和 chat template）
   https://huggingface.co/Qwen/Qwen3-0.6B

3. 你在 Stage 02 写的 notes.md（复习 chat template 知识）
```

不需要一开始读完所有文档。今天重点是建立直觉。

### 2.4 今天要记录什么

写到 `notes.md`：

```markdown
## Day 1 Notes

### What is SFT?

### What is instruction tuning?

### base model vs SFT model

### Why mask the prompt in labels?

### Questions
```

写到 `../docs/learning_log.md`：

```markdown
## 2026-xx-xx

### Today
- 理解 SFT、instruction tuning、base vs SFT model 的基本区别。

### Concepts
- SFT
- instruction tuning
- label masking

### Questions
-

### Next
- 学 LoRA 原理，理解它为什么省显存。
```

### 2.5 今天完成标准

你能不看资料说出：

```text
SFT 改变了 base model 的哪些行为？
为什么训练时要 mask 掉 prompt 部分的 label？
什么是 instruction tuning 数据的基本格式？
```

## 3. Day 2：理解 LoRA 原理

### 3.1 今天要理解什么

LoRA（Low-Rank Adaptation）的核心思想：

```text
不去动 base model 原来的权重
而是在原有权重旁边并联插入两个小矩阵 A 和 B
训练时只更新 A 和 B，base model 冻结
推理时 A × B 的输出加到原来权重的输出上
```

示意图：

```text
原始权重：W₀ (d × d)
LoRA 插入：W₀ + B × A
其中 A: (r × d)，B: (d × r)，r << d

trainable 参数 = 2 × r × d，而不是 d × d
```

为什么省显存？

```text
梯度和 optimizer states 只需要为 A、B 两个小矩阵分配。
base model 不需要梯度，显存大幅减少。
```

### 3.2 LoRA 插在哪里？

通常插在 Transformer 的 attention 层，具体是：

```text
Q_proj (query projection)
V_proj (value projection)
```

也可以插到：

```text
K_proj
O_proj
gate_proj / up_proj / down_proj (FFN)
```

PEFT 的 LoraConfig 控制这些：

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,           # 缩放系数
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

### 3.3 QLoRA 和 LoRA 的区别

```text
LoRA：
  base model 用 float16 / bfloat16 加载
  只有 LoRA 参数 float32 训练

QLoRA：
  base model 用 4-bit（NF4）量化加载，进一步省显存
  LoRA 参数仍然 float32
  需要 bitsandbytes
```

第一轮建议 LoRA（不量化），更好调试。如果 4090 显存不够，再尝试 QLoRA。

### 3.4 今天要看的资料

```text
1. PEFT LoRA 文档
   https://huggingface.co/docs/peft/developer_guides/lora

2. 可选：LoRA 原始 paper abstract
   https://arxiv.org/abs/2106.09685
```

论文不用硬啃。现在只理解：

```text
LoRA 在做什么
rank r 控制什么
为什么 adapter 文件很小
```

### 3.5 今天要记录什么

写到 `notes.md`：

```markdown
## LoRA Notes

### Core Idea

### A and B Matrices

### Why Memory-Efficient

### target_modules

### r and lora_alpha

### LoRA vs QLoRA
```

写到 `../docs/reading_notes/sft_lora.md`：

```markdown
## LoRA

### 原理

### 在 Transformer 哪里插入

### 参数量对比

### PEFT LoraConfig 关键参数

### 参考来源
```

更新 `../docs/concept_checklist.md`，能解释的打勾。

### 3.6 今天完成标准

你能回答：

```text
LoRA 为什么不需要更新 base model 权重？
adapter 文件为什么比完整 checkpoint 小很多？
rank r 越大意味着什么？
QLoRA 在 LoRA 基础上多做了什么？
```

## 4. Day 3：学 SFTTrainer API，读 TRL 文档

### 4.1 今天目标

今天不写训练脚本，先理解 SFTTrainer 的 API 和关键参数。

### 4.2 SFTTrainer 的基本用法

最简版本：

```python
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir="./outputs/qwen3_0_6b_sft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
    max_seq_length=512,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model("./outputs/qwen3_0_6b_sft/adapter")
```

### 4.3 关键参数逐一理解

理解这些参数，不要死记：

```text
r
  LoRA 的 rank。越大 adapter 越有表达能力，但参数越多。
  第一轮建议 r=8 或 r=16。

lora_alpha
  LoRA 缩放系数，通常设为 r 的 2 倍，比如 r=16, lora_alpha=32。

target_modules
  LoRA 插在哪些层。q_proj + v_proj 是常用组合。

per_device_train_batch_size
  每张 GPU 每步处理多少样本。

gradient_accumulation_steps
  多少步合并一次梯度更新。实际 batch size = per_device × accumulation。

max_seq_length
  输入最大 token 数。越短越省显存。第一轮用 512。

bf16
  使用 bfloat16 精度，4090 支持且推荐。

packing
  把多条短样本打包成一条，提高显存利用率。初学先用 False。
```

### 4.4 数据格式要求

SFTTrainer 接受两种格式：

```text
格式 1：messages 列（推荐）
  每行有 "messages" 字段，内含 list of dicts（role, content）
  SFTTrainer 自动调用 apply_chat_template

格式 2：text 列（已处理好的字符串）
  每行有 "text" 字段，是完整的对话字符串
```

第一版建议用格式 1，更清晰：

```python
from datasets import Dataset

data = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Return a JSON tool call for add(3, 5)."},
            {"role": "assistant", "content": "{\"tool\": \"add\", \"arguments\": {\"a\": 3, \"b\": 5}}"}
        ]
    },
    ...
]

dataset = Dataset.from_list(data)
```

### 4.5 今天要看的资料

```text
TRL SFTTrainer 文档（重点读 API 部分）：
https://huggingface.co/docs/trl/sft_trainer

TRL SFTConfig 参数：
https://huggingface.co/docs/trl/sft_trainer#trl.SFTConfig
```

### 4.6 今天要记录什么

写到 `notes.md`：

```markdown
## SFTTrainer Notes

### Key API

### Key Parameters

### Data Format

### LoRA Config

### Questions
```

### 4.7 今天完成标准

你能回答：

```text
SFTTrainer 的最小初始化需要哪几个参数？
max_seq_length 影响什么？
gradient_accumulation_steps 是什么意思？
SFTTrainer 如何识别哪些 token 算 loss？
```

## 5. Day 4：写 train_sft.py，跑 0.6B smoke test

### 5.1 今天目标

今天第一次真正训练 LoRA SFT。目标不是效果好，而是：

```text
代码能跑起来
SFTTrainer 能正常训练
LoRA adapter 能成功保存
训练日志能看到 loss 在降
```

### 5.2 环境检查

在项目根目录确认环境：

```bash
python scripts/check_env.py
```

确认安装：

```bash
pip install trl peft bitsandbytes accelerate
```

验证可以加载 Qwen3：

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
print(tokenizer.apply_chat_template(
    [{"role": "user", "content": "hello"}],
    tokenize=False, add_generation_prompt=True
))
```

### 5.3 脚本结构：train_sft.py

在 `scripts/train_sft.py` 里，推荐按这个顺序组织：

```text
1. 读取 config 文件（yaml）
2. 加载 tokenizer
3. 加载 dataset（从 jsonl 文件）
4. 数据校验（长度、格式）
5. 加载 base model
6. 创建 LoraConfig
7. 创建 SFTConfig（训练参数）
8. 创建 SFTTrainer
9. trainer.train()
10. trainer.save_model()
11. 输出训练摘要
```

关键代码段：

```python
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import json

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def load_dataset_from_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return Dataset.from_list(data)

def main(config_path):
    cfg = load_config(config_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype="auto",
        device_map="auto",
    )

    train_ds = load_dataset_from_jsonl(cfg["train_data"])
    eval_ds = load_dataset_from_jsonl(cfg["eval_data"])

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        bf16=cfg.get("bf16", True),
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        max_seq_length=cfg["max_seq_length"],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"] + "/adapter")
    print("Training complete. Adapter saved.")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
```

### 5.4 写配置文件：configs/qwen3_0_6b_lora_sft.yaml

```yaml
# Qwen3-0.6B LoRA SFT - smoke test config

model_name: "Qwen/Qwen3-0.6B"
train_data: "../../data/sft_toy/train.jsonl"
eval_data: "../../data/sft_toy/eval.jsonl"
output_dir: "./results/qwen3_0_6b_sft_run01"

# LoRA
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules:
  - "q_proj"
  - "v_proj"

# Training
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
lr_scheduler_type: "cosine"
warmup_ratio: 0.05
bf16: true
max_seq_length: 512

# Logging
logging_steps: 10
save_steps: 100
eval_steps: 50
```

### 5.5 跑 smoke test

确认 GPU 可用后，运行：

```bash
cd 03_qwen_sft_lora
python scripts/train_sft.py configs/qwen3_0_6b_lora_sft.yaml
```

smoke test 目标：

```text
不报错
trainer 打印出 training loss
有 step 数在递增
adapter 目录被创建
```

如果本地没有 GPU，先在 CPU 上用极少数据（10 条）测试代码逻辑，再到 GPU 服务器跑正式训练。

### 5.6 今天要记录什么

写到 `../reports/experiment_log.md`：

```markdown
## exp-003: Qwen3-0.6B LoRA SFT smoke test

### Date

### Device

### Command
```bash
python scripts/train_sft.py configs/qwen3_0_6b_lora_sft.yaml
```

### Config
- model: Qwen3-0.6B
- lora_r: 8
- lora_alpha: 16
- target_modules: q_proj, v_proj
- max_seq_length: 512
- per_device_batch_size: 4
- gradient_accumulation: 4
- epochs: 3
- data_size: (填实际条数)

### Result
- final train loss:
- final eval loss:
- adapter saved:

### Problems

### Notes
```

### 5.7 今天完成标准

```text
train_sft.py 代码写好
configs/qwen3_0_6b_lora_sft.yaml 写好
smoke test 能跑起来（不报错，loss 开始下降）
adapter 目录生成
experiment_log 有记录
```

## 6. Day 5：写推理脚本，对比 base vs SFT

### 6.1 今天目标

训练完成后，写推理脚本验证效果：

```text
infer_base.py：加载 base model，对同一组 prompt 推理
infer_sft.py：加载 base model + LoRA adapter，对同一组 prompt 推理
compare_outputs.py：并排对比输出
```

### 6.2 infer_base.py 关键逻辑

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
model.eval()

def infer(prompt_messages, max_new_tokens=200):
    text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy，方便对比
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

# 固定测试 prompts
test_prompts = [
    [
        {"role": "system", "content": "You are a helpful assistant that returns JSON tool calls."},
        {"role": "user", "content": "Calculate 3 + 5 and return a JSON tool call."},
    ],
    # 更多测试 prompt...
]

results = []
for i, prompt in enumerate(test_prompts):
    output = infer(prompt)
    results.append({"prompt_id": i, "output": output})
    print(f"[{i}] {output}\n")

with open("results/base_outputs.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

### 6.3 infer_sft.py 关键逻辑

加载 adapter 只需要两行额外代码：

```python
from peft import PeftModel

base_model_name = "Qwen/Qwen3-0.6B"
adapter_path = "./results/qwen3_0_6b_sft_run01/adapter"

model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)  # 加载 LoRA adapter
model.eval()
```

其余推理逻辑与 infer_base.py 完全相同，保证对比公平。

### 6.4 compare_outputs.py

读取两份输出，并排打印：

```python
import json

with open("results/base_outputs.jsonl") as f:
    base = [json.loads(l) for l in f]

with open("results/sft_outputs.jsonl") as f:
    sft = [json.loads(l) for l in f]

for b, s in zip(base, sft):
    print(f"=== Prompt {b['prompt_id']} ===")
    print(f"[BASE]  {b['output']}")
    print(f"[SFT]   {s['output']}")
    print()
```

### 6.5 今天要记录什么

将对比结果写到 `results/base_outputs.md` 和 `results/sft_outputs.md`：

```markdown
## Experiment: qwen3_0_6b_sft_run01

### Prompt 1
**User:** Calculate 3 + 5 and return a JSON tool call.

**Base output:**
（粘贴原始输出）

**SFT output:**
（粘贴原始输出）

**Observation:**
- base: 是否符合 JSON 格式？
- SFT: 是否符合 JSON 格式？
- 格式差异明显吗？

### Prompt 2
...
```

### 6.6 今天完成标准

```text
infer_base.py 跑通，输出保存到 base_outputs.jsonl
infer_sft.py 跑通，输出保存到 sft_outputs.jsonl
compare_outputs.py 能并排打印
base_outputs.md 和 sft_outputs.md 有观察记录
```

## 7. Day 6：评测与记录指标

### 7.1 今天目标

把主观对比变成数字，记录指标。

### 7.2 设计评测指标

因为任务是"JSON 工具调用格式"，可以设计这些指标：

```text
格式准确率（format_acc）：
  输出是否合法 JSON？（json.loads 能否不报错）

工具准确率（tool_acc）：
  JSON 里的 "tool" 字段是否等于预期值？

参数准确率（args_acc）：
  JSON 里的 "arguments" 字段是否完整匹配？

总体准确率（overall_acc）：
  三项全对才算对。
```

写到 `compare_outputs.py` 里：

```python
def evaluate(output, expected_tool, expected_args):
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return {"format_ok": False, "tool_ok": False, "args_ok": False}

    tool_ok = parsed.get("tool") == expected_tool
    args_ok = parsed.get("arguments") == expected_args
    return {"format_ok": True, "tool_ok": tool_ok, "args_ok": args_ok}
```

### 7.3 填写 results/sft_metrics.md

```markdown
# SFT Metrics

## Experiment: qwen3_0_6b_sft_run01

| Metric | Base | SFT |
|--------|------|-----|
| format_acc | x% | x% |
| tool_acc | x% | x% |
| args_acc | x% | x% |
| overall_acc | x% | x% |

## Observations
-

## Failure Cases
-
```

### 7.4 更新 reports/eval_results.md

```markdown
| Stage | Model | Task | Format Acc | Tool Acc | Args Acc | Notes |
|-------|-------|------|------------|----------|----------|-------|
| 03-SFT | Qwen3-0.6B | tool_call_json | x% | x% | x% | smoke test |
```

### 7.5 今天完成标准

```text
评测函数写好
base 和 SFT 的格式准确率都已计算
sft_metrics.md 有数字记录
eval_results.md 已更新
```

## 8. Day 7：正式跑 Qwen3-1.7B 实验

### 8.1 今天目标

0.6B smoke test 通过后，用 Qwen3-1.7B 跑一次正式实验，记录更有参考价值的结果。

### 8.2 写配置文件：configs/qwen3_1_7b_lora_sft.yaml

```yaml
# Qwen3-1.7B LoRA SFT - formal experiment

model_name: "Qwen/Qwen3-1.7B"
train_data: "../../data/sft_toy/train.jsonl"
eval_data: "../../data/sft_toy/eval.jsonl"
output_dir: "./results/qwen3_1_7b_sft_run01"

# LoRA
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  - "q_proj"
  - "v_proj"
  - "k_proj"
  - "o_proj"

# Training
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1.5e-4
lr_scheduler_type: "cosine"
warmup_ratio: 0.05
bf16: true
max_seq_length: 512

# Logging
logging_steps: 10
save_steps: 100
eval_steps: 50
```

注意：1.7B 比 0.6B 参数多约 3 倍，显存需求更高。4090 24GB 应该够，注意监控。

### 8.3 跑训练

```bash
python scripts/train_sft.py configs/qwen3_1_7b_lora_sft.yaml
```

再跑推理对比：

```bash
python scripts/infer_base.py --model Qwen/Qwen3-1.7B --out results/base_1_7b_outputs.jsonl
python scripts/infer_sft.py --adapter results/qwen3_1_7b_sft_run01/adapter --out results/sft_1_7b_outputs.jsonl
python scripts/compare_outputs.py --base results/base_1_7b_outputs.jsonl --sft results/sft_1_7b_outputs.jsonl
```

### 8.4 GPU 使用注意

```text
提前写好所有命令
租 GPU 后先确认环境
下载模型预计 5-15 分钟（取决于带宽）
训练 1.7B 预计 3-8 小时（取决于数据量和 epochs）
跑完立刻保存 adapter 和指标，然后关机
```

写到 `../reports/gpu_usage_log.md`：

```markdown
## 2026-xx-xx GPU Session

- Provider:
- GPU:
- Price:
- Start time:
- End time:
- Stage: 03 Qwen SFT LoRA
- Goal: 正式跑 Qwen3-1.7B LoRA SFT
- Model: Qwen/Qwen3-1.7B
- Dataset: sft_toy (xxx 条)
- Command: python scripts/train_sft.py configs/qwen3_1_7b_lora_sft.yaml
- Output path: results/qwen3_1_7b_sft_run01/
- Result:
- Cost estimate:
- Mistakes / notes:
```

### 8.5 今天完成标准

```text
qwen3_1_7b_lora_sft.yaml 写好
1.7B 训练跑通
1.7B 的 base vs SFT 对比完成
eval_results.md 有 1.7B 的指标记录
```

## 9. Day 8：写阶段总结

### 9.1 今天目标

把 Stage 03 收束成一个可以复盘的阶段成果。

写：

```text
results/stage03_summary.md
```

### 9.2 总结应该包含什么

```markdown
# Stage 03 Summary

## Goal

## What I Did

## Models

## Data

## Commands

## Key Results

## What I Learned

## Code Understanding

## LoRA Deep Dive

## Failure Cases

## Remaining Questions

## Next Stage
```

### 9.3 你最后需要弄清楚什么

这是 Stage 03 的核心验收问题。把答案写进 `stage03_summary.md`。

#### 1. SFT 改变了什么？

你要能讲：

```text
SFT 之前：base model 见到 "Return a JSON tool call" 可能会直接续写废话。
SFT 之后：model 学会了这类 prompt 对应的 response 格式。
SFT 改变的是"输出倾向"，不是"底层知识"。
```

#### 2. LoRA adapter 里存了什么？

你要能讲：

```text
adapter 文件里存的是 A 和 B 两组小矩阵。
base model 权重没有变化。
推理时把 W₀ + B×A 当做等效权重。
```

#### 3. 训练 loss 和格式准确率的关系

你要能讲：

```text
训练 loss 下降说明模型在训练数据上的 next token prediction 变好了。
但 loss 不直接等于"格式准确率"。
需要用固定 prompt 做推理才能知道格式准确率。
```

#### 4. 这次实验有什么局限？

你要能讲：

```text
数据量很小（100-500 条 toy 数据）
没有验证 out-of-distribution 泛化能力
没有做更大模型的对比
没有用真实 benchmark 评测
结果只能说明"模型在 toy 数据上学会了某种格式"
```

#### 5. 下一步是什么？

你要能讲：

```text
Stage 04 DPO：用 chosen/rejected 对来让模型更偏向高质量输出。
SFT 是基础，DPO 是在 SFT 基础上做的偏好对齐。
```

### 9.4 完成标准

```text
notes.md 有 SFT、LoRA、SFTTrainer 的概念和代码笔记
experiment_log.md 有 0.6B 和 1.7B 两组实验记录
base_outputs.md 和 sft_outputs.md 有对比输出
sft_metrics.md 有量化指标
stage03_summary.md 能回答验收问题
eval_results.md 有 Stage 03 的结果行
concept_checklist.md 勾选 Stage 03 相关概念
```

## 10. GPU 使用建议

### 10.1 本地先跑代码逻辑

在租 GPU 之前，本地先用 CPU + 10 条数据测试 train_sft.py 不报错：

```bash
# 临时修改 config，数据 10 条，epochs 1，max_iters 5
python scripts/train_sft.py configs/qwen3_0_6b_lora_sft.yaml
```

这样到 GPU 服务器上只需要换数据路径和参数，不用现场调试代码。

### 10.2 GPU 租用建议

```text
Qwen3-0.6B smoke test：
  本地 MPS/CPU 可以做最小测试
  如果要完整跑通，租 4090 1-2 小时

Qwen3-1.7B 正式实验：
  4090 24GB，按小时租
  预计 3-8 小时（取决于数据量）

不要月租。按小时，跑完立刻关机。
```

### 10.3 租 GPU 前检查清单

```text
1. 今天要跑哪个实验？（0.6B smoke test / 1.7B 正式实验）
2. 数据准备好了吗？（sft_toy/train.jsonl 和 eval.jsonl 确认路径）
3. 代码本地测试不报错了吗？
4. 配置文件写好了吗？
5. 运行命令写好了吗？
6. adapter 保存路径确认了吗？
7. 跑完如何下载 adapter 和 metrics？
```

## 11. 不要踩的坑

### 11.1 不要跳过 0.6B smoke test

直接跑 1.7B，代码有 bug 会浪费更多 GPU 时间。一定先用 0.6B 跑通。

### 11.2 不要把 adapter 忘了下载

训练结束后立刻：

```bash
# 下载 adapter 到本地
scp -r user@server:path/to/adapter ./results/adapter_backup/

# 或者 push 到 Hugging Face Hub
```

大 base model 不需要保存，之后可以重新下载。但 adapter 和指标必须保留。

### 11.3 不要忘记 labels 里的 -100

SFTTrainer 会自动处理 label masking，但如果你手写 dataset，一定确认 prompt 部分 label 设为 -100。否则模型会同时学"怎么问问题"和"怎么回答"，浪费训练资源。

### 11.4 不要只看 loss，要看实际输出

training loss 下降不代表格式准确率提升。一定要跑推理对比，才能真正验证 SFT 有没有效。

### 11.5 不要一开始上大数据

第一轮 200-500 条 toy 数据足够验证流程。大数据会让调试变慢，遇到问题更难排查。

## 12. 最小完成路线

如果你时间不够，至少完成：

```text
1. 理解 SFT 和 LoRA 基本原理
2. 写好 train_sft.py 和 configs/qwen3_0_6b_lora_sft.yaml
3. 用 0.6B 跑通一次完整流程（train -> save adapter -> infer -> compare）
4. 记录 base vs SFT 输出对比（哪怕只有 5 条 prompt）
5. 写 stage03_summary.md
```

## 13. 标准完成路线

推荐完成：

```text
1. Day 1-2 学 SFT、instruction tuning、LoRA 原理
2. Day 3 读 SFTTrainer 文档，理解 API
3. Day 4 写脚本，用 0.6B 跑 smoke test
4. Day 5 写推理脚本，对比 base vs SFT
5. Day 6 计算格式准确率，记录指标
6. Day 7 租 GPU，跑 1.7B 正式实验
7. Day 8 写总结，更新 eval_results.md
```

## 14. 进阶完成路线

如果你还有精力：

```text
1. 同时插入 q_proj / k_proj / v_proj / o_proj，对比和只插 q+v 的差别
2. 尝试不同 rank（r=4 / r=8 / r=16 / r=32），观察 loss 和准确率变化
3. 尝试 QLoRA（bitsandbytes 4-bit 量化），对比显存占用
4. 尝试 max_seq_length=1024，看是否对长输出有帮助
5. 写一个自动评测脚本，批量跑 20 条 prompt 并输出 accuracy
6. 把 LoRA adapter 合并到 base model 并用 transformers 直接加载（merge_and_unload）
```

进阶部分不是必须。先完成标准路线。
