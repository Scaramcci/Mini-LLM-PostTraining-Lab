# Stage 02 学习与实验计划：Tokenizer And Data

Stage 01 让你看到 GPT 是如何做 next token prediction 的；Stage 02 要补上一个关键问题：

```text
真实文本到底如何变成模型能吃的 input_ids？
```

这一阶段不需要 GPU，也不需要训练模型。你要把 tokenizer、chat template、attention mask、labels 和数据格式搞清楚，并为 Stage 03 的 SFT 准备一份小型 toy 数据。

建议用 5-7 天完成。如果你想更扎实，可以拉到 10 天。

## 0. 本阶段最终交付

完成 Stage 02 时，你应该交付：

```text
02_tokenizer_and_data/
├── README.md
├── notes.md
├── stage02_plan.md
├── scripts/
│   ├── inspect_tokenizer.py
│   ├── build_sft_toy_data.py
│   ├── validate_chat_format.py
│   └── split_dataset.py
├── data_examples/
│   ├── README.md
│   ├── sft_toy_10.jsonl
│   ├── preference_toy_10.jsonl
│   └── grpo_toy_10.jsonl
└── results/
    ├── tokenizer_examples.md
    ├── data_validation_report.md
    └── stage02_summary.md
```

同时维护：

```text
../docs/reading_notes/tokenizer.md
../docs/learning_log.md
../docs/concept_checklist.md
../reports/experiment_log.md
```

最终你需要能用自己的话讲清楚：

```text
1. token、token id、vocabulary 分别是什么？
2. BPE / SentencePiece 大概在解决什么问题？
3. tokenizer.encode 和 tokenizer.decode 做了什么？
4. special tokens 为什么重要？
5. chat template 为什么会影响 SFT？
6. input_ids、attention_mask、labels 分别是什么？
7. padding 和 truncation 分别解决什么问题？
8. 为什么训练集和验证集要分开？
9. 一条合格的 SFT JSONL 数据应该长什么样？
10. 为什么数据校验比直接训练更重要？
```

## 1. 学习顺序总览

推荐顺序：

```text
Day 1: 从 Stage 01 的字符级 token 过渡到真实 tokenizer
Day 2: 学 BPE / SentencePiece / special tokens
Day 3: 用 Qwen tokenizer 做 encode / decode 和 token inspection
Day 4: 学 chat template 和 messages 数据格式
Day 5: 构造 toy SFT 数据并校验
Day 6: 切分 train / validation，理解 labels / attention_mask
Day 7: 写 Stage 02 总结，准备进入 SFT
```

如果你每天时间有限，可以把 Day 3-5 拆慢一点。

## 2. Day 1：从字符级 token 过渡到真实 tokenizer

### 2.1 今天要理解什么

Stage 01 的 Shakespeare char-level 模型里，一个字符可以是一个 token。真实 LLM 通常不是这样，它会用 BPE / SentencePiece 之类的 tokenizer，把文本切成更灵活的子词片段。

你要理解：

```text
字符级 token 简单，但序列更长。
词级 token 直观，但没法处理所有新词。
子词 token 是折中：常见词可以整体表示，生僻词可以拆开表示。
```

例子：

```text
unbelievable
可能被拆成：un / believe / able
```

不同 tokenizer 的切法可能不同，所以：

```text
同一句话，用不同模型的 tokenizer，token id 和 token 数都可能不同。
```

### 2.2 今天要看的资料

按顺序看：

```text
1. Hugging Face Tokenizers 文档首页
   https://huggingface.co/docs/tokenizers/

2. 复习 Stage 01 的数据流：
   原始文本 -> token ids -> input batch -> logits -> loss
```

今天不需要看 tokenizer 的 Rust 实现。先理解概念。

### 2.3 今天要记录什么

写到 `notes.md`：

```markdown
## Day 1 Notes

### From char-level tokens to subword tokens

### Why not word-level only?

### Questions
```

写到 `../docs/reading_notes/tokenizer.md`：

```markdown
## Tokenizer Basics

### Token

### Vocabulary

### Token ID
```

### 2.4 今天完成标准

你能回答：

```text
为什么真实 LLM 不直接用字符级 tokenizer？
为什么也不直接用词级 tokenizer？
tokenizer 会不会影响模型效果？
```

## 3. Day 2：学习 BPE / SentencePiece / special tokens

### 3.1 今天要理解什么

今天重点理解“怎么切 token”以及“特殊 token 为什么重要”。

你要了解：

```text
BPE
SentencePiece
vocabulary
unknown token
bos token
eos token
pad token
special tokens
```

### 3.2 推荐理解顺序

```text
1. vocabulary
   tokenizer 维护一个 token 到 id 的表。

2. BPE
   从字符开始，逐步合并高频片段。

3. SentencePiece
   常见于多语言模型，通常直接从原始文本学习子词。

4. special tokens
   不是普通文本，而是控制边界和格式的标记。

5. eos token
   告诉模型回答应该结束。

6. pad token
   让一个 batch 里的序列长度对齐。
```

### 3.3 今天要看的资料

```text
1. Hugging Face Tokenizers quicktour
2. Qwen3 tokenizer / model docs
3. Transformers padding and truncation 文档，可选
```

### 3.4 今天要记录什么

写到 `notes.md`：

```markdown
## BPE / SentencePiece / Special Tokens

### BPE

### SentencePiece

### Special Tokens

### EOS / PAD
```

更新 `../docs/concept_checklist.md`，把能解释的 Stage 02 概念打勾。

### 3.5 今天完成标准

你能回答：

```text
BPE 的基本思想是什么？
special tokens 和普通文本 token 有什么区别？
如果没有 eos token，生成可能出现什么问题？
```

## 4. Day 3：用 Qwen tokenizer 做 encode / decode

### 4.1 今天要做什么

今天开始跑脚本，加载 Qwen tokenizer，观察同一句话如何变成 token id。

推荐模型：

```text
Qwen/Qwen3-0.6B
```

注意：这里只下载 tokenizer 文件，不下载完整模型权重。

### 4.2 环境检查

在项目根目录运行：

```bash
python3 scripts/check_env.py
```

如果还没装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4.3 运行 tokenizer inspection

在项目根目录：

```bash
python 02_tokenizer_and_data/scripts/inspect_tokenizer.py
```

指定模型：

```bash
python 02_tokenizer_and_data/scripts/inspect_tokenizer.py --model Qwen/Qwen3-0.6B
```

指定文本：

```bash
python 02_tokenizer_and_data/scripts/inspect_tokenizer.py --text "大模型到底是怎么处理中文的？"
```

如果网络无法访问 Hugging Face，就先读脚本和计划，等网络可用时再运行。

### 4.4 今天要观察什么

重点看：

```text
原始文本
token ids
tokens
decode 后是否等于原文
中文和英文 token 数差异
数字和符号如何切分
```

### 4.5 今天要记录什么

写到 `results/tokenizer_examples.md`：

```markdown
## tokenizer inspection: Qwen/Qwen3-0.6B

### Text

### Token IDs

### Tokens

### Decoded Text

### Observations
```

写到 `../reports/experiment_log.md`：

```markdown
## exp-02-001: inspect Qwen tokenizer

### Goal
观察 Qwen tokenizer 如何 encode/decode 中文、英文、数字和符号。

### Command

### Result

### Notes
```

### 4.6 今天完成标准

```text
能加载 Qwen tokenizer
能看到 token ids 和 tokens
能解释 encode / decode
能记录至少 3 个文本例子
```

## 5. Day 4：学习 chat template 和 messages 格式

### 5.1 今天要理解什么

LLM 聊天模型不是直接把 user / assistant 字符串随便拼起来训练的。不同模型有自己的 chat template，把 messages 转成模型期望的文本格式。

你要理解：

```text
system / user / assistant role
messages list
apply_chat_template
generation prompt
assistant answer boundary
```

### 5.2 推荐数据格式

本项目从这一阶段开始统一使用 messages 格式：

```json
{
  "id": "sft_001",
  "messages": [
    {"role": "system", "content": "You are a careful tool-use assistant."},
    {"role": "user", "content": "Calculate 17 * 23."},
    {"role": "assistant", "content": "{\"tool\": \"calculator\", \"arguments\": {\"expression\": \"17 * 23\"}}"}
  ],
  "meta": {
    "task": "calculator",
    "answer": "391"
  }
}
```

### 5.3 运行脚本观察 chat template

```bash
python 02_tokenizer_and_data/scripts/inspect_tokenizer.py --show-chat-template
```

### 5.4 今天要记录什么

写到 `notes.md`：

```markdown
## Chat Template

### Why messages are not enough

### What apply_chat_template returns

### Qwen chat format observations
```

写到 `results/tokenizer_examples.md`：

```markdown
## chat template example

### Messages

### Rendered Prompt

### Observations
```

### 5.5 今天完成标准

你能回答：

```text
messages 和真正送进模型的文本一样吗？
为什么不同模型需要不同 chat template？
add_generation_prompt 是什么？
SFT 时 assistant 回答边界为什么重要？
```

## 6. Day 5：构造 toy SFT 数据并校验

### 6.1 今天目标

构造一份小型 SFT 数据，为 Stage 03 做准备。

第一版任务建议：

```text
工具调用 + JSON 输出
```

原因：

```text
格式容易校验
后面可以接 Agent
也可以做 DPO / GRPO 的 reward
```

### 6.2 生成 toy 数据

在项目根目录运行：

```bash
python 02_tokenizer_and_data/scripts/build_sft_toy_data.py
```

默认会写入：

```text
02_tokenizer_and_data/data_examples/sft_toy_10.jsonl
```

### 6.3 校验数据

```bash
python 02_tokenizer_and_data/scripts/validate_chat_format.py 02_tokenizer_and_data/data_examples/sft_toy_10.jsonl
```

校验项包括：

```text
每行是合法 JSON
有 id
有 messages
messages 是 list
role 只能是 system/user/assistant
至少有一条 user
至少有一条 assistant
content 是非空字符串
assistant 内容如果像 JSON，需要能解析
```

### 6.4 今天要记录什么

写到 `results/data_validation_report.md`：

```markdown
## sft_toy_10 validation

### Command

### Result

### Problems

### Fixes
```

### 6.5 今天完成标准

```text
sft_toy_10.jsonl 存在
validate_chat_format.py 通过
你能解释这份数据每个字段的作用
```

## 7. Day 6：切分 train / validation，理解 labels 和 attention_mask

### 7.1 今天要理解什么

训练时不能只看训练集表现，还要留出验证集观察泛化。虽然 toy 数据很小，但你要从一开始养成习惯。

你要理解：

```text
train split
validation split
attention_mask
labels
padding
truncation
```

### 7.2 切分数据

```bash
python 02_tokenizer_and_data/scripts/split_dataset.py \
  02_tokenizer_and_data/data_examples/sft_toy_10.jsonl \
  --train-out 02_tokenizer_and_data/data_examples/sft_toy_train.jsonl \
  --val-out 02_tokenizer_and_data/data_examples/sft_toy_val.jsonl \
  --val-ratio 0.2
```

### 7.3 注意

`sft_toy_train.jsonl` 和 `sft_toy_val.jsonl` 是生成文件。小样例可以保留本地，但后面真实数据不要提交到 GitHub。

### 7.4 labels 的直觉

在 causal LM SFT 中：

```text
input_ids 是模型看到的 token id
labels 是模型要预测的 token id
attention_mask 表示哪些位置是真 token，哪些位置是 padding
```

很多 SFT trainer 会帮你处理 label shift，但你必须知道：

```text
模型本质仍然是在预测下一个 token。
```

### 7.5 今天要记录什么

写到 `notes.md`：

```markdown
## Train / Validation Split

### Why validation set matters

### attention_mask

### labels

### padding / truncation
```

写到 `results/stage02_summary.md` 的草稿里。

### 7.6 今天完成标准

```text
能把 SFT toy 数据切成 train / validation
能解释 attention_mask
能解释 labels
能解释 padding 和 truncation
```

## 8. Day 7：写阶段总结，准备 Stage 03

### 8.1 今天目标

把 Stage 02 收束成 Stage 03 可以直接使用的输入。

写：

```text
results/stage02_summary.md
```

### 8.2 总结应该包含什么

```markdown
# Stage 02 Summary

## Goal

## What I Learned

## Tokenizer Experiments

## Chat Template Findings

## Dataset Schema

## Validation Result

## Train / Validation Split

## Remaining Questions

## Ready For Stage 03
```

### 8.3 最后需要弄清楚什么

把下面问题写进 `stage02_summary.md`。

#### 1. tokenizer 做了什么？

```text
把文本切成 token，并映射成模型词表中的 token id。
```

#### 2. chat template 做了什么？

```text
把 system/user/assistant messages 渲染成模型训练或推理时期望的完整文本格式。
```

#### 3. SFT 数据一行应该包含什么？

```text
id
messages
meta
```

其中 `messages` 至少要有 user 和 assistant。

#### 4. 为什么需要数据校验？

```text
训练报错和训练效果差，很多时候不是模型问题，而是数据格式问题。
先校验可以避免把 GPU 时间浪费在坏数据上。
```

#### 5. Stage 03 会如何使用这些数据？

```text
SFTTrainer 会读取 messages 数据，通过 tokenizer 和 chat template 变成 input_ids / attention_mask / labels，然后训练模型模仿 assistant 的回答。
```

### 8.4 完成标准

```text
notes.md 有 tokenizer / chat template / labels 笔记
tokenizer_examples.md 有 encode/decode 示例
sft_toy_10.jsonl 格式正确
validate_chat_format.py 可运行
split_dataset.py 可运行
stage02_summary.md 回答了核心验收问题
```

## 9. 不需要 GPU

Stage 02 不需要租 GPU。

你可能需要网络下载：

```text
Qwen tokenizer 文件
```

但不需要下载完整模型权重，也不需要训练。

如果网络不稳定，可以先：

```text
读脚本
写 notes
手动查看 data_examples
等网络可用再运行 inspect_tokenizer.py
```

## 10. 不要踩的坑

### 10.1 不要把 toy 数据和真实数据混在一起

`data_examples/` 只放小样例。以后正式数据放到 `data/` 或本地路径，并且不要提交大文件。

### 10.2 不要忽略 chat template

同样的 messages，如果 template 不同，模型看到的文本就不同。SFT 时这会直接影响训练行为。

### 10.3 不要只看 decode 后是否一样

还要看 token 数、special tokens、中文切分、数字切分、换行和 JSON 标点。

### 10.4 不要等到训练时报数据错误

先用 `validate_chat_format.py` 把格式错误挡在 GPU 训练之前。

## 11. 最小完成路线

如果时间不够，至少完成：

```text
1. 看 Hugging Face Tokenizers 基础介绍
2. 跑 inspect_tokenizer.py
3. 看 Qwen chat template 输出
4. 生成 sft_toy_10.jsonl
5. 跑 validate_chat_format.py
6. 写 stage02_summary.md
```

## 12. 标准完成路线

推荐完成：

```text
1. Day 1-2 学 token / BPE / special tokens
2. Day 3 跑 tokenizer inspection
3. Day 4 学 chat template
4. Day 5 构造和校验 toy SFT 数据
5. Day 6 切分 train / validation
6. Day 7 写总结
```

## 13. 进阶完成路线

如果你还有精力：

```text
1. 比较 Qwen tokenizer 和 GPT-2 tokenizer 的 token 数差异
2. 构造 50-100 条工具调用 SFT 数据
3. 为每条数据加 task_type / difficulty / expected_tool
4. 统计平均 token length
5. 检查超长样本会不会被 truncation
```

进阶部分不是必须。Stage 02 的第一目标是：让 Stage 03 可以安全开始。
