# Stage 02 Notes

## Goal

理解 tokenizer、chat template 和 SFT 数据格式，为 Stage 03 的 Qwen LoRA SFT 准备可校验 toy 数据。

## Day 1 Notes

### From char-level tokens to subword tokens

Stage 01 里的 char-level tokenizer 把每个字符都当成一个 token。这个做法非常直观：词表小、实现简单、不会遇到未知词。但真实 LLM 通常不用纯字符级 token，因为同一句话会被拉成长得多的 token 序列，模型需要在更长上下文里学习单词、短语和语义结构，训练和推理都更浪费。

真实 tokenizer 更常见的做法是 subword tokenization。它介于字符级和词级之间：高频词或常见片段可以作为一个 token，低频词、新词、拼写变化和多语言文本可以被拆成更小的片段。这样既不会像字符级那样过长，也不会像纯词级那样遇到新词就失效。

### Why not word-level only?

纯 word-level tokenizer 看起来符合人的直觉，但问题很多：

- 词表会非常大，尤其是英文变形、代码、数字、标点、URL、多语言混排都会制造大量“新词”。
- 对中文、日文等没有空格分词边界的语言不自然，需要额外分词器，错误会传递到模型输入。
- 遇到训练集中没有出现过的词，只能用 unknown token，模型会丢掉很多细节。
- 同一个词的大小写、前后标点、复数、时态、拼写变体都可能变成不同词项，泛化不如 subword 灵活。

所以 subword 是折中方案：把“可复用的文字片段”作为单位，让模型在固定词表内覆盖开放世界文本。

### Questions

- 同一个文本换不同 tokenizer 后 token 数会差多少？这会直接影响显存和上下文长度。
- Qwen tokenizer 对中文是更偏向单字、词片段，还是混合切分？
- SFT 时 prompt 部分和 assistant answer 部分是否应该都计算 loss？后面需要看 trainer 的 label masking 逻辑。

## Reading Notes

### Hugging Face Tokenizers

Source: https://huggingface.co/docs/tokenizers/

Hugging Face Tokenizers 负责把原始文本转成模型可用的 token 序列和 token id。一个完整 tokenizer 通常不只是“按空格切开”，而是包含 normalization、pre-tokenization、model、post-processing、decoding 等步骤。

核心理解：

- `encode`：文本 -> token / token id / attention mask 等结构化结果。
- `decode`：token id -> 可读文本。
- `Encoding` 里通常能看到 `tokens`、`ids`、`attention_mask`、`offsets` 等信息。
- BPE 训练的大致过程是从字符开始，反复合并语料中最常见的相邻 token 对，直到达到目标词表大小。
- padding 后 tokenizer 会生成 attention mask，用 1 表示真实 token，用 0 表示 padding token。

### Transformers Chat Templates

Source: https://huggingface.co/docs/transformers/chat_templating

聊天模型虽然输入看起来是 `system/user/assistant` messages，但模型真正接收的仍然是一串 token。chat template 的作用是把 messages 渲染成该模型训练时见过的格式，包括角色标记、轮次边界、结束符等控制 token。

关键点：

- `messages` 是结构化数据；`apply_chat_template` 会把它变成模型期望的 prompt 文本或 token id。
- 不同模型的模板不同，不能随便混用。模板错了，模型可能看不懂角色边界，效果会明显变差。
- 推理时常用 `add_generation_prompt=True`，在末尾加 assistant 起始标记，提示模型接下来该生成 assistant 回复。
- SFT 训练时通常使用完整的 user + assistant 对话，并设置 `add_generation_prompt=False`，因为 assistant 答案已经在样本里。
- 如果先 `apply_chat_template(tokenize=False)` 再手动 tokenize，要注意不要重复添加 special tokens。

### Qwen Tokenizer

Source: https://huggingface.co/docs/transformers/model_doc/qwen3

本阶段脚本默认使用 `Qwen/Qwen3-0.6B` 的 tokenizer。Qwen3 是 causal LM 系列，Transformers 文档中 Qwen3 配置的默认 `vocab_size` 是 151936。脚本会通过 `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` 加载 tokenizer，并观察普通文本、中文、算式、JSON 字符串的 token ids、tokens 和 decode 结果。

需要重点观察：

- `bos_token`、`eos_token`、`pad_token` 及其 id 是否存在。
- 中文、英文、数字、符号、JSON 中的引号和大括号会如何切分。
- `decode(tokenizer(text).input_ids)` 是否能还原原始文本。
- `apply_chat_template` 渲染后是否出现 Qwen 预期的 role 边界和结束标记。

## Concept Notes

### Token / Token ID / Vocabulary

Token 是 tokenizer 切出来的最小建模单位，可以是一个字符、一个词、一个子词片段、一个标点，或者一个特殊控制标记。Token ID 是 token 在词表里的整数编号。Vocabulary 是 token 到 id 的映射表，也是模型 embedding 矩阵的索引空间。

模型本身不直接理解字符串，它看到的是 `input_ids`。例如文本先被 tokenizer 切成 tokens，再查 vocabulary 变成整数序列，模型用这些整数去查 embedding。

### BPE / SentencePiece

BPE 的直觉：从小单位开始，统计语料里最常一起出现的相邻片段，然后把它们合并成新 token。重复这个过程后，常见词或常见子串会被压缩成较少 token，少见词仍然可以拆成更小片段。

SentencePiece 更像是一套 tokenizer 工具和方法，常用于多语言模型。它可以直接从 raw text 学习子词，不强依赖空格分词；常见模型包括 BPE 和 Unigram。对于中文这类没有天然空格边界的语言，这种思路比纯 word-level 更自然。

### Special Tokens

Special tokens 不是普通文本内容，而是用来表达结构和控制信息的 token。例如：

- `bos_token`：序列开始。
- `eos_token`：序列结束，告诉模型回答可以停止。
- `pad_token`：batch 对齐长度时补位。
- role/control tokens：标记 system、user、assistant 或消息边界。

这些 token 会影响模型对输入结构的理解。特别是在聊天模型里，角色边界和结束标记如果错了，模型可能把 user 内容续写成 assistant 内容，或者不知道何时停止。

### Chat Template

Chat template 是 messages 到模型文本格式的转换规则。它把：

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
```

渲染成模型预训练或指令微调时熟悉的格式。SFT 时必须使用目标模型自己的 chat template，否则训练数据的格式和模型已有对话格式不一致，会让微调学习目标变得混乱。

### input_ids / attention_mask / labels

`input_ids` 是 tokenizer 输出的 token id 序列，也是模型的主要输入。

`attention_mask` 标记哪些位置是真实 token，哪些位置是 padding。通常 1 表示有效 token，0 表示 padding，模型计算 attention 时会忽略 padding 部分。

`labels` 是训练 loss 使用的目标 token id。对 causal LM 来说，本质任务仍然是 next token prediction。SFT 中常见做法是让 assistant 回复部分参与 loss，而 prompt 部分可用 `-100` mask 掉；具体是否 mask 由 data collator 或 trainer 配置决定。

### Padding / Truncation

Padding 解决 batch 内序列长度不一致的问题：把短样本补到同一长度，方便组成张量。Padding token 本身不应影响训练，所以需要 attention mask 和 label mask 配合。

Truncation 解决序列过长的问题：当 token 数超过模型最大上下文或训练配置的 `max_length` 时截断。截断很危险，因为可能切掉 assistant 答案、JSON 结尾或 eos token，所以 SFT 数据最好先统计长度分布，再决定最大长度。

## Dataset Notes

### SFT Toy Schema

本阶段 SFT toy 数据使用 JSONL，每一行是一条独立样本。当前文件是 `data_examples/sft_toy_10.jsonl`，共 10 条，任务包括 `calculator` 和 `json_validator`。

单条样本结构：

```json
{
  "id": "sft_calc_001",
  "messages": [
    {"role": "system", "content": "You are a careful tool-use assistant."},
    {"role": "user", "content": "Use a tool to calculate: 17 * 23"},
    {"role": "assistant", "content": "{\"tool\": \"calculator\", \"arguments\": {\"expression\": \"17 * 23\"}}"}
  ],
  "meta": {
    "task": "calculator",
    "answer": "391"
  }
}
```

字段含义：

- `id`：样本唯一标识，方便排查错误和复现实验。
- `messages`：对话数据主体，后续会通过 chat template 转成训练文本。
- `role`：只能是 `system`、`user`、`assistant`。
- `content`：每轮消息文本；assistant 在本 toy 数据里输出工具调用 JSON 字符串。
- `meta`：辅助信息，不一定直接送入模型，可用于评估、过滤或构造 reward。

### Validation Result

已运行：

```bash
python3 02_tokenizer_and_data/scripts/validate_chat_format.py 02_tokenizer_and_data/data_examples/sft_toy_10.jsonl
```

结果：

```text
checked 10 examples: OK
```

校验脚本检查了每行是否是合法 JSON、是否有非空 `id`、`messages` 是否为非空 list、role 是否合法、是否至少包含 user 和 assistant、content 是否非空，以及 assistant 内容如果像 JSON 是否能被 `json.loads` 解析。

### Train / Validation Split

已运行：

```bash
python3 02_tokenizer_and_data/scripts/split_dataset.py \
  02_tokenizer_and_data/data_examples/sft_toy_10.jsonl \
  --train-out 02_tokenizer_and_data/data_examples/sft_toy_train.jsonl \
  --val-out 02_tokenizer_and_data/data_examples/sft_toy_val.jsonl \
  --val-ratio 0.2
```

结果：

```text
input: 10
train: 8 -> 02_tokenizer_and_data/data_examples/sft_toy_train.jsonl
val: 2 -> 02_tokenizer_and_data/data_examples/sft_toy_val.jsonl
```

训练集用于更新模型参数，验证集用于观察模型是否只是记住训练数据。虽然 toy 数据很小，验证指标没有统计意义，但保留 split 流程可以让 Stage 03 的训练脚本从一开始就按正规流程组织。

## Questions

- Qwen 的 chat template 在 tool-use 场景下是否需要额外工具 schema，还是先让 assistant 直接输出 JSON 字符串？
- Stage 03 训练时应该只对 assistant JSON 输出算 loss，还是整段 chat template 都算 loss？
- toy 数据是否需要加入负例或格式错误样本，帮助模型学会拒绝无效工具调用？

## Next

- 跑 `inspect_tokenizer.py`，观察 Qwen tokenizer。
