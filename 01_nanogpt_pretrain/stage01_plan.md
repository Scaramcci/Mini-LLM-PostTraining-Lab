# Stage 01 学习与实验计划：nanoGPT Pretraining

这一阶段的核心目标不是“把模型训练得多好”，而是第一次真正理解：

```text
文本如何变成 token
token 如何进入 Transformer
Transformer 如何做 next token prediction
loss 是怎么来的
训练日志说明了什么
模型为什么能生成文本
```

建议用 5-7 天完成。如果你想慢一点，也可以拉到 10 天。不要急着往 SFT 走，Stage 01 是后面所有内容的地基。

## 0. 本阶段最终交付

完成 Stage 01 时，你应该交付这些内容：

```text
01_nanogpt_pretrain/
├── notes.md
├── results/
│   ├── sample_outputs.md
│   └── stage01_summary.md
└── nanoGPT/                 # 本地克隆，不提交到 GitHub

../docs/learning_log.md      # 每天学习记录
../docs/concept_checklist.md # 勾选已掌握概念
../reports/experiment_log.md # 训练实验记录
../reports/eval_results.md   # loss / sample 摘要
```

最终你需要能用自己的话讲清楚：

```text
1. 什么是 causal language modeling？
2. 为什么 GPT 训练目标是 next token prediction？
3. token embedding 和 position embedding 分别在做什么？
4. attention 里的 Q/K/V 是什么？
5. causal mask 为什么不能去掉？
6. train loss 和 validation loss 分别代表什么？
7. 为什么 loss 降了，生成文本通常会变好？
8. nanoGPT 里的 GPT / Block / CausalSelfAttention / MLP 分别负责什么？
9. sample.py 是如何从模型 logits 生成下一个 token 的？
10. 这次小实验的局限是什么？
```

## 1. 学习顺序总览

推荐顺序：

```text
Day 1: 建立语言模型直觉，不看太多代码
Day 2: 学 Transformer 的核心结构
Day 3: 读 nanoGPT README 和项目结构
Day 4: 跑 tiny Shakespeare 最小训练
Day 5: 读 nanoGPT 关键代码
Day 6: 采样、记录结果、解释 loss
Day 7: 写 Stage 01 总结
```

如果你时间更宽松，每个 Day 可以拆成两天。

## 2. Day 1：先建立语言模型直觉

### 2.1 今天要理解什么

先不要急着训练模型。今天只回答一个问题：

```text
GPT 到底在学什么？
```

你需要理解：

```text
文本是一串 token
语言模型读入前面的 token
语言模型预测下一个 token
训练时，正确答案就是原文本里的下一个 token
```

例子：

```text
输入：I love deep
目标：learning
```

模型不是“理解整篇文章后再写作”，而是在每一个位置上做分类：

```text
在词表里选出最可能的下一个 token
```

### 2.2 今天要看的资料

按顺序看：

```text
1. The Illustrated Transformer 前半部分
   https://jalammar.github.io/illustrated-transformer/

2. nanoGPT README 的开头和项目介绍
   https://github.com/karpathy/nanoGPT
```

不用看懂所有细节。今天只要形成直觉：

```text
GPT = Transformer decoder only language model
训练 = 给一段文本，预测每个位置的下一个 token
```

### 2.3 今天要记录什么

写到 `notes.md`：

```markdown
## Day 1 Notes

### What is a language model?

### What is next token prediction?

### Questions
```

写到 `../docs/learning_log.md`：

```markdown
## 2026-xx-xx

### Today
- 建立了 language model / next token prediction 的基本直觉。

### Concepts
- token
- next token prediction
- causal language modeling

### Questions
- 

### Next
- 学 Transformer 的结构。
```

### 2.4 今天完成标准

你能不看资料说出：

```text
GPT 的输入是什么？
GPT 的输出是什么？
训练标签从哪里来？
为什么这叫 self-supervised learning？
```

## 3. Day 2：学习 Transformer 核心结构

### 3.1 今天要理解什么

今天重点理解 Transformer block。先不追求公式推导，先理解每个部件的作用。

你要了解：

```text
token embedding
position embedding
self-attention
Q / K / V
multi-head attention
causal mask
MLP / FFN
residual connection
LayerNorm
```

### 3.2 推荐理解顺序

按这个顺序学：

```text
1. token embedding
   token id 变成向量。

2. position embedding
   让模型知道 token 的位置。

3. self-attention
   每个 token 看前面的 token，决定该关注谁。

4. causal mask
   防止当前位置看到未来 token。

5. MLP / FFN
   对每个位置的表示做非线性变换。

6. residual + LayerNorm
   让深层网络更稳定。
```

### 3.3 今天要看的资料

```text
1. The Illustrated Transformer 里 self-attention 部分
2. Attention Is All You Need 的 abstract 和 model architecture 图
3. 可选：Karpathy GPT from scratch 视频前半部分
```

论文不用硬啃全文。现在只看：

```text
Abstract
Introduction
Model Architecture 图
Scaled Dot-Product Attention 图
Multi-Head Attention 图
```

### 3.4 今天要记录什么

写到 `docs/reading_notes/transformer.md`：

```markdown
## Transformer Block

### Token Embedding

### Position Embedding

### Self-Attention

### Causal Mask

### MLP

### Residual + LayerNorm
```

更新 `../docs/concept_checklist.md`，能解释的就打勾。

### 3.5 今天完成标准

你能画出或写出这个流程：

```text
token ids
-> token embedding + position embedding
-> Transformer blocks
-> logits over vocabulary
-> cross entropy loss
```

## 4. Day 3：读 nanoGPT 项目结构

### 4.1 今天要做什么

今天开始接触代码，但先不训练。

进入 Stage 01：

```bash
cd 01_nanogpt_pretrain
```

如果还没有 nanoGPT：

```bash
git clone https://github.com/karpathy/nanoGPT.git
```

进入 nanoGPT：

```bash
cd nanoGPT
```

先看文件结构：

```bash
find . -maxdepth 2 -type f | sort
```

重点文件：

```text
README.md
model.py
train.py
sample.py
config/train_shakespeare_char.py
data/shakespeare/prepare.py
```

### 4.2 每个文件先知道什么

```text
README.md
告诉你项目怎么跑。

model.py
定义 GPT 模型结构。

train.py
训练主程序。

sample.py
加载 checkpoint 并生成文本。

config/train_shakespeare_char.py
tiny Shakespeare 字符级训练配置。

data/shakespeare/prepare.py
下载和处理 tiny Shakespeare 数据。
```

### 4.3 今天不要做什么

不要一上来改代码。今天只读结构。

### 4.4 今天要记录什么

写到 `notes.md`：

```markdown
## nanoGPT File Map

### README.md

### model.py

### train.py

### sample.py

### config/train_shakespeare_char.py

### data/shakespeare/prepare.py
```

### 4.5 今天完成标准

你能回答：

```text
训练入口是哪个文件？
模型定义在哪个文件？
数据准备在哪个文件？
采样入口是哪个文件？
配置文件控制哪些参数？
```

## 5. Day 4：跑 tiny Shakespeare 最小训练

### 5.1 今天目标

今天第一次真正跑模型。目标不是效果，而是跑通流程：

```text
准备数据 -> 训练 -> 保存 checkpoint -> sample 生成
```

### 5.2 环境检查

在项目根目录跑：

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

再次检查：

```bash
python scripts/check_env.py
```

### 5.3 准备数据

进入 nanoGPT：

```bash
cd 01_nanogpt_pretrain/nanoGPT
python data/shakespeare/prepare.py
```

如果你在 Stage 01 目录，可以用脚本：

```bash
./scripts/run_prepare.sh
```

注意：这个脚本要求 `01_nanogpt_pretrain/nanoGPT/` 已经存在。

### 5.4 跑最小训练

本地 CPU smoke test：

```bash
cd 01_nanogpt_pretrain/nanoGPT
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --max_iters=200
```

或者在 `01_nanogpt_pretrain` 目录：

```bash
./scripts/run_train_smoke.sh cpu 200
```

如果你是 Apple Silicon 且 PyTorch MPS 可用：

```bash
./scripts/run_train_smoke.sh mps 1000
```

如果你之后租 4090：

```bash
./scripts/run_train_smoke.sh cuda 2000
```

### 5.5 训练时观察什么

看日志里的：

```text
iter
loss
time
mfu
```

重点不是 mfu，重点是：

```text
loss 是否在下降？
训练是否能稳定跑完？
有没有显存/设备/依赖报错？
```

### 5.6 今天要记录什么

写到 `../reports/experiment_log.md`：

```markdown
## exp-001: nanoGPT tiny Shakespeare smoke test

### Date

### Device

### Command

### Config
- max_iters:
- device:
- compile:

### Result
- final train loss:
- final val loss:

### Problems

### Notes
```

如果报错，不要只截图。记录：

```text
报错全文关键几行
运行命令
你当时所在目录
Python / PyTorch 版本
最后怎么解决
```

### 5.7 今天完成标准

```text
数据 prepare 成功
训练命令跑起来
你能找到 checkpoint 输出目录
experiment_log 有一条记录
```

## 6. Day 5：读 nanoGPT 关键代码

### 6.1 今天目标

训练跑通后，再读代码会更有感觉。今天重点看 `model.py`。

重点类：

```text
GPTConfig
LayerNorm
CausalSelfAttention
MLP
Block
GPT
```

### 6.2 阅读顺序

不要从第一行硬读到最后一行。按这个顺序：

```text
1. GPTConfig
   看模型有哪些超参数。

2. GPT.__init__
   看 embedding、blocks、lm_head 怎么搭起来。

3. GPT.forward
   看输入 idx 如何变成 logits 和 loss。

4. Block.forward
   看 attention 和 MLP 怎么串起来。

5. CausalSelfAttention.forward
   看 Q/K/V、attention、mask、输出。

6. MLP.forward
   看 FFN 结构。
```

### 6.3 你要特别盯住的变量

```text
idx
targets
tok_emb
pos_emb
x
logits
loss
q, k, v
att
```

### 6.4 今天要记录什么

写到 `notes.md`：

```markdown
## model.py Reading

### GPTConfig

### GPT.__init__

### GPT.forward

### Block.forward

### CausalSelfAttention.forward

### MLP.forward

### Variables
- idx:
- targets:
- logits:
- loss:
- q/k/v:
- att:
```

### 6.5 今天完成标准

你能回答：

```text
idx 的 shape 是什么？
targets 是什么？
logits 的 shape 是什么？
loss 在哪里算？
为什么训练时需要 targets，采样时不需要？
attention mask 在哪里生效？
```

## 7. Day 6：采样、观察输出、解释 loss

### 7.1 今天目标

用训练出的 checkpoint 生成文本，并观察模型到底学到了什么。

### 7.2 运行 sample

在 nanoGPT 目录：

```bash
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

或者在 `01_nanogpt_pretrain` 目录：

```bash
./scripts/run_sample.sh cpu out-shakespeare-char
```

如果用 MPS：

```bash
./scripts/run_sample.sh mps out-shakespeare-char
```

如果用 CUDA：

```bash
./scripts/run_sample.sh cuda out-shakespeare-char
```

### 7.3 观察输出

不要只说“好”或“坏”。你要观察：

```text
是否像 Shakespeare 风格？
是否有英文单词结构？
是否有换行、角色名、标点？
是否重复？
是否胡言乱语？
训练 iter 更多后是否改善？
```

### 7.4 可以做一个小对比

建议至少跑两组：

```text
max_iters=200
max_iters=1000
```

比较：

```text
loss 是否更低？
sample 是否更像文本？
训练时间增加多少？
```

### 7.5 今天要记录什么

写到 `results/sample_outputs.md`：

```markdown
## exp-001

### Config
- device:
- max_iters:
- final loss:

### Sample

```text
粘贴 1-2 段生成文本
```

### Observation
- 
```

写到 `../reports/eval_results.md`：

```text
Experiment | Dataset | Device | Train Loss | Val Loss | Notes
```

### 7.6 今天完成标准

```text
能成功 sample
sample_outputs.md 有生成样例
eval_results.md 有 loss 摘要
能解释 loss 下降和输出质量之间的大致关系
```

## 8. Day 7：写阶段总结

### 8.1 今天目标

把 Stage 01 收束成一个可以复盘的阶段成果。

写：

```text
results/stage01_summary.md
```

### 8.2 总结应该包含什么

```markdown
# Stage 01 Summary

## Goal

## What I Did

## Commands

## Key Results

## What I Learned

## Code Understanding

## Failure Cases

## Remaining Questions

## Next Stage
```

### 8.3 你最后需要弄清楚什么

这是 Stage 01 的核心验收问题。你可以把答案写进 `stage01_summary.md`。

#### 1. 数据如何进入模型？

你要能讲：

```text
原始文本 -> tokenizer/字符映射 -> token ids -> batch -> idx/targets
```

#### 2. 模型如何输出预测？

你要能讲：

```text
idx -> token embedding + position embedding -> blocks -> logits
```

#### 3. loss 如何计算？

你要能讲：

```text
logits 和 targets 做 cross entropy
loss 衡量模型预测下一个 token 的错误程度
```

#### 4. causal mask 为什么必要？

你要能讲：

```text
训练时每个位置只能看自己和之前的 token，不能偷看未来 token。
否则 next token prediction 会变成作弊。
```

#### 5. sample.py 如何生成文本？

你要能讲：

```text
给初始 token
模型输出 logits
根据 temperature/top-k 得到概率分布
采样下一个 token
拼回输入
重复生成
```

#### 6. 这个小实验有什么局限？

你要能讲：

```text
数据很小
模型很小
训练时间短
生成只是模仿局部风格
不代表真正理解语言
```

### 8.4 完成标准

```text
notes.md 有概念和代码阅读笔记
experiment_log.md 有至少 1 条训练实验
sample_outputs.md 有生成样例
eval_results.md 有 loss 记录
stage01_summary.md 能回答验收问题
concept_checklist.md 勾选 Stage 01 相关概念
```

## 9. GPU 使用建议

Stage 01 不建议一开始租 GPU。

推荐：

```text
本地 CPU 跑 200 iter smoke test
本地 MPS 跑 1000 iter，如果可用
本地太慢或想体验 GPU，再租 4090 跑 1-3 小时
```

不要为 Stage 01 月租 GPU。

如果租 GPU，本阶段目标只是：

```text
更快跑完 nanoGPT
观察 loss 曲线
生成更像样的 sample
熟悉云 GPU 环境
```

租之前写到 `../docs/gpu_rental_notes.md`：

```text
我要跑哪个命令？
预计跑多久？
输出保存到哪里？
跑完如何关机？
```

跑完写到 `../reports/gpu_usage_log.md`。

## 10. 不要踩的坑

### 10.1 不要先追求看懂所有公式

先跑通，再读代码，再回头看公式。你现在需要的是建立结构感。

### 10.2 不要把 nanoGPT 子仓库提交上去

`.gitignore` 已经忽略：

```text
01_nanogpt_pretrain/nanoGPT/
```

你可以本地 clone nanoGPT，但不要把整个第三方仓库塞进自己的 GitHub。

### 10.3 不要把 checkpoint 提交上去

训练输出、checkpoint、`.pt`、`.pth` 都不要提交。只记录：

```text
命令
配置
loss
生成样例
结论
```

### 10.4 不要只跑不写

Stage 01 的价值不在于生成 Shakespeare，而在于你能解释这个过程。

## 11. 最小完成路线

如果你时间不够，至少完成：

```text
1. 看 The Illustrated Transformer 的 attention 部分
2. clone nanoGPT
3. prepare Shakespeare 数据
4. CPU 跑 200 iter
5. sample 一次
6. 记录 loss 和 sample
7. 写 stage01_summary.md
```

## 12. 标准完成路线

推荐完成：

```text
1. Day 1-2 学语言模型和 Transformer
2. Day 3 读 nanoGPT 文件结构
3. Day 4 跑 200 iter smoke test
4. Day 5 读 model.py
5. Day 6 跑 1000 iter，并 sample 对比
6. Day 7 写总结
```

## 13. 进阶完成路线

如果你还有精力：

```text
1. 对比 char-level 和 GPT-2 BPE tokenizer
2. 改 block_size 看效果
3. 改 n_layer / n_head / n_embd 看 loss 变化
4. 画一张 loss 曲线
5. 用中文小语料跑一个极小模型
```

进阶部分不是必须。先完成标准路线。

