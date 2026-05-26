# Stage 01 Summary

## Goal

本阶段目标是跑通 nanoGPT 在 tiny Shakespeare 字符级数据集上的最小预训练流程，并理解 GPT 从 token 输入、Transformer 前向传播、loss 计算到文本生成的完整链路。

这一阶段不追求训练出高质量模型，重点是建立下面这些直觉：

- 文本如何变成 token id。
- token id 如何进入 GPT。
- Transformer block 如何处理上下文。
- causal language model 如何通过 next token prediction 学习。
- train loss / validation loss 如何反映模型训练状态。
- 训练好的 checkpoint 如何用于 sample 生成。

## What I Did

- 阅读并整理了 nanoGPT 的核心文件结构，包括 `README.md`、`model.py`、`train.py`、`sample.py` 和数据准备脚本。
- 学习了 causal language modeling、attention mask、train loss vs validation loss 等核心概念。
- 重点阅读了 `model.py` 中的 `GPTConfig`、`GPT`、`Block`、`CausalSelfAttention` 和 `MLP`。
- 使用 tiny Shakespeare 字符级数据集在本地 MacBook 的 MPS backend 上训练了一个 baby GPT。
- 将训练迭代数设置为 `1000`，观察 loss 从随机初始化状态逐步下降。
- 保存了 checkpoint，并记录了模型生成样例。

## Commands

训练命令：

```bash
./scripts/run_train_smoke.sh mps 1000
```

该脚本实际进入 `01_nanogpt_pretrain/nanoGPT` 后执行：

```bash
python train.py config/train_shakespeare_char.py --device=mps --compile=False --max_iters=1000
```

关键配置：

- dataset: `shakespeare_char`
- device: `mps`
- compile: `False`
- max_iters: `1000`
- batch_size: `64`
- block_size: `256`
- n_layer: `6`
- n_head: `6`
- n_embd: `384`
- dropout: `0.2`
- learning_rate: `1e-3`
- vocab_size: `65`
- parameters: `10.65M`
- tokens per iteration: `16,384`

## Key Results

训练过程中的关键 loss：

| Step | Train Loss | Val Loss |
| --- | ---: | ---: |
| 0 | 4.2874 | 4.2823 |
| 250 | 1.9638 | 2.0676 |
| 500 | 1.5304 | 1.7323 |
| 750 | 1.3666 | 1.5910 |
| 1000 | 1.2750 | 1.5254 |

主要观察：

- train loss 从 `4.2874` 降到 `1.2750`，说明模型确实在从数据里学习字符级模式。
- validation loss 从 `4.2823` 降到 `1.5254`，说明模型不只是记住训练 batch，也对验证集有一定泛化。
- 每次 eval 和 checkpoint 保存时耗时约 `50s+`，普通训练迭代大约 `440ms-460ms`。
- checkpoint 在 step `250`、`500`、`750`、`1000` 保存，因为 validation loss 持续改善。
- 生成样例已经能出现角色名、换行、类似莎士比亚戏剧的格式，但仍有大量拼写错误、生造词和语义不连贯。

## What I Learned

- GPT 的训练任务非常朴素：给定前面的 token，预测下一个 token。
- tiny Shakespeare 字符级模型的词表很小，只有 `65` 个字符/token，所以模型是在字符层面学习拼写、换行、角色名和文本风格。
- loss 下降通常会带来更像训练语料的输出，但 loss 低不等于文本已经真正通顺。
- train loss 和 validation loss 都下降时，说明训练是有效的；如果之后 train loss 继续下降而 val loss 上升，就可能开始过拟合。
- MPS 能跑通这个小模型，但 eval 阶段比较慢，训练日志里的长耗时主要出现在评估和保存 checkpoint 附近。
- `compile=False` 对 Mac/MPS smoke test 更稳，适合先跑通流程。

## Code Understanding

`GPTConfig` 是模型配置表，决定上下文长度、词表大小、层数、head 数、embedding 维度和 dropout。

`GPT.__init__` 搭建完整模型：token embedding、position embedding、多层 Transformer Block、最终 LayerNorm 和 `lm_head`。其中 `wte.weight = lm_head.weight` 使用了 weight tying，让输入 embedding 和输出分类头共享参数。

`GPT.forward` 是完整前向传播：`idx` 进入 embedding，叠加 position embedding，通过多个 Block，最后得到 `logits`。训练时用 `targets` 计算 cross entropy loss；推理时只取最后一个位置的 logits 来预测下一个 token。

`Block.forward` 是 GPT 的基本堆叠单元，结构是：

```text
LayerNorm -> CausalSelfAttention -> Residual
LayerNorm -> MLP -> Residual
```

`CausalSelfAttention.forward` 先把输入映射成 `q/k/v`，再做 scaled dot-product attention。causal mask 保证每个位置只能看自己和左边的 token，不能偷看未来。

`MLP.forward` 对每个位置的隐藏表示做非线性变换：先从 `n_embd` 扩到 `4 * n_embd`，经过 GELU，再投回 `n_embd`。

关键变量：

- `idx`: 输入 token id，形状是 `(B, T)`。
- `targets`: 下一个 token 标签，形状通常也是 `(B, T)`。
- `logits`: 模型对词表中每个 token 的预测分数，训练时形状是 `(B, T, vocab_size)`。
- `loss`: logits 和 targets 之间的 cross entropy。
- `q/k/v`: attention 中的 query、key、value。
- `att`: 每个 token 对上下文位置的注意力权重。

## Failure Cases

- 训练过程中出现 `torch.cuda.amp.GradScaler` 的 warning。原因是代码路径使用了 CUDA AMP 的 GradScaler，但当前机器没有 CUDA，PyTorch 自动禁用了它。这次训练仍然正常完成。
- MPS 后端的 MFU 显示很低，大约 `0.7%-0.8%`，这个指标主要是 CUDA/GPU 利用率语境下的参考值，在 Mac MPS 上不必过度解读。
- eval interval 设为 `250`，每次评估要跑 `eval_iters=200`，所以 step `0/250/500/750/1000` 附近耗时很长。
- 当前只训练了 `1000` iter，生成文本仍然有明显问题：拼写错误、生造词、句子结构混乱、长期语义不稳定。
- 这是 character-level 模型，不是 BPE tokenizer 模型，所以它学的是字符序列规律，和真实大语言模型的 tokenization 还有距离。

## Remaining Questions

- 如果继续训练到 `5000` iter，validation loss 会继续下降还是开始过拟合？
- 改变 `temperature` 和 `top_k` 后，生成质量会如何变化？
- character-level tokenizer 和 BPE tokenizer 在训练效率、上下文长度利用率、生成质量上差别有多大？
- 为什么 MPS 上 eval 比普通 iteration 慢这么多？是否主要由 `eval_iters=200` 导致？
- `q/k/v` 在训练后是否能可视化出某些 attention pattern？
- `dropout=0.2` 对这个小模型是否合适？如果设为 `0.0` 会不会更快过拟合？

## Next Stage

Stage 02: tokenizer and data.

下一阶段重点：

- 理解 character-level tokenizer、BPE tokenizer 和 GPT-2 tokenizer 的区别。
- 阅读 `data/shakespeare/prepare.py`，弄清楚原始文本如何变成 `train.bin` 和 `val.bin`。
- 尝试记录不同采样参数下的生成结果。
- 对比训练前、训练中、训练后的 sample 质量变化。
- 开始理解 tokenizer 如何影响上下文长度、训练效率和模型能力。
