# Stage 01: nanoGPT Pretraining

本阶段目标：从零训练一个很小的 GPT，理解 Transformer 和 causal language modeling 的基本流程。

## What To Learn

- token embedding
- position embedding
- causal self-attention
- causal mask
- multi-head attention
- residual connection
- LayerNorm
- MLP / FFN
- cross entropy loss
- next token prediction
- train loss / validation loss
- sampling

## Recommended Resources

- nanoGPT: https://github.com/karpathy/nanoGPT
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Let's build GPT from scratch: https://www.youtube.com/watch?v=kCc8FmEb1nY

## Dataset

第一轮使用 nanoGPT 自带 tiny Shakespeare 数据。

你不需要手动下载大型数据集。nanoGPT 的 `data/shakespeare/prepare.py` 会处理数据。

## Suggested Workflow

```bash
# 1. 进入本目录
cd 01_nanogpt_pretrain

# 2. 获取 nanoGPT
git clone https://github.com/karpathy/nanoGPT.git

# 3. 准备 Shakespeare 数据
cd nanoGPT
python data/shakespeare/prepare.py

# 4. 运行最小训练
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --max_iters=200

# 5. 采样
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

如果你使用 Mac 且 PyTorch 支持 MPS，可以尝试：

```bash
python train.py config/train_shakespeare_char.py --device=mps --compile=False --max_iters=1000
```

如果本地太慢，再按小时租 4090 跑 1-3 小时。

## Outputs To Record

- loss 曲线或训练日志摘要
- 生成样例
- 你对 GPT block 的理解
- 失败和报错

写到：

- `notes.md`
- `results/sample_outputs.md`
- `results/stage01_summary.md`
- `../reports/experiment_log.md`

## Done Criteria

- [ ] 跑通 tiny Shakespeare 数据准备
- [ ] 跑通一次最小训练
- [ ] 成功 sample 生成文本
- [ ] 能解释 causal LM
- [ ] 能解释 attention mask
- [ ] 能说出 train loss 和 val loss 的区别
- [ ] 完成 `results/stage01_summary.md`
