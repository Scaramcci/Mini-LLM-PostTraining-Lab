# Stage 01 Notes

## Goal

从 nanoGPT 入门 Transformer 和 causal language modeling。

## Reading Notes

## model.py Reading

### GPTConfig

`GPTConfig` 是 GPT 模型的配置表，使用 `@dataclass` 定义。它不负责计算，只负责集中保存模型结构和训练相关的超参数。

主要字段：

- `block_size`: 模型一次最多能看的上下文长度，也就是最大 token 序列长度。
- `vocab_size`: 词表大小，决定模型最后要在多少个 token 里预测下一个 token。
- `n_layer`: Transformer Block 的层数。
- `n_head`: self-attention 的 head 数量。
- `n_embd`: 每个 token embedding 的向量维度，也是 Transformer 内部隐藏状态维度。
- `dropout`: dropout 概率，用于正则化，减少过拟合。
- `bias`: 是否在线性层和 LayerNorm 中使用 bias。

可以把 `GPTConfig` 理解成模型的“设计图纸”。后面的 `GPT.__init__` 会根据这张图纸真正搭建网络。

### GPT.__init__

`GPT.__init__` 负责搭建完整 GPT 模型。它接收一个 `config`，然后根据配置创建 token embedding、position embedding、多层 Transformer Block、最终 LayerNorm 和语言模型输出头。

核心结构在 `self.transformer` 里：

- `wte`: token embedding，把 token id 变成向量。
- `wpe`: position embedding，给每个位置一个可学习的位置向量。
- `drop`: embedding 相加后的 dropout。
- `h`: `n_layer` 个 `Block` 组成的列表，是模型主体。
- `ln_f`: 所有 Block 之后的最终 LayerNorm。

`self.lm_head` 是最后的语言模型头，把隐藏向量从 `n_embd` 维映射到 `vocab_size` 维。输出的每一维对应词表里一个 token 的分数。

这里还有一个重要技巧：weight tying。

```python
self.transformer.wte.weight = self.lm_head.weight
```

这表示输入 token embedding 和输出 lm_head 共用同一套权重。直觉上，模型输入时需要理解每个 token 的向量表示，输出时也需要用 token 向量来判断下一个 token 是谁。共享权重可以减少参数量，也常常能提升语言模型效果。

初始化部分做了两件事：

- `self.apply(self._init_weights)`: 初始化 Linear 和 Embedding 权重。
- 对所有 `c_proj.weight` 做特殊缩放初始化，参考 GPT-2 的做法，让深层 residual 结构训练更稳定。

### GPT.forward

`GPT.forward(idx, targets=None)` 是完整模型的前向传播。

输入：

- `idx`: token id 矩阵，形状是 `(batch_size, sequence_length)`。
- `targets`: 训练目标，也是 token id 矩阵，形状通常和 `idx` 一样。

主要步骤：

1. 读取 `idx` 的形状，得到 batch 大小 `b` 和序列长度 `t`。
2. 检查 `t <= block_size`，防止输入超过模型最大上下文长度。
3. 创建位置索引 `pos = [0, 1, ..., t-1]`。
4. 用 `wte(idx)` 得到 token embedding，形状是 `(b, t, n_embd)`。
5. 用 `wpe(pos)` 得到 position embedding，形状是 `(t, n_embd)`。
6. 把 token embedding 和 position embedding 相加，再做 dropout。
7. 依次通过所有 Transformer Block。
8. 经过最终 LayerNorm。
9. 如果传入 `targets`，计算所有位置的 logits 和 cross entropy loss。
10. 如果没有传入 `targets`，说明是推理生成，只计算最后一个位置的 logits。

训练时：

```python
logits = self.lm_head(x)
loss = F.cross_entropy(...)
```

推理时：

```python
logits = self.lm_head(x[:, [-1], :])
loss = None
```

推理阶段只取最后一个位置，是因为生成下一个 token 时，只需要当前上下文最后位置的预测结果。

### Block.forward

`Block.forward(x)` 是一个 Transformer Block 的前向传播。

代码非常短：

```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
return x
```

它包含两段结构：

1. `LayerNorm -> CausalSelfAttention -> Residual Add`
2. `LayerNorm -> MLP -> Residual Add`

这里使用的是 pre-LayerNorm，也就是先归一化，再送进 attention 或 MLP。残差连接 `x + ...` 的作用是保留原始信息，让每一层只需要学习“对当前表示做什么补充或修改”，从而让深层网络更容易训练。

可以把 `Block` 理解成 GPT 主体里反复堆叠的基本单元。

### CausalSelfAttention.forward

`CausalSelfAttention.forward(x)` 负责计算因果自注意力。

输入 `x` 的形状是：

```text
(B, T, C)
```

含义是：

- `B`: batch size
- `T`: sequence length
- `C`: embedding dimension，也就是 `n_embd`

第一步是通过一个线性层同时算出 q/k/v：

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```

`self.c_attn(x)` 的最后一维是 `3 * n_embd`，然后被切成三份，分别是 query、key、value。

接着把 q/k/v reshape 成多头注意力需要的形状：

```text
(B, nh, T, hs)
```

其中：

- `nh`: number of heads
- `hs`: head size，也就是 `C // n_head`

如果当前 PyTorch 支持 Flash Attention，就直接调用：

```python
scaled_dot_product_attention(..., is_causal=True)
```

这里的 `is_causal=True` 会自动处理 causal mask。

如果不支持 Flash Attention，就走手写版本：

```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
y = att @ v
```

这几行对应 scaled dot-product attention：

```text
softmax(QK^T / sqrt(d_k))V
```

其中 causal mask 会把未来位置填成 `-inf`，softmax 后这些位置的权重就变成 0，保证当前位置不能看到未来 token。

最后把多个 head 的结果拼回 `(B, T, C)`，再经过输出投影 `c_proj` 和 dropout。

### MLP.forward

`MLP.forward(x)` 是 Transformer Block 里的前馈网络。

流程是：

```python
x = self.c_fc(x)
x = self.gelu(x)
x = self.c_proj(x)
x = self.dropout(x)
return x
```

`c_fc` 会把维度从 `n_embd` 扩大到 `4 * n_embd`，`GELU` 提供非线性变换，`c_proj` 再把维度投回 `n_embd`。

Attention 负责让不同 token 之间交换信息；MLP 则对每个 token 当前位置的表示做更深的非线性加工。它不直接混合不同位置的信息，但会增强每个位置内部的特征表达能力。

### Variables

- `idx`: 输入 token id，形状通常是 `(batch_size, block_size)` 或 `(B, T)`。它是模型真正读进去的文本数字化结果。
- `targets`: 训练标签，形状通常和 `idx` 一样。`targets` 中每个位置是对应输入位置要预测的“下一个 token”。如果为 `None`，表示当前是推理生成，不计算 loss。
- `logits`: 模型输出的未归一化分数。训练时形状通常是 `(B, T, vocab_size)`；推理时只保留最后一个位置，形状是 `(B, 1, vocab_size)`。logits 经过 softmax 后可以变成 token 概率分布。
- `loss`: 训练时的 cross entropy loss，用来衡量模型预测和真实 `targets` 的差距。推理时没有 `targets`，所以 `loss = None`。
- `q/k/v`: self-attention 中的 query、key、value。`q` 表示当前位置想找什么信息，`k` 表示每个位置能被怎样匹配，`v` 表示每个位置真正提供的内容。多头拆分后形状是 `(B, n_head, T, head_size)`。
- `att`: attention 权重矩阵。手写 attention 路径中，`att` 的形状是 `(B, n_head, T, T)`，表示每个 head 里，每个 token 对上下文各位置的关注权重。causal mask 会让未来位置的权重变成 0。

## nanoGPT File Map

### README.md
nanoGPT 项目的官方说明文档。介绍了项目的初衷（最简单、最容易 hack 的纯 PyTorch GPT 开发库），以及如何准备数据、进行训练和基于提示词推理生成的命令行指南。

### model.py
GPT 模型的核心脑部架构定义（极简地仅集中在这一份文件中）。重点模块有：
- **`GPT`**: 整个语言模型的顶层类，包含词嵌入 (wte)、位置嵌入 (wpe)、多层 Transformer Block 堆叠结构以及预测概率的输出头。
- **`Block`**: 组成骨干网络的单个 Transformer 块，封装了层归一化、自注意力和 MLP。
- **`CausalSelfAttention`**: 因果自注意力层。利用序列掩码（causal mask）约束，确保当前词只能与过去的词发生交互，防止模型作弊看到“未来”的 token。
- **`MLP`**: 多层感知机，放置在注意力计算后，对提炼出的特征进行深度非线性变换。

### train.py
模型的持续训练脚本。它主导了模型“学习”的全过程：
- 配置训练环境（支持单卡、或如 DDP 这样的分布式多机多卡）。
- 使用 `get_batch` 从预处理好的数据中不断抓取批次样本。
- 组合并初始化 `AdamW` 优化器和带有 Warmup 热身的余弦退火学习率策略。
- 进行大循环训练：执行前向传播预测、比对真实标签计算 Loss，然后反向传播产生梯度，并不断修正更新网络权重；且将中间表现优秀的权重存成文件 (ckpt)。

### sample.py
模型参数固定后的推理采样 / 生成生成脚本。
- 读取由 `train.py` 保存的模型检查点文件（`ckpt.pt`）并恢复其状态。
- 对输入的字符提示（prompt）通过 `model.generate` 进行自回归的（autoregressive）生成：读入过去的词，预测最高概率的下一个词，塞回输入并循环。
- 可搭配设定 `temperature`（温度，越高生成越有发散力/创造性）和 `top_k`（截断阈值，避免选到极不可能的错词）等超参数控制生成结果的风格。

### config/train_shakespeare_char.py
用于做模型小型化微调或测试实验的配置脚本。
- `train.py` 中的默认配置通常是针对在庞大的 OpenWebText 上练上亿参数的大模型（对算力要求极高）。该文件正是用于覆盖前者的默认参数使用。
- 文件中搭建了一个微缩小模型：层数骤降到 `n_layer=6`，输入上下文仅为 `block_size=256`，使得其完全可以在 MacBook 等常规算力设备上用极短时间就跑通训练流程。

### data/shakespeare/prepare.py
将原始语料变成模型可消化数字的数据清理工。
- 发送请求从网络下载了开源的莎士比亚剧本小型数据集（Tiny Shakespeare）。
- 按照 90% (train) 与 10% (val) 切分布局，把数据集分割成用以训练和测验的两块。
- 利用分词编码器（如 GPT-2 的 `tiktoken` BPE 或基于单个字母字符的映射字典），将一长串人类文字彻底转码离散化成一个个代表标识（Token）的整型 ID (如 `[12, 45, ... ]`)。
- 最终高效写成紧凑的二进制序列 `train.bin` 与 `val.bin` 文件供 `train.py` 中的张量极速吞吐读取。

## Concept Notes

### Causal Language Modeling

因果语言建模 (Causal Language Modeling, CLM) 是一种自回归训练核心任务。在 CLM 中，模型的目标是只能根据序列中已经出现的词（左侧的历史上下文）来预测紧接着的下一个词（Next Token）。之所以称为“因果”，是因为它严格遵循时间序列的前后因果方向——当前时刻 $t$ 的输出只能由 $t$ 和 $t$ 之前的输入决定，绝不允许“偷看”未来的内容。我们熟知的 GPT (Generative Pre-trained Transformer) 就是在这个海量无监督文本预测下一个词任务上一路喂大的。

### Attention Mask

注意力掩码 (Attention Mask) 是在自注意力机制结算时用于主动“遮蔽”特定位置信息的一种矩阵手段。在 GPT 这类因果语言模型中，主要运用的是一种名为**下三角矩阵**的因果掩码 (Causal Mask)。
掩码的操作逻辑是：计算 Attention 分数时，在矩阵运算级别将当前 token 之后的“未来的”所有待预测位置填充为极小的值（如负无穷 `-inf`）。这样当经过 `Softmax` 激活函数转化为概率权重时，这些保留为负无穷的未来位置权重就会平滑地变为 `0`。这在数学计算层面硬性保障了模型在做文本生成预训练时无法看参考答案“作弊”。

### Train Loss vs Validation Loss

在模型训练中，Loss（损失分数）用于量化度量模型输出预测的概率与其实际真实结果文本差距的大小：
- **Train Loss (训练损失)**：模型在不断被反复阅读和梯度更新的数据集（训练集）上的预测误差。随着训练时间增加，它通常会稳步持续下降。
- **Validation Loss (验证损失)**：模型在一个全新、从未参与过学习的小型数据集（即验证集，如 `val.bin`）上的预测误差。它是衡量模型**泛化能力**的最核心数据，代表了模型如果在野生全新数据环境下的真实水准。
- **二者之间的跷跷板现象**：如果观察到 Train Loss 不断下降，但随后 Validation Loss 下降缓慢不仅停住，甚至反而开始上升了，这就是学术意义上的发生了**过拟合 (Overfitting)**——表示这颗模型的“大脑结构”正在像应试教育一样“死记硬背”训练集里面的细节（能背下来题），而丧失了举一反三的通用推导规律。碰到这情况，我们一般就需要引入 Dropout 丢弃率或者早停等正则化手段。

## Questions

## Next

- 先跑通 tiny Shakespeare。


## Day 1 Notes

### What is a language model?

大语言模型是通过预测 某个位置的token最大可能性 来实现输出的概率模型

### What is next token prediction?

next token prediction 是给出一串 token，来预测下一个 token 是什么的概率模型

### Questions

Jieba 分词和别的分词区别在哪里
