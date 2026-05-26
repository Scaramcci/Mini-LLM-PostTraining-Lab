# Reading Notes: Transformer

## Reading List

- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- nanoGPT: https://github.com/karpathy/nanoGPT
- Let's build GPT from scratch: https://www.youtube.com/watch?v=kCc8FmEb1nY

## Notes

### What Is A Language Model?

语言模型的目标是：根据前面的文本，预测后面最可能出现的内容。

在现代大模型里，文本不会直接以“字”或“词”的形式进入模型，而是先被 tokenizer 切成一串 token。模型看到的是 token id，再通过 embedding 变成向量。训练时，它通常做的是 next token prediction：给定前面的 token，预测下一个 token 的概率分布。

举例：

```text
输入：我 今天 想 喝
目标：奶茶 / 咖啡 / 水 / ...
```

模型输出的不是一个固定答案，而是词表中每个 token 的概率。训练时，如果正确答案是“咖啡”，模型就会被鼓励把“咖啡”的概率调高，把其他 token 的概率相对调低。

所以，语言模型本质上可以理解成一个“条件概率模型”：

```text
P(next token | previous tokens)
```

当这个动作反复进行时，模型就可以一个 token 一个 token 地生成完整文本。

### What Is Causal LM?

Causal Language Model，也就是因果语言模型，指的是模型只能根据当前位置之前的 token 来预测当前位置之后的 token，不能偷看未来。

例如句子：

```text
我 今天 想 喝 咖啡
```

如果模型正在预测“喝”后面的 token，它可以看到：

```text
我 今天 想 喝
```

但不能看到后面的“咖啡”。这就是 causal 的意思：信息只能从左到右流动，当前 token 的预测只能依赖过去，不能依赖未来。

这和 GPT 的生成方式完全一致：GPT 每次只生成下一个 token，然后把新生成的 token 接回上下文里，再继续预测下一个 token。

在 Transformer 里，causal LM 主要通过 causal mask 实现。mask 会把当前位置右侧的 token 屏蔽掉，让 self-attention 只能关注当前位置及其左边的内容。

训练目标通常是 cross entropy loss。模型输出词表上的概率分布，真实答案是下一个 token。预测越接近真实 token，loss 越低。

### What Is Attention?

Attention 解决的问题是：当模型处理一个 token 时，它应该重点参考上下文里的哪些 token？

传统的神经网络很难灵活地“按需读取”上下文。Attention 的直觉更像是：每个 token 都会向其他 token 提问，“哪些信息和我现在的理解有关？”然后根据相关程度加权汇总信息。

Self-attention 里有三个核心向量：

- Query: 当前 token 想找什么信息
- Key: 每个 token 能提供什么索引或标签
- Value: 每个 token 真正携带的内容

计算过程可以粗略理解为：

1. 用 Query 和所有 Key 做相似度计算，得到注意力分数。
2. 对分数做缩放和 softmax，变成权重。
3. 用这些权重对 Value 做加权求和，得到当前位置的新表示。

Scaled dot-product attention 的形式是：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

其中 `sqrt(d_k)` 是缩放项。它的作用是防止点积结果过大，导致 softmax 后概率过于尖锐，训练不稳定。

Multi-head attention 则是把注意力拆成多个 head。每个 head 可以学习不同的关系，比如有的 head 关注相邻 token，有的 head 关注语法结构，有的 head 关注长距离依赖。最后把多个 head 的结果拼接起来，再经过线性层融合。

### What Did I Understand From nanoGPT?

nanoGPT 把 GPT 的核心结构压缩成了一个非常适合学习的版本。它没有复杂的工程包装，重点就是把数据、模型、loss、训练循环和采样过程串起来。

我目前可以把 nanoGPT 理解成下面这条链路：

```text
文本 -> tokenizer/字符编码 -> token id -> token embedding + position embedding
-> 多层 Transformer block -> logits -> cross entropy loss / 采样生成
```

几个关键理解：

- 输入 `idx` 的形状通常是 `(batch_size, block_size)`，表示一个 batch 里的多段 token 序列。
- `token_embedding_table(idx)` 把 token id 映射成向量。
- `position_embedding_table(pos)` 给每个位置一个位置信息。
- token embedding 和 position embedding 相加后，进入多个 Transformer block。
- 每个 block 主要由 self-attention、MLP、residual connection 和 LayerNorm 组成。
- 最后线性层输出 `logits`，形状大致是 `(batch_size, block_size, vocab_size)`。
- 训练时用 logits 和真实的下一个 token 计算 cross entropy loss。
- 生成时从最后一个位置的 logits 得到概率分布，再采样出下一个 token。

nanoGPT 最重要的启发是：GPT 不是一个神秘黑盒，它的基本训练任务非常朴素，就是“看前文，猜下一个 token”。复杂能力来自大量数据、足够大的模型、长时间训练，以及 Transformer 结构本身的表达能力。

## Transformer Block

### Token Embedding

Token embedding 是把离散的 token id 转成连续向量的过程。

token id 本身只是一个编号，比如 `15496`，这个数字没有天然的语义距离。模型不能直接从编号大小理解“猫”和“狗”更相近，或者“猫”和“银行”更远。所以需要 embedding table，把每个 token id 映射到一个可训练向量。

可以理解成：

```text
token id -> embedding vector
```

在代码里通常是：

```python
tok_emb = token_embedding_table(idx)
```

训练开始时，这些向量一般是随机初始化的。随着 next token prediction 训练进行，语义相近、用法相近的 token 会逐渐学到相近或可组合的表示。

### Position Embedding

Transformer 的 self-attention 本身不天然知道 token 的顺序。如果只看一组 token 向量，它并不知道“我 爱 你”和“你 爱 我”的顺序差异。因此需要加入位置信息。

Position embedding 的作用是告诉模型：每个 token 出现在序列的第几个位置。

在 nanoGPT 这类实现里，常见做法是学习一个 position embedding table：

```python
pos_emb = position_embedding_table(pos)
x = tok_emb + pos_emb
```

这里的相加很关键：token embedding 表示“这个 token 是什么”，position embedding 表示“这个 token 在哪里”。二者相加后，模型既知道内容，也知道顺序。

除了 learned position embedding，还有 sinusoidal position embedding、RoPE 等方式。GPT 类模型中常见的是 learned position embedding 或 RoPE。

### Self-Attention

Self-attention 是 Transformer block 的核心。它让序列里的每个 token 都能根据上下文重新更新自己的表示。

“self”的意思是 Query、Key、Value 都来自同一段输入序列。也就是说，句子内部的 token 彼此互相看，而不是去看另一段外部输入。

对于每个位置，模型会做三件事：

1. 生成 Query：我现在需要什么信息？
2. 生成 Key：我可以被别人如何匹配？
3. 生成 Value：如果别人关注我，我能提供什么内容？

然后用 Query 和 Key 的相似度决定注意力权重，再用权重加权 Value。

在 causal LM 里，self-attention 不是随便看全句，而是只能看当前位置及其左边的 token。这样模型训练时才和生成时保持一致。

Self-attention 的好处是它能建模长距离依赖。例如：

```text
小明把书放进书包，因为他明天要考试。
```

模型在理解“他”时，可以通过 attention 关注到前面的“小明”。

### Causal Mask

Causal mask 用来防止模型在训练时偷看未来 token。

在训练语言模型时，我们通常会把一整段文本同时送进模型。这样并行效率很高，但也带来一个问题：如果不加 mask，某个位置可能会看到右边的答案。

例如：

```text
我 今天 想 喝 咖啡
```

预测“喝”后面的 token 时，如果模型已经看到了“咖啡”，训练就变成作弊了。这样训练出来的模型在真实生成时会崩，因为生成时未来 token 根本不存在。

Causal mask 通常是一个下三角矩阵。它允许位置 `t` 关注 `0..t` 的 token，禁止关注 `t+1..end` 的 token。

直观形式：

```text
可以看：
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

在 attention 分数里，被 mask 的位置通常会被设置成一个非常小的值，比如 `-inf`，这样经过 softmax 后权重接近 0。

### MLP

MLP 也常被叫做 FFN，Feed-Forward Network。它是 Transformer block 里除了 attention 之外的另一部分。

如果说 self-attention 负责“从上下文中收集信息”，那么 MLP 更像是“对每个位置收集到的信息做进一步加工”。

MLP 通常对每个 token 位置独立计算，不直接混合不同位置的信息。位置之间的信息交换主要发生在 attention 里；MLP 则负责提升非线性表达能力。

典型结构是：

```text
Linear -> activation -> Linear
```

在 GPT 类模型里，隐藏层维度通常会先扩大，比如从 `n_embd` 扩到 `4 * n_embd`，再投影回 `n_embd`。这样模型有更大的中间空间来组合特征。

常见激活函数包括 GELU、ReLU、SwiGLU 等。nanoGPT 里常见的是 GELU。

### Residual + LayerNorm

Residual connection 和 LayerNorm 是让深层 Transformer 稳定训练的关键组件。

Residual connection，也叫残差连接，做法是把模块的输入加到模块输出上：

```text
x = x + attention(x)
x = x + mlp(x)
```

它的直觉是：每一层不必从零开始重写全部表示，只需要学习“在原有表示上补充或修改什么”。这让梯度更容易传播，也让模型能堆得更深。

LayerNorm 的作用是对每个 token 的隐藏向量做归一化，让数值分布更稳定。这样训练时不容易出现激活值过大、梯度不稳定等问题。

GPT 类模型常用 pre-LayerNorm 结构：

```text
x = x + attention(layer_norm(x))
x = x + mlp(layer_norm(x))
```

也就是说，先 LayerNorm，再进入 attention 或 MLP，最后通过 residual connection 加回原来的 `x`。

可以把一个 Transformer block 简化理解为：

```text
输入 x
-> LayerNorm -> Self-Attention -> Residual Add
-> LayerNorm -> MLP -> Residual Add
-> 输出 x
```

这个结构重复堆叠很多层，就形成了 GPT 的主体。

## Training And Generation

### Cross Entropy Loss

Cross entropy loss 衡量的是：模型给真实下一个 token 的概率有多高。

如果真实答案是“咖啡”，模型给“咖啡”的概率越高，loss 越低；给错 token 的概率越高，loss 越高。

训练时，每个位置都可以产生一个 next token prediction。例如：

```text
输入：我 今天 想 喝
目标：今天 想 喝 咖啡
```

模型会在每个位置预测下一个 token，然后把所有位置的 loss 平均起来。

### Train Loss vs Validation Loss

Train loss 是模型在训练集上的 loss，validation loss 是模型在没参与训练的数据上的 loss。

如果 train loss 和 validation loss 都下降，通常说明模型确实在学习。如果 train loss 持续下降，但 validation loss 不降反升，可能说明模型开始过拟合训练数据。

所以 validation loss 更能反映模型对新文本的泛化能力。

### Sampling Temperature

Temperature 控制生成时的随机性。

模型输出 logits 后，通常会经过 softmax 变成概率分布。temperature 会在 softmax 前调整 logits：

```text
logits = logits / temperature
```

temperature 越低，概率分布越尖锐，模型越倾向于选择最可能的 token，生成更稳定但可能更死板。temperature 越高，概率分布越平，低概率 token 更容易被采样出来，生成更有变化但也更容易跑偏。

### Top-k Sampling

Top-k sampling 是生成时的一种筛选策略。它只保留概率最高的 `k` 个 token，再从这 `k` 个 token 里采样。

例如 `top_k = 50`，模型每一步只从概率最高的 50 个 token 中选择下一个 token。这样可以避免模型采样到概率极低、明显不合理的 token，同时又保留一定随机性。

Temperature 和 top-k 经常一起使用：temperature 控制分布形状，top-k 控制候选范围。 
