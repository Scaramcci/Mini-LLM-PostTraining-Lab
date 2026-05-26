# Stage 01 Notes

## Goal

从 nanoGPT 入门 Transformer 和 causal language modeling。

## Reading Notes

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
