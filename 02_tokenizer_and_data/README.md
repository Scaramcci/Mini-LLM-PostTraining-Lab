# Stage 02: Tokenizer And Data

本阶段目标：理解文本如何变成 token，理解 chat template 和 SFT 数据格式，并为 Stage 03 的 Qwen LoRA SFT 准备一份小型、可校验的数据集。

## Stage Plan

详细学习顺序、脚本使用方式、记录要求和验收问题见：

```text
stage02_plan.md
```

## What To Learn

- token / token id
- BPE / SentencePiece
- encode / decode
- special tokens
- chat template
- `input_ids`
- `attention_mask`
- `labels`
- padding
- truncation
- train / validation split
- SFT data schema

## Recommended Resources

- Hugging Face Tokenizers: https://huggingface.co/docs/tokenizers/
- Transformers chat templates: https://huggingface.co/docs/transformers/chat_templating
- Qwen3 docs: https://huggingface.co/docs/transformers/model_doc/qwen3
- Alpaca dataset: https://huggingface.co/datasets/tatsu-lab/alpaca
- GSM8K dataset: https://huggingface.co/datasets/openai/gsm8k

## No GPU Needed

Stage 02 不训练模型，不需要租 GPU。你只需要本地 Python 环境和 `transformers`。

第一次运行 Qwen tokenizer 会从 Hugging Face 下载 tokenizer 文件，通常很小，不是模型权重。

## Outputs To Record

- tokenizer encode/decode 示例
- chat template 展开后的文本
- toy SFT 数据格式说明
- 数据校验结果
- train / validation split 结果

写到：

- `notes.md`
- `results/tokenizer_examples.md`
- `results/data_validation_report.md`
- `results/stage02_summary.md`
- `../reports/experiment_log.md`

## Done Criteria

- [ ] 能解释 token 和 token id
- [ ] 能解释 encode / decode
- [ ] 能解释 BPE 的基本思想
- [ ] 能用 Qwen tokenizer 处理普通文本
- [ ] 能用 `apply_chat_template` 处理 messages
- [ ] 有一份 toy SFT JSONL
- [ ] 数据校验脚本能跑通
- [ ] 数据能切分成 train / validation
- [ ] 完成 `results/stage02_summary.md`
