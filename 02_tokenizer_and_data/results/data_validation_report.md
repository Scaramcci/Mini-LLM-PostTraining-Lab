# Data Validation Report

## sft_toy_10.jsonl

### Command

```bash
python 02_tokenizer_and_data/scripts/validate_chat_format.py 02_tokenizer_and_data/data_examples/sft_toy_10.jsonl
```

### Result

待补充。

### Problems

待补充。

### Fixes

待补充。

## Train / Validation Split

### Command

```bash
python 02_tokenizer_and_data/scripts/split_dataset.py \
  02_tokenizer_and_data/data_examples/sft_toy_10.jsonl \
  --train-out 02_tokenizer_and_data/data_examples/sft_toy_train.jsonl \
  --val-out 02_tokenizer_and_data/data_examples/sft_toy_val.jsonl \
  --val-ratio 0.2
```

### Result

待补充。
