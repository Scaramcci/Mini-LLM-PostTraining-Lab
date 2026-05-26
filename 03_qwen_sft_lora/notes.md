# Stage 03 学习笔记

## SFT (Supervised Fine-Tuning)

### 什么是 SFT？

### SFT 和 pretraining 的区别

### SFT 改变了模型什么行为？

## LoRA

### 什么是 LoRA？

### LoRA 为什么省显存？

### LoRA 的 rank / alpha 参数含义

### LoRA 应用在哪些层？

## QLoRA

### 什么是 QLoRA？

### QLoRA 和 LoRA 的区别

## TRL SFTTrainer

### SFTTrainer 基本用法

### 关键参数

- `per_device_train_batch_size`:
- `gradient_accumulation_steps`:
- `learning_rate`:
- `num_train_epochs`:
- `max_seq_length`:
- `logging_steps`:
- `save_steps`:

## 训练观察

### 训练 loss 变化

### 过拟合判断

### base vs SFT 输出对比

## 代码阅读笔记

### train_sft.py

### 配置文件

### adapter 保存和加载

## Questions

-
