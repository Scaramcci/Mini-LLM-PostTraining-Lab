# Stage 04 总结：DPO 偏好对齐

> 完成 Stage 04 后填写此文件。
> 这是你这个阶段的核心交付之一，应该能独立讲清楚每一步。

---

## 目标

- 理解 DPO 的核心机制
- 构造 toy preference 数据（chosen / rejected 对）
- 跑通 TRL DPOTrainer
- 量化 SFT → DPO 的效果变化
- 记录失败样例，理解 DPO 的局限

---

## 我做了什么

（完成后填写）

```text
1. 
2. 
3. 
4. 
5. 
```

---

## 关键命令

（记录你最终使用的关键命令）

```bash
# 构造偏好数据
python scripts/build_preference_data.py

# 校验数据格式
python scripts/validate_preference_data.py

# DPO 训练（0.6B smoke test）
python scripts/train_dpo.py --smoke_test

# DPO 训练（1.7B 正式）
python scripts/train_dpo.py \
    --model_name Qwen/Qwen3-1.7B \
    --output_dir results/dpo_qwen3_1_7b \
    --num_train_epochs 3 \
    --bf16

# 对比 SFT vs DPO
python scripts/compare_sft_dpo.py \
    --sft_model_path /path/to/sft_output \
    --dpo_model_path results/dpo_output \
    --base_model Qwen/Qwen3-1.7B
```

---

## 关键结果

### 数据集

| 项目 | 数值 |
|------|------|
| train 样例数 | |
| val 样例数   | |
| rejected 类型数 | 6 |

### 训练结果

| 实验 | 模型 | beta | 最终 loss | 备注 |
|------|------|------|-----------|------|
| dpo-001 | Qwen3-0.6B | | | smoke test |
| dpo-002 | Qwen3-1.7B | | | 正式 |

### 评测结果（SFT vs DPO）

| 评估项 | SFT | DPO | 结论 |
|--------|-----|-----|------|
| 平均分         | | | |
| is_valid_json  | | | |
| tool_correct   | | | |
| has_arguments  | | | |

---

## 我学到了什么

### DPO 的核心机制

（用自己的话写，不超过 200 字）

```text
TODO
```

### DPO 和 SFT 的本质区别

```text
SFT 是：

DPO 是：

关键区别：
```

### beta 的作用

```text
TODO: 你在实验中感受到的 beta 大小的影响？
```

### reference model 的必要性

```text
TODO: 为什么不能直接做 MLE（最大似然）？reference model 解决了什么问题？
```

---

## 失败案例

（记录 DPO 后变差的样例，这些往往比成功案例更有学习价值）

### 案例 1

```text
Prompt：
SFT 输出（OK）：
DPO 输出（变差）：
我的猜测：
```

---

## 这个实验的局限

```text
数据方面：

模型方面：

训练方面：
```

---

## 下一步（Stage 05 预告）

Stage 05 将做 GRPO（Group Relative Policy Optimization）：

```text
和 DPO 的区别：
- DPO：给定 chosen/rejected，直接优化
- GRPO：模型自己生成多条 rollout，用 reward function 打分，再优化

Stage 05 重点：
1. 设计可验证的 reward function（不用 LLM-as-Judge）
2. 跑通 TRL GRPOTrainer
3. 对比 SFT / DPO / GRPO 的输出
```

---

## 概念验收

完成本阶段后，你应该能不看资料回答：

- [ ] DPO 的训练数据格式是什么？
- [ ] policy model 和 reference model 分别是什么？它们有什么区别？
- [ ] beta 参数控制什么？增大 beta 会发生什么？
- [ ] DPO 为什么不需要单独的 Reward Model？
- [ ] SFT 完成后为什么还要做 DPO？DPO 解决了 SFT 的什么问题？
- [ ] DPO 之后你观察到了哪些改变？哪些没变？
