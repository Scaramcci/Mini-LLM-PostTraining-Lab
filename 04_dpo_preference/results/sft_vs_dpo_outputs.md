# SFT vs DPO 输出对比

> 此文件由 `scripts/compare_sft_dpo.py` 自动生成后手动补充分析。
> 运行前请先完成 Stage 03（SFT）和本阶段 DPO 训练。

---

## 运行方式

```bash
python scripts/compare_sft_dpo.py \
    --sft_model_path /path/to/sft_output \
    --dpo_model_path results/dpo_output \
    --base_model Qwen/Qwen3-1.7B
```

---

## 汇总结果

（运行脚本后填写）

| 模型 | 平均分 |
|------|--------|
| SFT  |        |
| DPO  |        |

**结论：**

> TODO: 填写你的观察和结论。

---

## 详细对比样例

（运行 compare_sft_dpo.py 后，表格会自动生成在这里，或者你可以手动粘贴几条典型样例）

---

## 人工标注补充

（脚本只做自动评分，这里记录你人工查看后的补充判断）

### 典型改善案例

```text
Prompt：
SFT 输出：（有问题的）
DPO 输出：（改善了）
原因分析：
```

### 典型退化案例

```text
Prompt：
SFT 输出：（本来是好的）
DPO 输出：（变差了）
原因分析：
可能改进：
```

---

## 结论与下一步

```text
DPO 是否值得做？为什么？

和 SFT 相比，DPO 的主要改变是：

下一步（Stage 05 GRPO）的启发：
```
