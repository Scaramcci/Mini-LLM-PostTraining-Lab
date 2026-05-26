# Preference Data Examples

记录 Stage 04 构造的 preference（偏好）数据样例，方便快速查阅和复盘数据质量。

---

## 数据统计

| 项目 | 数值 |
|------|------|
| 总样例数（train） | （运行 build_preference_data.py 后填写） |
| 总样例数（val）   | （运行 build_preference_data.py 后填写） |
| 涉及工具数       | 5（calculator, weather, translate, search, unit_convert） |
| rejected 类型数  | 6 |

---

## Rejected 类型说明

| 类型 | 描述 | 设计意图 |
|------|------|----------|
| `non_json` | 直接给文字答案，不是 JSON | 让模型不要绕过工具调用 |
| `wrong_key_name` | JSON 存在但 key 名称错（如 `function`/`params`） | 对齐输出 schema |
| `wrong_tool_name` | tool name 错误 | 提高工具选择准确率 |
| `extra_text_pollution` | 废话 + 正确 JSON 污染 | 去除解释性前缀 |
| `empty_arguments` | arguments 为空 `{}` | 确保参数完整性 |
| `invalid_json` | 不完整 / 无法解析的 JSON | 格式完整性 |

---

## 数据样例

### 样例 1：calculator（正确 vs 非 JSON）

```json
{
  "prompt": [
    {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON tool calls..."},
    {"role": "user",   "content": "Calculate 17 * 23."}
  ],
  "chosen": [
    {"role": "assistant", "content": "{\"tool\": \"calculator\", \"arguments\": {\"expression\": \"17 * 23\"}}"}
  ],
  "rejected": [
    {"role": "assistant", "content": "I will calculate this for you. The answer is approximately 3712."}
  ]
}
```

**分析：**
- chosen：格式正确，tool 正确，arguments 正确
- rejected：直接给了文字答案，没有使用工具，是模型偷懒的典型模式

---

### 样例 2：weather（正确 vs wrong tool name）

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "What is the weather in Tokyo?"}
  ],
  "chosen": [
    {"role": "assistant", "content": "{\"tool\": \"weather\", \"arguments\": {\"city\": \"Tokyo\"}}"}
  ],
  "rejected": [
    {"role": "assistant", "content": "{\"tool\": \"search\", \"arguments\": {\"city\": \"Tokyo\"}}"}
  ]
}
```

**分析：**
- chosen：格式正确，tool 正确
- rejected：tool name 错（search 而不是 weather），但 JSON 格式本身合法

---

### 样例 3：translate（正确 vs extra text pollution）

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "Translate 'Good morning' to French."}
  ],
  "chosen": [
    {"role": "assistant", "content": "{\"tool\": \"translate\", \"arguments\": {\"text\": \"Good morning\"}}"}
  ],
  "rejected": [
    {"role": "assistant", "content": "Sure! I can help with that. Here is the tool call you need:\n{\"tool\": \"translate\", \"arguments\": {\"text\": \"Good morning\"}}\nHope that helps!"}
  ]
}
```

**分析：**
- chosen：简洁，只有 JSON
- rejected：有废话包裹，JSON 本身对，但输出不规范

---

## 数据质量自查

运行校验脚本：

```bash
python scripts/validate_preference_data.py
```

通过后在下面打勾：

- [ ] train 数据格式校验通过（0 errors）
- [ ] val 数据格式校验通过（0 errors）
- [ ] chosen 和 rejected 类型分布合理
- [ ] 没有 chosen == rejected 的样例

---

## 数据改进记录

（在实验过程中，如果发现数据问题，记录在这里）

```text
问题：
改进方法：
改进时间：
```
