"""
build_preference_data.py
构造 Stage 04 DPO 训练用的 toy preference 数据集。

数据任务：JSON tool-call（延续 Stage 03 的 SFT 任务）
格式：
  {
    "prompt": [
      {"role": "system", "content": "..."},
      {"role": "user",   "content": "..."}
    ],
    "chosen": [
      {"role": "assistant", "content": "...（格式正确的 JSON tool call）"}
    ],
    "rejected": [
      {"role": "assistant", "content": "...（各种错误类型）"}
    ]
  }

运行方式：
  python scripts/build_preference_data.py

输出：
  data/preference_toy_train.jsonl
  data/preference_toy_val.jsonl
"""

import json
import random
import os
from pathlib import Path

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "data"
TRAIN_FILE = OUTPUT_DIR / "preference_toy_train.jsonl"
VAL_FILE   = OUTPUT_DIR / "preference_toy_val.jsonl"

SYSTEM_PROMPT = (
    "You are a helpful assistant that outputs only valid JSON tool calls. "
    "Do not include any explanation or extra text. "
    "Always respond with a single JSON object in the format: "
    '{\"tool\": \"<tool_name>\", \"arguments\": {<key>: <value>}}'
)

RANDOM_SEED  = 42
VAL_RATIO    = 0.1   # 10% 作为 validation

# ─────────────────────────────────────────────
# 工具定义（tool name → 参数模板）
# ─────────────────────────────────────────────
TOOLS = {
    "calculator": {
        "description": "Evaluates a mathematical expression.",
        "arg_key": "expression",
    },
    "weather":  {
        "description": "Gets the current weather for a city.",
        "arg_key": "city",
    },
    "translate": {
        "description": "Translates text from one language to another.",
        "arg_key": "text",
    },
    "search": {
        "description": "Searches the web for a query.",
        "arg_key": "query",
    },
    "unit_convert": {
        "description": "Converts a value from one unit to another.",
        "arg_keys": ["value", "from_unit", "to_unit"],
    },
}

# ─────────────────────────────────────────────
# 原始样例模板（prompt, chosen_arg, tool_name）
# ─────────────────────────────────────────────
RAW_EXAMPLES = [
    # calculator
    ("Calculate 17 * 23.",               "17 * 23",           "calculator"),
    ("What is 144 / 12?",                "144 / 12",          "calculator"),
    ("Compute the square root of 256.",  "sqrt(256)",         "calculator"),
    ("What is 2 to the power of 10?",   "2 ** 10",           "calculator"),
    ("Evaluate 3.14 * 5 * 5.",           "3.14 * 5 * 5",      "calculator"),
    ("What is (100 - 37) * 2?",          "(100 - 37) * 2",    "calculator"),
    ("Calculate 1000 / 8.",              "1000 / 8",          "calculator"),
    ("What is 99 + 1?",                  "99 + 1",            "calculator"),

    # weather
    ("What is the weather in Paris?",       "Paris",      "weather"),
    ("Tell me the weather in Tokyo.",       "Tokyo",      "weather"),
    ("How is the weather in New York?",     "New York",   "weather"),
    ("Get the weather for London.",         "London",     "weather"),
    ("What's the weather like in Sydney?",  "Sydney",     "weather"),
    ("Check weather for Berlin.",           "Berlin",     "weather"),

    # translate
    ('Translate "Hello, world!" to Spanish.',          "Hello, world!",       "translate"),
    ('Translate "Good morning" to French.',            "Good morning",        "translate"),
    ('Translate "Thank you very much" to Japanese.',   "Thank you very much", "translate"),
    ('Translate "I love programming" to German.',      "I love programming",  "translate"),
    ('Translate "Where is the library?" to Chinese.',  "Where is the library?", "translate"),

    # search
    ("Search for the latest news on AI.",            "latest news on AI",           "search"),
    ("Look up information about quantum computing.", "quantum computing",            "search"),
    ("Find recent papers on large language models.", "recent papers large language models", "search"),
    ("Search for Python tutorials for beginners.",   "Python tutorials for beginners", "search"),

    # unit_convert — 特殊处理，跳过此处，在下方单独生成
]

UNIT_CONVERT_EXAMPLES = [
    # (prompt, value, from_unit, to_unit)
    ("Convert 100 kilometers to miles.",   100,  "km",       "miles"),
    ("How many pounds is 70 kilograms?",   70,   "kg",       "pounds"),
    ("Convert 32 degrees Fahrenheit to Celsius.", 32, "Fahrenheit", "Celsius"),
    ("Convert 5 liters to gallons.",        5,   "liters",   "gallons"),
    ("How many inches is 2 meters?",        2,   "meters",   "inches"),
    ("Convert 1000 milliliters to cups.",  1000, "milliliters", "cups"),
]


def make_chosen(tool_name: str, args: dict) -> str:
    """构造 chosen 回答：格式正确的 JSON tool call。"""
    return json.dumps({"tool": tool_name, "arguments": args}, ensure_ascii=False)


def make_rejected_variants(tool_name: str, args: dict, prompt: str, all_tools: list) -> list:
    """
    针对一个 chosen，构造多种 rejected 变体。
    返回 rejected 回答字符串的列表。
    """
    variants = []

    # 类型 1：不是 JSON，直接给文字答案
    variants.append(
        f"I will calculate this for you. The answer is approximately {random.randint(1, 9999)}."
    )

    # 类型 2：格式接近但 key 名称错误
    wrong_key = json.dumps({"function": tool_name, "params": args}, ensure_ascii=False)
    variants.append(wrong_key)

    # 类型 3：tool name 错误（随机选另一个工具）
    wrong_tools = [t for t in all_tools if t != tool_name]
    if wrong_tools:
        wrong_tool = random.choice(wrong_tools)
        variants.append(
            json.dumps({"tool": wrong_tool, "arguments": args}, ensure_ascii=False)
        )

    # 类型 4：废话 + 正确 JSON（extra text pollution）
    variants.append(
        f"Sure! I can help with that. Here is the tool call you need:\n"
        + json.dumps({"tool": tool_name, "arguments": args}, ensure_ascii=False)
        + "\nHope that helps!"
    )

    # 类型 5：arguments 内容错误（空 arguments）
    variants.append(
        json.dumps({"tool": tool_name, "arguments": {}}, ensure_ascii=False)
    )

    # 类型 6：无法解析的半截 JSON
    variants.append('{"tool": "' + tool_name + '", "arguments": {')

    return variants


def build_example(prompt: str, chosen_content: str, rejected_content: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "chosen": [
            {"role": "assistant", "content": chosen_content},
        ],
        "rejected": [
            {"role": "assistant", "content": rejected_content},
        ],
    }


def main():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tools = list(TOOLS.keys())
    all_examples = []

    # ── 普通工具样例 ──
    for prompt, arg_val, tool_name in RAW_EXAMPLES:
        tool_info = TOOLS[tool_name]
        args = {tool_info["arg_key"]: arg_val}
        chosen_content = make_chosen(tool_name, args)
        rejected_variants = make_rejected_variants(tool_name, args, prompt, all_tools)
        for rej in rejected_variants:
            all_examples.append(build_example(prompt, chosen_content, rej))

    # ── unit_convert 样例 ──
    for prompt, value, from_unit, to_unit in UNIT_CONVERT_EXAMPLES:
        tool_name = "unit_convert"
        args = {"value": value, "from_unit": from_unit, "to_unit": to_unit}
        chosen_content = make_chosen(tool_name, args)
        rejected_variants = make_rejected_variants(tool_name, args, prompt, all_tools)
        for rej in rejected_variants:
            all_examples.append(build_example(prompt, chosen_content, rej))

    # ── 打乱并分 train/val ──
    random.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * VAL_RATIO))
    val_examples   = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    # ── 写出 ──
    def write_jsonl(path: Path, data: list):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  ✓ {len(data):>4} examples → {path}")

    print("\n=== build_preference_data.py ===")
    print(f"Total examples: {len(all_examples)}")
    write_jsonl(TRAIN_FILE, train_examples)
    write_jsonl(VAL_FILE,   val_examples)
    print("Done.\n")

    # ── 快速预览 ──
    print("─── Sample (first 2 examples from train) ───")
    for ex in train_examples[:2]:
        print(json.dumps(ex, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()
