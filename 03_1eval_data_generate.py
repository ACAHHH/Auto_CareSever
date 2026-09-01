import os, json
from collections import defaultdict
from pathlib import Path

SFT_VALIDATION = r"D:\Program Files\PythonProject_dl&llm\LLM\Auto_CareSever\data\sft_validation.jsonl"
OUTPUT_DIR = "./custom_eval/text/qa1"
SUBSET_NAME = "auto_service"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{SUBSET_NAME}.jsonl")

_ROLE_LABEL = {"user": "用户", "assistant": "客服", "tool": "系统"}


def extract_qr(record):
    """从一条 sft_validation 记录提取 (query, response)。

    - response = 最后一条非空 assistant 回复（最终答复）
    - query    = 去掉该条回复后的【完整上文】转成带角色文本，
                 让模型看到上下文后续写最终答复（多轮/工具调用才能答到最后一轮）
    """
    msgs = record.get("messages", [])
    # 找最后一条非空 assistant 的下标
    last_idx = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and m.get("content", "").strip():
            last_idx = i
    if last_idx < 0:
        return "", ""
    response = msgs[last_idx]["content"].strip()
    prelude = msgs[:last_idx]
    lines = [f"{_ROLE_LABEL.get(m.get('role'), '客服')}：{m.get('content', '')}" for m in prelude]
    query = "\n".join(lines) + "\n客服："
    return query.strip(), response


def load_sft_validation(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def get_eval_data():
    records = load_sft_validation(SFT_VALIDATION)
    print("sft_validation 记录数:", len(records))

    # 先按 category 分组，统计覆盖情况
    by_cat = defaultdict(list)
    by_type = defaultdict(list)
    eval_items = []
    seen = set()

    for r in records:
        q, resp = extract_qr(r)
        if not q or not resp:
            continue
        key = (q, resp)
        if key in seen:
            continue
        seen.add(key)
        by_cat[r.get("category", "?")].append(r)
        by_type[r.get("sample_type", "?")].append(r)
        eval_items.append({"query": q, "response": resp})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in eval_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n评估集数据量: {len(eval_items)}")
    print("分类覆盖情况:")
    for k, v in sorted(by_cat.items()):
        print(f"  [{k}]  {len(v)} 条")
    print("变体覆盖情况:")
    for k, v in sorted(by_type.items()):
        print(f"  [{k}]  {len(v)} 条")
    print(f"\n评估集已保存到: {OUTPUT_FILE}")
    print("\n第一条:")
    print(eval_items[0] if eval_items else "无")


if __name__ == "__main__":
    get_eval_data()
