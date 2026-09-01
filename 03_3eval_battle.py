import os
from evalscope import TaskConfig, run_task
# ============================================================
# Arena Pairwise 配置
# ============================================================
# 本次实验统一放到同一个目录
WORK_DIR = f"./outputs/arena_demo1/"
JUDGE_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
JUDGE_API_URL = "https://api.siliconflow.cn/v1"
SFT_REPORT_PATH = os.path.join(
    WORK_DIR,
    "reports",
    "sft_demo",
)

DPO_REPORT_PATH = os.path.join(
    WORK_DIR,
    "reports",
    "dpo_demo",
)
# ============================================================
# 中文 Pairwise Judge Prompt
# ============================================================
JUDGE_SYSTEM_PROMPT = """
            你是一名严格、公正的大语言模型回答质量评估员。
            
            现在会给你同一个用户问题，以及两个 AI 助手生成的回答：
            回答 A 和回答 B。
            
            请比较两个回答的整体质量，重点考虑：
            1. 有用性
            2. 正确性与诚实性
            3. 相关性
            4. 清晰度
            5. 指令遵循
            6. 完整性
            7. 整体人类偏好
            
            不要因为回答更长就认为回答更好。
            如果两个回答质量非常接近，应当判定为平局。
            
            最终必须严格使用以下五种格式之一：
            [[A>>B]]
            [[A>B]]
            [[A=B]]
            [[B>A]]
            [[B>>A]]
"""

JUDGE_PROMPT_TEMPLATE = """
            【用户问题】
            {question}
            
            【回答 A】
            {answer_1}
            
            【回答 B】
            {answer_2}
            
            请比较回答 A 和回答 B 的整体质量。
            
            请先进行简短分析，然后给出最终结论。
"""

# ============================================================
# 定义竞技场配置
# ============================================================
arena_cfg = TaskConfig(
    model_id="Arena",
    # 必须使用 general_arena
    datasets=[
        "general_arena"
    ],
    dataset_args={
        "general_arena": {
            # 中文 Judge Prompt
            "system_prompt": JUDGE_SYSTEM_PROMPT,
            "prompt_template": JUDGE_PROMPT_TEMPLATE,
            "extra_params": {
                # 候选模型
                "models": [
                    {
                        "name": "dpo_demo",
                        "report_path": DPO_REPORT_PATH,
                    },
                    {
                        "name": "sft_demo",
                        "report_path": SFT_REPORT_PATH,
                    },
                ],
                # SFT 作为 baseline
                "baseline": "sft_demo",
            }
        }
    },

    # DeepSeek Judge
    judge_model_args={
        "model_id": JUDGE_MODEL,
        "api_url": JUDGE_API_URL,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "generation_config": {
            "temperature": 0.0,
            "max_tokens": 2048,
            "retries": 3,
            "extra_body": {
                "thinking": {
                    "type": "disabled"
                }
            },
        },
    },

    # Judge 并发
    eval_batch_size=15,
    # Arena 单独放一个目录
    work_dir=os.path.join(
        WORK_DIR,
        "arena"
    ),
    no_timestamp=True,
)

# ============================================================
# 6. Arena评估
# ============================================================
print()
print("=" * 70)
print("STEP 2：SFT vs DPO Pairwise Arena")
print("=" * 70)

run_task(
    task_cfg=arena_cfg
)
