from evalscope import TaskConfig, run_task

SFT_MODEL = "./finetuned/sft_lora"
DPO_MODEL = "./finetuned/dpo_Qlora"

DATASET_PATH = "custom_eval/text/qa1"
SUBSET = "ultrafeedback"

# 本次实验统一放到同一个目录
WORK_DIR = f"./outputs/arena_demo1/"

# ============================================================
# SFT + DPO 候选模型推理
# ============================================================

models = [
    {
        "model": SFT_MODEL,
        "model_id": "sft_demo",
    },
    {
        "model": DPO_MODEL,
        "model_id": "dpo_demo",
    },
]

task_list = [

    TaskConfig(
        # 本地模型
        model=item["model"],
        model_id=item["model_id"],
        # EvalScope 1.6.0 本地 checkpoint，固定写法
        eval_type="llm_ckpt",

        # 统一使用 general_qa
        datasets=[
            "general_qa"
        ],

        dataset_args={
            "general_qa": {
                "dataset_id": DATASET_PATH,
                "subset_list": [SUBSET],
                # Arena 第一阶段只需要生成回答
                # 不需要计算 BLEU / Rouge
                "metric_list": ["Rouge"]
            }
        },

        # 推理 Batch Size
        eval_batch_size=16,
        # 生成参数
        generation_config={
            "batch_size": 16,
            "max_tokens": 2048,
            "do_sample": False,
        },

        # 两个模型必须写入同一个实验目录
        work_dir=WORK_DIR,
        # 不让 EvalScope 再自动添加时间戳
        no_timestamp=True,
        seed=42,
    )
    for item in models
]

print("=" * 70)
print("STEP 1：SFT + DPO Candidate Inference")
print("=" * 70)

run_task(
    task_cfg=task_list
)
