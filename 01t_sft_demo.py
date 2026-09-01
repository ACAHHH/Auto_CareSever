import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "../../models/Qwen3-0.6B/"
log_dir = "./logs/sft_lora"
parameters_dir = "./finetuned/sft_lora"

model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenzier = AutoTokenizer.from_pretrained(model_dir)

dataset = load_dataset("json",
                    data_files={"train": "./data/sft_train_split.jsonl", "test": "./data/sft_val_split.jsonl"})
data = dataset.remove_columns(["seed_id", "split", "category_id", "category", "sample_type"])
train_data = data["train"].shuffle()
test_data = data["test"].shuffle()
# print(train_data.column_names,len(train_data["messages"]))
# print(test_data.column_names,len(test_data["messages"]))


lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)

os.environ["TENSORBOARD_LOGGING_DIR"] = log_dir
config = SFTConfig(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    max_steps=450,

    logging_strategy="steps",
    logging_steps=10,
    report_to="tensorboard",
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_steps=0.1,

    eval_strategy="steps",
    eval_steps=30,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=30,
    save_total_limit=3,
    output_dir=parameters_dir,

    bf16=True,
    gradient_checkpointing=True,
    activation_offloading=False,
    max_length=768,
    assistant_only_loss=True,
    chat_template_path="./data/new_chat_template.jinja"
)

trainer = SFTTrainer(
    model=peft_model,
    args=config,
    train_dataset=train_data,
    eval_dataset=test_data,
    processing_class=tokenzier
)

trainer.train()
trainer.save_model(parameters_dir)
