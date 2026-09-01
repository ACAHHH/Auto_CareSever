import os, torch
from datasets import load_dataset
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.dpo_trainer import DPOTrainer
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)

base_model_dir = "../../models/Qwen3-0.6B"
sft_adapter_dir = "./finetuned/sft_lora"
log_dir = "./logs/dpo_Qlora"
parameters_dir = "./finetuned/dpo_Qlora"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_dir,
    quantization_config=quantization_config,
    dtype=torch.bfloat16
)

model = prepare_model_for_kbit_training(model=model)

model = PeftModel.from_pretrained(
    model,
    sft_adapter_dir,
    is_trainable=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
with open("./data/new_chat_template.jinja", mode="r", encoding="utf-8") as f:
    tokenizer.chat_template = f.read()

dataset = load_dataset("json",
                    data_files={"train": "./data/dpo_train_split.jsonl", "test": "./data/dpo_eval_split.jsonl"})
data = dataset.remove_columns(["seed_id","defect_type"])
train_data = data["train"].shuffle()
test_data = data["test"].shuffle()
# print(train_data.column_names,len(train_data))
# print(test_data.column_names,len(test_data))

os.environ["TENSORBOARD_LOGGING_DIR"] = log_dir
config = DPOConfig(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    max_steps=460,
    logging_strategy="steps",
    logging_steps=10,
    report_to="tensorboard",
    learning_rate=3e-6,
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
    beta=0.1,
    optim="adamw_torch",
    max_length=512
)

if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=train_data,
    eval_dataset=test_data,
    processing_class=tokenizer
)

trainer.train()
trainer.save_model(parameters_dir)
