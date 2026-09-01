"""
加载模型参数，读取用户传入的prompt，让模型自回归生成，检测微调后模型的效果
"""
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM

prompt = "您好，天窗关闭后遮阳帘仍停在半开位置，能帮我处理一下吗？"
model_path = r"D:\Program Files\PythonProject_dl&llm\LLM\Auto_CareSever\finetuned\sft_lora"
# model_path = r"D:\Program Files\PythonProject_dl&llm\models\Qwen3-0.6B"

# 加载模型的参数
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 让模型做自回归生成
message_list = [
    {"role": "user", "content": prompt}
]
result: dict[str, List] = tokenizer.apply_chat_template(
    message_list,
    tokenize=True,
    return_dict=True,
    add_generation_prompt=True)

token_id: List[int] = result["input_ids"]  # 表示的是，经过聊天模板格式化之后，分词所得的token_ids序列

input_ids_tensor = torch.tensor([token_id], dtype=torch.long).to("cuda")
# generated_result: 包含输入和输出的token_ids的tensor， shape: batch_size, seq_len
generated_result = model.generate(input_ids_tensor, max_new_tokens=500)

# 从generated_result里面获取输出，进行解码
generated_new_tokens = generated_result[0][len(token_id):].tolist()

generated_text = tokenizer.decode(generated_new_tokens, skip_special_tokens=True)

print(generated_text)
