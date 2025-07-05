from unsloth import FastLanguageModel
from data_process import SYSTEM_PROMPT
import torch
from vllm import SamplingParams

max_seq_length = 2048
lora_rank = 64

# 加载未微调模型
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-7B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5,
)

# 加载微调模型
finetuned_model, finetuned_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./model",  # 合并后的模型目录
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5,
)

# 构造输入
text = base_tokenizer.apply_chat_template([
    {"role" : "system", "content" : SYSTEM_PROMPT},
    {"role" : "user", "content" : "Is Aspirin good for cardio vascular function?"},
], tokenize = False, add_generation_prompt = True)

sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)

# 未微调模型推理
base_output = base_model.fast_generate(
    text,
    sampling_params = sampling_params,
)[0].outputs[0].text

# 微调模型推理
finetuned_output = finetuned_model.fast_generate(
    text,
    sampling_params = sampling_params,
)[0].outputs[0].text

print("未微调模型推理结果：", base_output)
print("微调后模型推理结果：", finetuned_output)