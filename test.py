from unsloth import FastLanguageModel
from data_process import SYSTEM_PROMPT, load_saved_dataset
from reward_functions import correctness_reward_func
from vllm import SamplingParams

max_seq_length = 2048
lora_rank = 64

# 加载测试集
_, test_dataset = load_saved_dataset("data_cache")

test_dataset = [sample for sample in test_dataset if isinstance(sample, dict)]

test_dataset = test_dataset[:100]

# # 加载基础模型
# base_model, base_tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "Qwen/Qwen2.5-3B-Instruct",
#     max_seq_length = max_seq_length,
#     load_in_4bit = True,
#     fast_inference = True,
#     max_lora_rank = lora_rank,
#     gpu_memory_utilization = 0.5,
# )

# 加载微调后模型（合并后模型）
finetuned_model, finetuned_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./model",  # 合并后模型目录
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5,
)

sampling_params = SamplingParams(
    temperature = 0.0,  # 评测时建议设为0，保证确定性
    top_p = 1.0,
    max_tokens = 1024,
)

def get_outputs(model, tokenizer, dataset):
    outputs = []
    for idx, sample in enumerate(dataset):
        try:
            text = tokenizer.apply_chat_template(
                sample["prompt"],
                tokenize=False,
                add_generation_prompt=True
            )
            output = model.fast_generate(
                text,
                sampling_params=sampling_params,
            )[0].outputs[0].text
            outputs.append(output)
            if idx % 5 == 0:
                print(f"已完成 {idx+1}/{len(dataset)} 条推理")
        except Exception as e:
            print(f"第{idx+1}条推理出错: {e}")
            outputs.append("")
    return outputs

# 推理
# base_outputs = get_outputs(base_model, base_tokenizer, test_dataset)
finetuned_outputs = get_outputs(finetuned_model, finetuned_tokenizer, test_dataset)

from reward_functions import correctness_reward_func

answers = [sample["answer"] for sample in test_dataset]
db_set = [sample.get("db_set", "gsm8k") for sample in test_dataset]
prompts = [sample["prompt"] for sample in test_dataset]

def wrap_outputs(outputs):
    return [[{"content": o}] for o in outputs]

# base_rewards = correctness_reward_func(
#     prompts=prompts,
#     completions=wrap_outputs(base_outputs),
#     answer=answers,
#     db_set=db_set
# )
finetuned_rewards = correctness_reward_func(
    prompts=prompts,
    completions=wrap_outputs(finetuned_outputs),
    answer=answers,
    db_set=db_set
)

# base_acc = sum([1 for r in base_rewards if r > 0]) / len(base_rewards)
finetuned_acc = sum([1 for r in finetuned_rewards if r > 0]) / len(finetuned_rewards)

# print(f"基础模型测试集正确率: {base_acc:.2%}")
print(f"微调后模型测试集正确率: {finetuned_acc:.2%}")