from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from data_process import prepare_and_save_dataset, load_saved_dataset
from reward_functions import (
    correctness_reward_func, int_reward_func, strict_format_reward_func, soft_format_reward_func, xmlcount_reward_func,safety_reward_func
)
from trl import GRPOConfig, GRPOTrainer
import torch
import os

PatchFastRL("GRPO", FastLanguageModel)

max_seq_length = 4096
lora_rank = 64

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-7B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

save_dir = "data_cache"
if not (os.path.exists(f"{save_dir}/train") and os.path.exists(f"{save_dir}/test")):
    print("未检测到本地缓存数据，正在处理并保存数据集……")
    prepare_and_save_dataset(save_dir)
train_dataset, test_dataset = load_saved_dataset(save_dir)
print(f"train size: {len(train_dataset)}, test size: {len(test_dataset)}")

training_args = GRPOConfig(
    use_vllm = True,
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    num_generations = 6,
    max_prompt_length = 2048,
    max_completion_length = 2048,
    max_steps = 750,
    save_steps = 100,
    max_grad_norm = 0.1,
    report_to = "none",
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
        safety_reward_func,
    ],
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset=test_dataset,
)
trainer.train()

model.save_lora("grpo_saved_lora")
model.save_pretrained_merged("model", tokenizer) 