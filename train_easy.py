# train_grpo.py
import os
import re
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator


# Absolute path inside the container that you already bind-mount
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs/Llama-3.1-8B")

dataset = load_dataset("trl-lib/tldr", split="train")

# Reward: length close to 20 chars
def reward_len(completions, **kwargs):
    return [-abs(20 - len(c)) for c in completions]

model_id = "meta-llama/Llama-3.1-8B"

model = AutoModelForCausalLM.from_pretrained(
        model_id,
        
    )

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"
tokenizer.pad_token = "<|finetune_right_pad_id|>"
tokenizer.pad_token_id = 128004

# add LoRA to model
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    beta=0.04,
    save_strategy="steps",
    save_steps=100,
    logging_steps=10,
    save_total_limit=2,
    report_to="wandb",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)

# Resume from the latest checkpoint in OUTPUT_DIR if present
def _latest_checkpoint(base_dir: str):
    if not os.path.isdir(base_dir):
        return None
    candidates = []
    for name in os.listdir(base_dir):
        m = re.match(r"checkpoint-(\d+)$", name)
        if m:
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                candidates.append((int(m.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]

resume_path = _latest_checkpoint(OUTPUT_DIR)

if resume_path:
    print(f"Resuming training from: {resume_path}")
    trainer.train(resume_from_checkpoint=resume_path)
else:
    print("No checkpoint found. Starting fresh training run.")
    trainer.train()
