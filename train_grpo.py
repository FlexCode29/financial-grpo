"""
Multi-GPU GRPO fine-tuning on ROCm with optional vLLM off-loading
===============================================================

Save as train_grpo.py, then run e.g.

# terminal 1 – vLLM server on GPU 0
HIP_VISIBLE_DEVICES=0 trl vllm-serve \
  --model unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit \
  --max-model-len 4096

# terminal 2 – GRPO training on GPUs 1-7
export HIP_VISIBLE_DEVICES=1,2,3,4,5,6,7
accelerate launch --multi_gpu --num_processes 7 train_grpo.py \
  --model_name unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit \
  --use_vllm --vllm_device rocm:0 \
  --report_to wandb
"""

import os
import argparse
import torch
from accelerate import Accelerator
from huggingface_hub import login
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

# project-specific imports
from src.data_loader import load_financial_dataset
from src.reward import financial_reward_function
from src.callbacks import PeriodicEvalCallback

FINANCIAL_SYSTEM_PROMPT = """You are an expert financial analyst AI. Your task is to predict future stock performance based on 20 years of financial data.
Respond in XML format, giving reasoning first.

<reasoning>
[Your analysis]
</reasoning>
<answer>
<returns_1y>[bad/neutral/good]</returns_1y>
<volatility_1y>[low/medium/high]</volatility_1y>
<returns_5y>[bad/neutral/good]</returns_5y>
<volatility_5y>[low/medium/high]</volatility_5y>
</answer>"""

# ---------- helper functions ----------
def build_chat(messages, tokenizer, add_generation_prompt=True):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    out = ""
    for m in messages:
        out += f"<|start_header_id|>{m['role']}<|end_header_id|>\n{m['content']}<|eot_id|>"
    if add_generation_prompt:
        out += "<|start_header_id|>assistant<|end_header_id|>\n"
    return out

def format_prompts(batch, tokenizer):
    prompts = []
    for prompt_text, is_rl in zip(batch["prompt"], batch["is_grpo_task"]):
        if is_rl:
            msgs = [
                {"role": "system", "content": FINANCIAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]
        else:
            msgs = [{"role": "user", "content": prompt_text}]
        prompts.append(build_chat(msgs, tokenizer, add_generation_prompt=True))
    return {"prompt": prompts}

# ---------- main ----------
def main(args):
    accelerator = Accelerator()
    if accelerator.is_main_process:
        token = os.getenv("HF_TOKEN")
        if token:
            login(token=token)

    if args.report_to == "wandb" and accelerator.is_main_process:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # -------- model load --------
    dtype = torch.bfloat16 if (torch.cuda.is_available() or torch.backends.hip.is_available()) else torch.float32

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.use_vllm,           # 4-bit LoRA clashes with vLLM
        dtype=dtype,
        token=os.getenv("HF_TOKEN"),
        fast_inference=False,
    )

    # add chat template once
    if tokenizer.chat_template is None:
        tokenizer.chat_template = """
{% for m in messages -%}
<|start_header_id|>{{m['role']}}<|end_header_id|>
{{m['content']}}<|eot_id|>
{% endfor -%}
{% if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>
{% endif -%}""".strip()

    # LoRA only when vLLM is off
    if not args.use_vllm:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=args.lora_rank,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        if accelerator.is_main_process:
            model.print_trainable_parameters()
    else:
        if accelerator.is_main_process:
            print("vLLM mode: full-weight training (LoRA disabled)")

    # -------- data --------
    train_ds = load_financial_dataset(
        split="train",
        test_size=args.test_split_size,
        k_folds=args.k_folds,
        fold_index=args.fold_index,
    )
    eval_ds = load_financial_dataset(
        split="test",
        test_size=args.test_split_size,
        k_folds=args.k_folds,
        fold_index=args.fold_index,
    )
    train_ds = train_ds.map(lambda b: format_prompts(b, tokenizer), batched=True)

    # -------- trainer config --------
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=1,
        report_to=args.report_to,
        remove_unused_columns=False,
        warmup_steps=args.warmup_steps,
        optim="paged_adamw_8bit",
        bf16=torch.cuda.is_available() or torch.backends.hip.is_available(),
        use_vllm=args.use_vllm,
        vllm_device=args.vllm_device,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )

    eval_cb = PeriodicEvalCallback(
        tokenizer=tokenizer,
        eval_dataset=eval_ds,
        eval_steps=args.eval_steps,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        reward_funcs=[financial_reward_function],
        tokenizer=tokenizer,
        callbacks=[eval_cb],
    )

    if accelerator.is_main_process:
        print("Starting GRPO training")
    trainer.train()
    if accelerator.is_main_process:
        print("Training finished, saving model or adapters")
    trainer.save_model()

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit")
    parser.add_argument("--output_dir", type=str, default="outputs_70b_grpo")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--report_to", type=str, default="none", choices=["wandb", "none"])
    parser.add_argument("--wandb_project", type=str, default="unsloth_70b_grpo")
    parser.add_argument("--test_split_size", type=float, default=0.2)
    parser.add_argument("--k_folds", type=int, default=1)
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--use_vllm", action="store_true", help="Enable remote vLLM generation")
    parser.add_argument("--vllm_device", type=str, default="rocm:0", help="Device id reserved for vLLM")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    main(args)
