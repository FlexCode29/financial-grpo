import argparse
import os
import torch
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from huggingface_hub import login

from src.data_loader import load_financial_dataset
from src.reward import financial_reward_function
from src.callbacks import PeriodicEvalCallback

FINANCIAL_SYSTEM_PROMPT = """You are an expert financial analyst AI. Your task is to predict future stock performance based on 20 years of financial data.
Respond in the following XML format, providing your reasoning first, followed by the answer.

<reasoning>
[Your detailed analysis and reasoning for the predictions goes here.]
</reasoning>
<answer>
<returns_1y>[bad/neutral/good]</returns_1y>
<volatility_1y>[low/medium/high]</volatility_1y>
<returns_5y>[bad/neutral/good]</returns_5y>
<volatility_5y>[low/medium/high]</volatility_5y>
</answer>"""


def main(args):
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token: login(token=hf_token)
    if args.report_to == 'wandb':
        import wandb
        wandb.init(project=args.wandb_project, config=args)
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("ROCm-enabled GPU detected. Using GPU for training.")
        dtype, load_in_4bit, bf16_enabled = torch.bfloat16, False, True
    else:
        print("!!! WARNING: No compatible GPU detected. Falling back to CPU for smoke test. !!!")
        dtype, load_in_4bit, bf16_enabled = torch.float32, False, False
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name, max_seq_length=args.max_seq_length,
        dtype=dtype, load_in_4bit=load_in_4bit, token=hf_token, fast_inference=False,
    )
    if tokenizer.chat_template is None:          # ← only patch once
        tokenizer.chat_template = """
        {% for m in messages %}
        {{ '<|begin_of_text|>' if loop.first }}
        {% if m['role'] == 'system' -%}
        <|start_header_id|>system<|end_header_id|>
        {{ m['content'] }}<|eot_id|>
        {% elif m['role'] == 'user' -%}
        <|start_header_id|>user<|end_header_id|>
        {{ m['content'] }}<|eot_id|>
        {% elif m['role'] == 'assistant' -%}
        <|start_header_id|>assistant<|end_header_id|>
        {{ m['content'] }}<|eot_id|>
        {% endif %}{% endfor %}
        {% if add_generation_prompt -%}
        <|start_header_id|>assistant<|end_header_id|>
        {% endif -%}""".strip()
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_rank, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank, use_gradient_checkpointing="unsloth", random_state=42
    )
    print("LoRA adapters added to the model.")

    print("Loading datasets for training and periodic evaluation...")
    train_dataset = load_financial_dataset(
        split='train',
        test_size=args.test_split_size,
        k_folds=args.k_folds,
        fold_index=args.fold_index
    )
    eval_dataset = load_financial_dataset(
        split='test',
        test_size=args.test_split_size,
        k_folds=args.k_folds,
        fold_index=args.fold_index
    )

    def format_prompt(examples):
        prompts = []
        for i, prompt in enumerate(examples['prompt']):
            if examples['is_grpo_task'][i]:
                messages = [{"role": "system", "content": FINANCIAL_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(formatted_prompt)
        return {"prompt": prompts}

    train_dataset = train_dataset.map(format_prompt, batched=True)
    print(f"Dataset prepared with {len(train_dataset)} training examples.")

    training_args = GRPOConfig(
        output_dir=args.output_dir, num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length, max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, lr_scheduler_type="cosine", max_steps=args.max_steps,
        save_strategy="steps", save_steps=100, logging_steps=1,
        report_to=args.report_to, remove_unused_columns=False,
        warmup_steps=20, optim="paged_adamw_8bit", bf16=torch.cuda.is_available(),
        use_vllm=False,
    )

    periodic_eval_callback = PeriodicEvalCallback(
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        eval_steps=args.eval_steps
    )

    grpo_trainer = GRPOTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[financial_reward_function],
        tokenizer=tokenizer,
        callbacks=[PeriodicEvalCallback(tokenizer, eval_dataset)] # Pass the callback
    )

    print("\n--- Starting GRPO Training (with vLLM acceleration) ---")
    grpo_trainer.train()
    print("--- Training Finished ---")

    print(f"Saving final LoRA adapters to {args.output_dir}")
    grpo_trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GRPO training for financial prediction.")
    parser.add_argument("--eval_steps", type=int, default=100, help="Run evaluation every N steps.")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2-1.5B-Instruct-GGUF", help="Base model.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for final adapters.")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=1536)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--test_split_size", type=float, default=0.2, help="Fraction of data for testing (used if k_folds=1).")
    parser.add_argument("--k_folds", type=int, default=1, help="Number of folds for cross-validation.")
    parser.add_argument("--fold_index", type=int, default=0, help="The current fold index to run (0-based).")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])
    parser.add_argument("--wandb_project", type=str, default="financial-grpo-llm")
    args = parser.parse_args()
    main(args)

""" 
VERSION FOR CPU SMOKE TESTING
def main(args):
    if args.hf_token:
        login(token=args.hf_token)

    if args.report_to == 'wandb':
        import wandb
        wandb.init(project=args.wandb_project, config=args)
    
    # --- START OF CHANGE ---
    # 1. Automatically detect the device and set model loading parameters
    if torch.cuda.is_available():
        print("CUDA is available! Using GPU.")
        device = "cuda"
        dtype = torch.bfloat16
        load_in_4bit = True
    else:
        print("CUDA not available. Running on CPU for smoke test.")
        device = "cpu"
        dtype = torch.float32  # CPU does not support bfloat16
        load_in_4bit = False   # 4-bit loading is a GPU-only feature
    # --- END OF CHANGE ---
    
    print(f"Loading base model: {args.model_name}")
    # 2. Use the variables we just defined
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=args.hf_token,
    )

    # The rest of the function remains exactly the same...
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=args.max_seq_length,
    )
    print("LoRA adapters added to the model.")

    train_dataset = load_financial_dataset(split='train')

    def format_prompt(examples):
        prompts = []
        for i, prompt in enumerate(examples['prompt']):
            if examples['is_grpo_task'][i]:
                messages = [{"role": "system", "content": FINANCIAL_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(formatted_prompt)
        return {"prompt": prompts}

    train_dataset = train_dataset.map(format_prompt, batched=True)
    print(f"Dataset prepared with {len(train_dataset)} training examples.")

    training_args = GRPOConfig(
        output_dir=args.output_dir, num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length, max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True, learning_rate=args.learning_rate,
        lr_scheduler_type="cosine", max_steps=args.max_steps,
        save_strategy="steps", save_steps=100, logging_steps=1, # Log every step for quick feedback
        report_to=args.report_to, remove_unused_columns=False,
        warmup_steps=5, optim="adamw_8bit", bf16=False, # bf16 is a GPU feature
    )

    grpo_trainer = GRPOTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[financial_reward_function],
        tokenizer=tokenizer,
    )

    print("\n--- Starting GRPO Training (CPU Smoke Test) ---")
    grpo_trainer.train()
    print("--- Training Finished ---")

    print(f"Saving final LoRA adapters to {args.output_dir}")
    grpo_trainer.save_model()
 """