"""
train_grpo.py
Multi-GPU GRPO fine-tuning on ROCm

• Local policy and optimizer run on GPUs 1-N
• A separate vLLM server (GPU 0) generates K candidate completions
  via its /generate endpoint, returning `{"completion_ids": [[...], ...]}`.

Run examples
============

# terminal 1 (inference GPU 0)
HIP_VISIBLE_DEVICES=0 ./run.sh server \
  --model meta-llama/meta-Llama-3.1-8B-Instruct \
  --max-model-len 4096

# terminal 2 (training GPUs 1-7)
export HIP_VISIBLE_DEVICES=1,2,3,4,5,6,7
./run.sh train \
  --use_remote_vllm \
  --vllm_remote_url http://localhost:8000 \
  --report_to wandb
"""
import os
# --- Critical Environment Variables (set BEFORE torch import) ---
os.environ["TORCH_DISTRIBUTED_USE_DTENSOR"] = "0"
os.environ["TORCH_DIST_DDP_SHARDING"] = "0"
os.environ["ACCELERATE_USE_TP"] = "false"
os.environ["PYTORCH_ENABLE_DISTRIBUTED"] = "1"
# For more verbose NCCL, uncomment next line. Can be very noisy.
# os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
# For CUDA errors, makes them synchronous. Slows things down significantly. Use if suspecting CUDA misbehavior.
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

os.environ["TORCH_DISTRIBUTED_DISABLE_TENSOR"] = "1"
os.environ["TORCH_DISTRIBUTED_USE_FUNCTIONAL"] = "0"

import argparse
import types
import requests
import torch
import torch.nn.utils.rnn as rnn_utils
from accelerate import Accelerator
from huggingface_hub import login
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

# ------------- project-specific imports -------------
from src.data_loader import load_financial_dataset
from src.reward import financial_reward_function
from src.callbacks import PeriodicEvalCallback
# ----------------------------------------------------

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

# ---------- patch: remote vLLM generation ----------
def attach_remote_generate(model, tokenizer, remote_url, max_new_tokens):
    """
    Monkey-patch model.generate so that GRPOTrainer calls a remote
    vLLM server instead of local autoregressive sampling.
    """
    remote_url = remote_url.rstrip("/") + "/generate"

    def remote_generate(self, input_ids, attention_mask=None, **gen_kwargs):
        # 1. decode batched prompts
        prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        # 2. ask vLLM for completions
        payload = {
            "prompts": prompts,
            "max_tokens": gen_kwargs.get("max_new_tokens", max_new_tokens),
        }
        resp = requests.post(remote_url, json=payload, timeout=600)
        resp.raise_for_status()
        completion_ids = resp.json()["completion_ids"]  # list[list[int]]
        # 3. convert to torch tensors and pad
        tensors = [torch.tensor(seq, device=input_ids.device) for seq in completion_ids]
        padded = rnn_utils.pad_sequence(
            tensors, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        return padded

    # bind the method to the model instance
    model.generate = types.MethodType(remote_generate, model)

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

    dtype = torch.bfloat16 if (torch.cuda.is_available() or torch.backends.hip.is_available()) else torch.float32

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,  # 4-bit clashes with vLLM engine
        dtype=dtype,
        token=os.getenv("HF_TOKEN"),
        fast_inference=False,
        device_map={"": accelerator.device},
        gpu_memory_utilization=0.6,
    )

    # ensure a chat template exists
    if tokenizer.chat_template is None:
        tokenizer.chat_template = """
{% for m in messages -%}
<|start_header_id|>{{m['role']}}<|end_header_id|>
{{m['content']}}<|eot_id|>
{% endfor -%}
{% if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>
{% endif -%}""".strip()

    # LoRA only when not using remote vLLM
    if not args.use_remote_vllm:
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
            print(f"Using remote vLLM at {args.vllm_remote_url} for generation")
        attach_remote_generate(
            model,
            tokenizer,
            remote_url=args.vllm_remote_url,
            max_new_tokens=args.max_completion_length,
        )

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
        ddp_find_unused_parameters = False,
        warmup_steps=args.warmup_steps,
        optim="paged_adamw_8bit",
        bf16=torch.cuda.is_available() or torch.backends.hip.is_available(),
        # we tell TRL *not* to spin up a local vLLM engine
        use_vllm=False,
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
    parser.add_argument("--model_name", type=str, default="meta-llama/meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs_8b_grpo")
    parser.add_argument("--max_seq_length", type=int, default=2048 + 512)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--report_to", type=str, default="none", choices=["wandb", "none"])
    parser.add_argument("--wandb_project", type=str, default="unsloth_grpo")
    parser.add_argument("--test_split_size", type=float, default=0.2)
    parser.add_argument("--k_folds", type=int, default=1)
    parser.add_argument("--fold_index", type=int, default=0)

    # --- new flags for remote vLLM ---
    parser.add_argument("--use_remote_vllm", action="store_true",
                        help="Use an external vLLM server for generation")
    parser.add_argument("--vllm_remote_url", type=str, default="http://localhost:8000",
                        help="Base URL of the vLLM REST server")

    args = parser.parse_args()
    main(args)
