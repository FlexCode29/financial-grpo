#!/usr/bin/env python3
# train_grpo.py
# Multi-GPU GRPO fine-tuning on ROCm

###############################################################################
#                         Critical environment flags                          #
###############################################################################
import os

os.environ["TORCH_DISTRIBUTED_USE_DTENSOR"] = "0"
os.environ["TORCH_DIST_DDP_SHARDING"] = "0"
os.environ["ACCELERATE_USE_TP"] = "false"
os.environ["PYTORCH_ENABLE_DISTRIBUTED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_DISTRIBUTED_DISABLE_TENSOR"] = "1"
os.environ["TORCH_DISTRIBUTED_USE_FUNCTIONAL"] = "0"

###############################################################################
#                                   Imports                                   #
###############################################################################
import argparse
import types
import requests
import torch
import torch.nn.utils.rnn as rnn_utils
from contextlib import nullcontext
from accelerate import Accelerator
from huggingface_hub import login
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

# project-specific
from src.data_loader import load_financial_dataset
from src.reward import financial_reward_function
from src.callbacks import PeriodicEvalCallback

###############################################################################
#          PEFT rename and no-adapter guard (works for both APIs)             #
###############################################################################
def patch_adapter_methods():
    """
    * Adds disable_adapter / enable_adapter aliases if only plural forms exist.
    * Wraps disable_adapter(s) so it becomes a no-op context manager when the
      model has no adapters, preventing the ValueError raised in HF > 0.21.
    """
    from contextlib import contextmanager
    import inspect

    try:
        from transformers import LlamaForCausalLM
        classes = [LlamaForCausalLM]
    except ImportError:
        classes = []

    try:
        import peft
        if hasattr(peft, "PeftModel"):
            classes.append(peft.PeftModel)
    except Exception:
        pass

    @contextmanager
    def _noop_context():
        yield

    def _wrap_disable(orig_fn):
        def wrapped(self, *a, **kw):
            try:
                return orig_fn(self, *a, **kw)
            except ValueError as e:
                if "No adapter loaded" in str(e):
                    return _noop_context()
                raise
        return wrapped

    for cls in classes:
        # alias singular to plural if only one exists
        if hasattr(cls, "disable_adapters") and not hasattr(cls, "disable_adapter"):
            setattr(cls, "disable_adapter", cls.disable_adapters)
        if hasattr(cls, "enable_adapters") and not hasattr(cls, "enable_adapter"):
            setattr(cls, "enable_adapter", cls.enable_adapters)

        # wrap whichever disable method actually exists
        for name in ("disable_adapter", "disable_adapters"):
            if hasattr(cls, name):
                fn = getattr(cls, name)
                # Avoid double wrapping
                if inspect.isfunction(fn) or inspect.ismethod(fn):
                    setattr(cls, name, _wrap_disable(fn))


patch_adapter_methods()

###############################################################################
#                           Static system prompt                              #
###############################################################################
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

###############################################################################
#                      Prompt building and truncation                         #
###############################################################################
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


def truncate_to_tokens(text, tokenizer, max_tokens):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    ids = ids[-max_tokens:]  # keep only the tail
    return tokenizer.decode(ids, skip_special_tokens=False)


def format_prompts(batch, tokenizer, max_prompt_len):
    prompts = []
    for prompt_text, is_rl in zip(batch["prompt"], batch["is_grpo_task"]):
        if is_rl:
            msgs = [
                {"role": "system", "content": FINANCIAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]
        else:
            msgs = [{"role": "user", "content": prompt_text}]
        chat = build_chat(msgs, tokenizer, add_generation_prompt=True)
        chat = truncate_to_tokens(chat, tokenizer, max_prompt_len)
        prompts.append(chat)
    return {"prompt": prompts}

###############################################################################
#                  Safe wrappers around model.generate                        #
###############################################################################
def make_safe_generate(model, tokenizer):
    """
    Ensures every generate call returns <prompt> plus at least one token.
    """
    original = model.generate

    def safe_generate(self, input_ids, attention_mask=None, **kw):
        out = original(input_ids, attention_mask=attention_mask, **kw)
        if out.size(1) == input_ids.size(1):
            pad_tok = (
                tokenizer.eos_token_id
                or tokenizer.pad_token_id
                or tokenizer.unk_token_id
                or 0
            )
            pad = torch.full(
                (out.size(0), 1), pad_tok, dtype=out.dtype, device=out.device
            )
            out = torch.cat([out, pad], dim=1)
        return out

    model.generate = types.MethodType(safe_generate, model)

###############################################################################
#                    Remote vLLM monkey-patch (safe)                          #
###############################################################################
def attach_remote_generate(model, tokenizer, remote_url, max_new_tokens):
    remote_url = remote_url.rstrip("/") + "/generate"

    def remote_generate(self, input_ids, attention_mask=None, **gkw):
        prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        payload = {
            "prompts": prompts,
            "max_tokens": gkw.get("max_new_tokens", max_new_tokens),
        }
        resp = requests.post(remote_url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()

        if "completion_ids" in data:
            comp_ids = data["completion_ids"]
        elif "outputs" in data:
            comp_ids = [o["token_ids"] for o in data["outputs"]]
        else:
            raise ValueError("vLLM response missing token ids")

        fixed = [
            seq if seq else [tokenizer.eos_token_id or tokenizer.pad_token_id or 0]
            for seq in comp_ids
        ]
        tensors = [torch.tensor(seq, device=input_ids.device) for seq in fixed]
        comps = rnn_utils.pad_sequence(
            tensors, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        return torch.cat([input_ids, comps], dim=1)

    model.generate = types.MethodType(remote_generate, model)
    make_safe_generate(model, tokenizer)  # extra guard

###############################################################################
#                                   Main                                      #
###############################################################################
def main(args):
    if args.max_seq_length < args.max_prompt_length + args.max_completion_length:
        raise ValueError(
            f"max_seq_length {args.max_seq_length} must be at least "
            f"max_prompt_length + max_completion_length "
            f"({args.max_prompt_length + args.max_completion_length})"
        )

    accelerator = Accelerator()
    if accelerator.is_main_process and (token := os.getenv("HF_TOKEN")):
        login(token=token)

    if args.report_to == "wandb" and accelerator.is_main_process:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() or torch.backends.hip.is_available()
        else torch.float32
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,  # avoids clash with vLLM
        dtype=dtype,
        token=os.getenv("HF_TOKEN"),
        fast_inference=False,
        device_map={"": accelerator.device},
        gpu_memory_utilization=0.6,
    )

    if tokenizer.chat_template is None:
        tokenizer.chat_template = """
{% for m in messages -%}
<|start_header_id|>{{m['role']}}<|end_header_id|>
{{m['content']}}<|eot_id|>
{% endfor -%}
{% if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>
{% endif -%}""".strip()

    # generation patch
    if args.use_remote_vllm:
        if accelerator.is_main_process:
            print(f"Using remote vLLM at {args.vllm_remote_url} for generation")
        attach_remote_generate(model, tokenizer, args.vllm_remote_url, args.max_completion_length)
    else:
        make_safe_generate(model, tokenizer)
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.lora_rank,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        if accelerator.is_main_process:
            model.print_trainable_parameters()

    ############ data ############
    train_ds = load_financial_dataset(
        split="train", test_size=args.test_split_size, k_folds=args.k_folds, fold_index=args.fold_index
    )
    eval_ds = load_financial_dataset(
        split="test", test_size=args.test_split_size, k_folds=args.k_folds, fold_index=args.fold_index
    )

    train_ds = train_ds.map(
        lambda b: format_prompts(b, tokenizer, args.max_prompt_length), batched=True
    )

    ############ trainer config ############
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
        ddp_find_unused_parameters=False,
        warmup_steps=args.warmup_steps,
        optim="paged_adamw_8bit",
        bf16=torch.cuda.is_available() or torch.backends.hip.is_available(),
        use_vllm=False,  # manually handled
    )

    eval_cb = PeriodicEvalCallback(
        tokenizer=tokenizer, eval_dataset=eval_ds, eval_steps=args.eval_steps
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

###############################################################################
#                                   CLI                                       #
###############################################################################
if __name__ == "__main__":
    DEFAULTS = dict(
        model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
        output_dir="outputs_8b_grpo",
        max_seq_length=2048 + 512,
        max_prompt_length=2048,
        max_completion_length=512,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default=DEFAULTS["model_name"])
    p.add_argument("--output_dir", default=DEFAULTS["output_dir"])
    p.add_argument("--max_seq_length", type=int, default=DEFAULTS["max_seq_length"])
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--max_prompt_length", type=int, default=DEFAULTS["max_prompt_length"])
    p.add_argument("--max_completion_length", type=int, default=DEFAULTS["max_completion_length"])
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--report_to", choices=["wandb", "none"], default="none")
    p.add_argument("--wandb_project", default="unsloth_grpo")
    p.add_argument("--test_split_size", type=float, default=0.2)
    p.add_argument("--k_folds", type=int, default=1)
    p.add_argument("--fold_index", type=int, default=0)
    p.add_argument("--use_remote_vllm", action="store_true")
    p.add_argument("--vllm_remote_url", default="http://localhost:8000")
    main(p.parse_args())
