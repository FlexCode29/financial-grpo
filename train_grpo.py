#!/usr/bin/env python3
# train_grpo.py  – LoRA policy · remote-vLLM sampler · ROCm

###############################################################################
#                         Critical environment flags                          #
###############################################################################
import os

os.environ["TORCH_DISTRIBUTED_USE_DTENSOR"]       = "0"
os.environ["TORCH_DIST_DDP_SHARDING"]             = "0"
os.environ["ACCELERATE_USE_TP"]                   = "false"
os.environ["PYTORCH_ENABLE_DISTRIBUTED"]          = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]             = "expandable_segments:True"
os.environ["TORCH_COMPILE_DISABLE"]               = "1"
os.environ["TORCHINDUCTOR_DISABLE"]               = "1"
os.environ["TORCH_DISTRIBUTED_DISABLE_TENSOR"]    = "1"
os.environ["TORCH_DISTRIBUTED_USE_FUNCTIONAL"]    = "0"

###############################################################################
#                                   Imports                                   #
###############################################################################
import argparse, inspect, types, requests, torch
from contextlib import contextmanager, nullcontext
import torch.nn.utils.rnn as rnn_utils
from accelerate import Accelerator
from huggingface_hub import login
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

# project-specific
from src.data_loader import load_financial_dataset
from src.reward      import financial_reward_function
from src.callbacks   import PeriodicEvalCallback

###############################################################################
#      PEFT rename + safe disable_adapter wrapper for newer HF versions       #
###############################################################################
def patch_adapter_methods() -> None:
    try:
        from transformers import LlamaForCausalLM
        targets = [LlamaForCausalLM]
    except ImportError:
        targets = []

    try:
        import peft
        if hasattr(peft, "PeftModel"):
            targets.append(peft.PeftModel)
    except Exception:
        pass

    @contextmanager
    def _noop():
        yield

    def _wrap_disable(fn):
        def inner(self, *a, **k):
            try:
                return fn(self, *a, **k)
            except ValueError as e:
                if "No adapter loaded" in str(e):
                    return _noop()
                raise
        return inner

    for cls in targets:
        if hasattr(cls, "disable_adapters") and not hasattr(cls, "disable_adapter"):
            setattr(cls, "disable_adapter", cls.disable_adapters)
        if hasattr(cls, "enable_adapters") and not hasattr(cls, "enable_adapter"):
            setattr(cls, "enable_adapter", cls.enable_adapters)

        for name in ("disable_adapter", "disable_adapters"):
            if hasattr(cls, name):
                fn = getattr(cls, name)
                if inspect.isfunction(fn) or inspect.ismethod(fn):
                    setattr(cls, name, _wrap_disable(fn))

patch_adapter_methods()

###############################################################################
#                           Static system prompt                              #
###############################################################################
FINANCIAL_SYSTEM_PROMPT = """You are an expert financial analyst AI. Your task is to predict future stock performance based on 20 years of financial data.
Respond in XML format, giving reasoning first.
YOU MUST ALWAYS RETURN A VALID XML DOCUMENT.
The XML document must contain the following elements:
<returns_1y> - predicted returns for the next year, one of: [bad, neutral, good]
<volatility_1y> - predicted volatility for the next year, one of: [low, medium, high]
<returns_5y> - predicted returns for the next five years, one of: [bad, neutral, good]
<volatility_5y> - predicted volatility for the next five years, one of: [low, medium, high]

This is the overall structure of your response:
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
#                    Prompt helpers (build + truncate)                        #
###############################################################################
def build_chat(msgs, tok, add_gen=True):
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen)
    out = ""
    for m in msgs:
        out += f"<|start_header_id|>{m['role']}<|end_header_id|>\n{m['content']}<|eot_id|>"
    if add_gen:
        out += "<|start_header_id|>assistant<|end_header_id|>\n"
    return out

def truncate(text, tok, max_len):
    ids = tok.encode(text, add_special_tokens=False)
    return text if len(ids) <= max_len else tok.decode(ids[-max_len:], skip_special_tokens=False)

def format_prompts(batch, tok, max_len):
    outs = []
    for txt, is_rl in zip(batch["prompt"], batch["is_grpo_task"]):
        msgs = [{"role": "system", "content": FINANCIAL_SYSTEM_PROMPT},
                {"role": "user",   "content": txt}] if is_rl else [{"role": "user", "content": txt}]
        outs.append(truncate(build_chat(msgs, tok, True), tok, max_len))
    return {"prompt": outs}

###############################################################################
#                    Safe wrappers around model.generate                      #
###############################################################################
def make_safe_generate(model, tok):
    original = model.generate
    def safe(self, input_ids, attention_mask=None, **kw):
        out = original(input_ids, attention_mask=attention_mask, **kw)
        if out.size(1) == input_ids.size(1):
            pad_id = tok.eos_token_id or tok.pad_token_id or 0
            pad = torch.full((out.size(0), 1), pad_id, dtype=out.dtype, device=out.device)
            out = torch.cat([out, pad], dim=1)
        return out
    model.generate = types.MethodType(safe, model)

###############################################################################
#                    Remote vLLM monkey-patch (plus safety)                   #
###############################################################################
def attach_remote_generate(model, tok, url, max_new):
    url = url.rstrip("/") + "/generate"

    def remote(self, input_ids, attention_mask=None, **kw):
        payload = {
            "prompts": tok.batch_decode(input_ids, skip_special_tokens=False),
            "max_tokens": kw.get("max_new_tokens", max_new),
        }
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()

        # vLLM returns either "completion_ids" or OpenAI-style "outputs"
        comp_ids = data.get("completion_ids") \
                   or [o["token_ids"] for o in data["outputs"]]

        # --- NEW: pretty-print the decoded completions ----------------------
        #for i, ids in enumerate(comp_ids):
        #    text = tok.decode(ids, skip_special_tokens=False)
        #    print(f"[vLLM completion {i}] {text[300:]}")   # last 300 chars
        # --------------------------------------------------------------------

        if not comp_ids:
            comp_ids = [[tok.eos_token_id]]
        tensors = [torch.tensor(seq or [tok.eos_token_id],
                                device=input_ids.device) for seq in comp_ids]
        comp_pad = rnn_utils.pad_sequence(
            tensors, batch_first=True, padding_value=tok.pad_token_id)
        return torch.cat([input_ids, comp_pad], dim=1)

    model.generate = types.MethodType(remote, model)
    make_safe_generate(model, tok)

###############################################################################
#                                   Main                                      #
###############################################################################
def main(args):
    if args.max_seq_length < args.max_prompt_length + args.max_completion_length:
        raise ValueError("max_seq_length must be >= prompt + completion length")

    accelerator = Accelerator()
    if accelerator.is_main_process and (token := os.getenv("HF_TOKEN")):
        login(token)

    dtype = torch.bfloat16 if (torch.cuda.is_available() or torch.backends.hip.is_available()) else torch.float32

    # -------- Load base model --------
    model, tok = FastLanguageModel.from_pretrained(
        args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        dtype=dtype,
        token=os.getenv("HF_TOKEN"),
        fast_inference=False,
        device_map={"": accelerator.device},
        gpu_memory_utilization=0.6,
    )

    # -------- Add LoRA adapter --------
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # -------- KEEP adapter active even inside Unsloth's context manager -----
    from transformers import LlamaForCausalLM
    LlamaForCausalLM.disable_adapter  = lambda *a, **k: nullcontext()
    LlamaForCausalLM.disable_adapters = LlamaForCausalLM.disable_adapter
    model.disable_adapter  = lambda *a, **k: nullcontext()
    model.disable_adapters = model.disable_adapter
    # ------------------------------------------------------------------------

    # -------- Patch generation to remote vLLM --------
    attach_remote_generate(model, tok, args.vllm_remote_url, args.max_completion_length)

    # -------- Data --------
    train_ds = load_financial_dataset(split="train", test_size=args.test_split_size,
                                      k_folds=args.k_folds, fold_index=args.fold_index)
    eval_ds  = load_financial_dataset(split="test",  test_size=args.test_split_size,
                                      k_folds=args.k_folds, fold_index=args.fold_index)
    train_ds = train_ds.map(lambda b: format_prompts(b, tok, args.max_prompt_length),
                            batched=True)

    # -------- Trainer --------
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
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        reward_funcs=[financial_reward_function],
        tokenizer=tok,
        callbacks=[PeriodicEvalCallback(tokenizer=tok, eval_dataset=eval_ds,
                                        eval_steps=args.eval_steps)],
    )

    if accelerator.is_main_process:
        print("Starting GRPO training")
    trainer.train()
    if accelerator.is_main_process:
        print("Training finished, saving adapter")
    trainer.save_model()

###############################################################################
#                                   CLI                                       #
###############################################################################
if __name__ == "__main__":
    DEFAULTS = dict(
        model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
        output_dir="outputs_8b_grpo",
        max_seq_length=2048+512,
        max_prompt_length=2048,
        max_completion_length=512,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=DEFAULTS["model_name"])
    parser.add_argument("--output_dir", default=DEFAULTS["output_dir"])
    parser.add_argument("--max_seq_length", type=int, default=DEFAULTS["max_seq_length"])
    parser.add_argument("--lora_rank",  type=int, default=32)
    parser.add_argument("--num_generations", type=int, default=7)  # must divide dp_size
    parser.add_argument("--max_prompt_length", type=int, default=DEFAULTS["max_prompt_length"])
    parser.add_argument("--max_completion_length", type=int, default=DEFAULTS["max_completion_length"])
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps",  type=int, default=50)
    parser.add_argument("--max_steps",     type=int, default=1000)
    parser.add_argument("--save_steps",    type=int, default=100)
    parser.add_argument("--eval_steps",    type=int, default=200)
    parser.add_argument("--report_to", choices=["wandb","none"], default="none")
    parser.add_argument("--wandb_project", default="unsloth_grpo")
    parser.add_argument("--test_split_size", type=float, default=0.2)
    parser.add_argument("--k_folds", type=int, default=1)
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--vllm_remote_url", default="http://localhost:8000")
    main(parser.parse_args())
