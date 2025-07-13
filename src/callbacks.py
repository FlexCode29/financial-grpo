from transformers import TrainerCallback, TrainerState, TrainerControl
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
import wandb # Import wandb to log metrics

from src.reward import _parse_prediction
#from train_grpo import FINANCIAL_SYSTEM_PROMPT
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

class PeriodicEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, eval_steps=100, num_samples=32):
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        # Take a small, fixed sample from the eval dataset to run quick evaluations
        # We only need the GRPO-task examples for this evaluation
        grpo_eval_dataset = eval_dataset.filter(lambda example: example['is_grpo_task'])
        self.sample_dataset = grpo_eval_dataset.shuffle(seed=42).select(range(min(num_samples, len(grpo_eval_dataset))))

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Check if it's time to evaluate based on the global step
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            print(f"\n--- Running periodic evaluation at step {state.global_step} ---")
            model = kwargs['model']
            
            results = []
            for example in self.sample_dataset:
                messages = [{"role": "system", "content": FINANCIAL_SYSTEM_PROMPT}, {"role": "user", "content": example['prompt']}]
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
                completion = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

                results.append({
                    "gt_ret_1y": example['ground_truth_returns_1y'],
                    "pred_ret_1y": _parse_prediction(completion, "returns_1y"),
                })
            
            results_df = pd.DataFrame(results).dropna()
            if not results_df.empty:
                acc = accuracy_score(results_df['gt_ret_1y'], results_df['pred_ret_1y'])
                # Log this metric to W&B under a specific "eval" namespace
                if 'wandb' in args.report_to:
                    wandb.log({"eval/periodic_accuracy": acc, "step": state.global_step})
                print(f"Periodic eval accuracy on {len(results_df)} samples: {acc:.4f}")
            else:
                print("Periodic eval: No valid predictions were generated in the sample.")