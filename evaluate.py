import wandb
import argparse
import torch
import pandas as pd
import numpy as np
from unsloth import FastLanguageModel
from sklearn.metrics import mean_squared_error, accuracy_score
from datasets import load_dataset

# === The Critical Connections ===
from src.data_loader import load_financial_dataset
from src.reward import _parse_prediction
from train_grpo import FINANCIAL_SYSTEM_PROMPT
# ==============================

# --- Mappings for Metric Calculation ---
BAND_TO_INT_MAPPING = {"returns": {"bad": 0, "neutral": 1, "good": 2}, "volatility": {"low": 0, "medium": 1, "high": 2}}

# --- Assumptions for VaR/CVaR Calculation ---
# To calculate VaR/CVaR from bands, we must assume a return distribution for each band.
# These are baseline assumptions and can be tuned.
BAND_TO_DISTRIBUTION = {
    "bad":     {"mean": -0.15, "std": 0.20}, # Assume -15% avg return, 20% std dev
    "neutral": {"mean":  0.05, "std": 0.15}, # Assume +5% avg return, 15% std dev
    "good":    {"mean":  0.20, "std": 0.25}, # Assume +20% avg return, 25% std dev
}

def calculate_var_cvar(predicted_bands, confidence_level=0.95):
    """Calculates VaR and CVaR based on a list of predicted return bands."""
    simulated_returns = []
    for band in predicted_bands:
        if band in BAND_TO_DISTRIBUTION:
            params = BAND_TO_DISTRIBUTION[band]
            # For each prediction, simulate 1000 possible outcomes based on the assumed distribution
            simulated_returns.extend(np.random.normal(params["mean"], params["std"], 1000))
    
    if not simulated_returns:
        return np.nan, np.nan

    returns_array = np.array(simulated_returns)
    # Value at Risk (VaR) is the percentile of the returns
    var = np.percentile(returns_array, 100 * (1 - confidence_level))
    # Conditional Value at Risk (CVaR) is the average of returns worse than VaR
    cvar = returns_array[returns_array <= var].mean()
    
    return var, cvar

def evaluate_fino_benchmark(model, tokenizer, wandb_run):
    """Evaluates the model on the Fino1 benchmark dataset."""
    print("\n--- Evaluating on Fino1 Benchmark ---")
    try:
        fino_ds = load_dataset("glarna/fino-demonstrate-reasoning", split="test")
        
        correct = 0
        results = []
        for i, item in enumerate(fino_ds):
            if i >= 100: break # Limit to first 100 examples for speed
            
            prompt = item['instruction'] + "\n" + item['input']
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
            prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split('[/INST]')[-1].strip()

            is_correct = prediction.lower() == item['output'].lower()
            if is_correct:
                correct += 1
            results.append({"prompt": prompt, "prediction": prediction, "answer": item['output'], "correct": is_correct})
        
        accuracy = correct / len(results)
        print(f"Fino1 Benchmark Accuracy: {accuracy:.2f}")
        if wandb_run:
            wandb_run.log({"fino1_accuracy": accuracy})
            wandb_run.log({"fino1_results": wandb.Table(dataframe=pd.DataFrame(results))})
    except Exception as e:
        print(f"Could not run Fino1 benchmark: {e}")


def main(args):
    """Main function to evaluate a trained model."""
    print("--- Starting Full Evaluation ---")
    wandb_run = None
    if args.report_to == 'wandb':
        wandb_run = wandb.init(project=args.wandb_project, name=f"eval_{args.run_name}", config=args)

    # --- 1. Load Model ---
    print(f"Loading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model, max_seq_length=2048, dtype=torch.bfloat16, load_in_4bit=True,
    )
    print(f"Loading and merging LoRA adapters from: {args.model_path}")
    model = FastLanguageModel.from_pretrained(model, args.model_path)

    # --- 2. Evaluate on our custom GRPO task ---
    print("\n--- Evaluating on Custom GRPO Task ---")
    test_dataset = load_financial_dataset(split='test', test_size=args.test_split_size)
    results = []
    for example in test_dataset:
        if not example['is_grpo_task']: continue

        messages = [{"role": "system", "content": FINANCIAL_SYSTEM_PROMPT}, {"role": "user", "content": example['prompt']}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
        completion = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        results.append({
            "gt_ret_1y": example['ground_truth_returns_1y'], "gt_vol_1y": example['ground_truth_volatility_1y'],
            "pred_ret_1y": _parse_prediction(completion, "returns_1y"), "pred_vol_1y": _parse_prediction(completion, "volatility_1y"),
        })

    results_df = pd.DataFrame(results)
    metrics = {}
    for band_type in ["returns", "volatility"]:
        gt_col, pred_col = f"gt_{'ret' if band_type == 'returns' else 'vol'}_1y", f"pred_{'ret' if band_type == 'returns' else 'vol'}_1y"
        eval_df = results_df.dropna(subset=[gt_col, pred_col])
        metrics[f"accuracy_{band_type}_1y"] = accuracy_score(eval_df[gt_col], eval_df[pred_col])
        gt_int, pred_int = eval_df[gt_col].map(BAND_TO_INT_MAPPING[band_type]), eval_df[pred_col].map(BAND_TO_INT_MAPPING[band_type])
        metrics[f"mse_{band_type}_1y"] = mean_squared_error(gt_int, pred_int)
    
    # Calculate VaR/CVaR from the predicted return bands
    valid_return_preds = results_df['pred_ret_1y'].dropna().tolist()
    var_95, cvar_95 = calculate_var_cvar(valid_return_preds)
    metrics["portfolio_VaR_95"] = var_95
    metrics["portfolio_CVaR_95"] = cvar_95

    print("\n--- Custom Task Evaluation Metrics ---")
    print(metrics)
    if wandb_run:
        wandb_run.log(metrics)
        wandb_run.log({"custom_task_results": wandb.Table(dataframe=results_df)})

    # --- 3. Evaluate on Fino1 Benchmark ---
    evaluate_fino_benchmark(model, tokenizer, wandb_run)

    if wandb_run: wandb_run.finish()
    print("\n--- Evaluation Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned financial LLM.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved LoRA adapters.")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen2-1.5B-Instruct-GGUF", help="Base model.")
    parser.add_argument("--test_split_size", type=float, default=0.2)
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])
    parser.add_argument("--wandb_project", type=str, default="financial-grpo-llm")
    parser.add_argument("--run_name", type=str, default="unnamed-run")
    args = parser.parse_args()
    main(args)