#!/usr/bin/env bash
###############################################################################
# run.sh — Accelerate‑based wrapper for GRPO fine‑tuning with Unsloth
#
# Usage
#   ./run.sh <model_name_or_path> [hf_repo_id] [MODE]
#     model_name_or_path  (required)  local dir or HF repo slug
#     hf_repo_id          (optional)  target repo for FINAL_TRAIN push
#     MODE                (optional)  CV | FINAL_TRAIN    (default: CV)
#
# Environment
#   export HF_TOKEN and WANDB_API_KEY  (or put them in .env.local).
#   accelerate config   # run once; choose LOCAL_MACHINE, bf16, 1 process.
###############################################################################
set -euo pipefail
[[ -f ".env.local" ]] && source ".env.local"

MODEL_NAME="${1:-}"
HF_REPO_ID="${2:-}"
MODE="${3:-CV}"

if [[ -z "$MODEL_NAME" ]]; then
  echo "USAGE: ./run.sh <model_name_or_path> [hf_repo_id] [MODE]"
  exit 1
fi
if [[ "$MODE" == "FINAL_TRAIN" && -z "$HF_REPO_ID" ]]; then
  echo "FINAL_TRAIN selected but no HF repo id supplied."
  exit 1
fi
: "${HF_TOKEN?Need HF_TOKEN}"  : "${WANDB_API_KEY?Need WANDB_API_KEY}"

# -------- ROCm fragmentation help -------------------------------------------
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# -------- hyper‑parameters ---------------------------------------------------
K_FOLDS=5
MAX_STEPS=800
LEARNING_RATE=8e-6
LORA_RANK=16
NUM_GENERATIONS=2
PER_DEVICE_BATCH_SIZE=1
GRAD_ACC_STEPS=4
MAX_PROMPT=4096
MAX_COMP=512
WANDB_PROJECT="deepseek-r1-70b-grpo"

TRAIN_SCRIPT="train_grpo.py"

launch() {
  accelerate launch --num_processes 1 "$TRAIN_SCRIPT" \
    --model_name "$MODEL_NAME"             \
    --output_dir "$OUTPUT_DIR"             \
    --learning_rate "$LEARNING_RATE"       \
    --max_steps "$MAX_STEPS"               \
    --lora_rank "$LORA_RANK"               \
    --num_generations "$NUM_GENERATIONS"   \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS"        \
    --max_prompt_length "$MAX_PROMPT"      \
    --max_completion_length "$MAX_COMP"    \
    --device_map balanced                  \
    --report_to wandb                      \
    --wandb_project "$WANDB_PROJECT"       \
    "$@"
}

# -------- main logic ---------------------------------------------------------
if [[ "$MODE" == "CV" ]]; then
  for FOLD in $(seq 0 $((K_FOLDS-1))); do
    OUTPUT_DIR="./outputs/cv_fold_${FOLD}"
    launch --k_folds "$K_FOLDS" --fold_index "$FOLD"

    python evaluate.py \
      --model_path "$OUTPUT_DIR" --base_model "$MODEL_NAME" \
      --k_folds "$K_FOLDS" --fold_index "$FOLD"             \
      --report_to wandb --wandb_project "$WANDB_PROJECT" \
      --run_name "eval_fold_${FOLD}_of_${K_FOLDS}"
  done
  python aggregate_results.py --base_dir "./outputs" --k_folds "$K_FOLDS"

elif [[ "$MODE" == "FINAL_TRAIN" ]]; then
  OUTPUT_DIR="./outputs/final_model"
  launch --k_folds 1 --push_to_hub --hf_repo_id "$HF_REPO_ID"

  python evaluate.py \
    --model_path "$OUTPUT_DIR" --base_model "$MODEL_NAME" \
    --k_folds 1 --report_to wandb --wandb_project "$WANDB_PROJECT" \
    --run_name "final_model_eval"
else
  echo "MODE must be CV or FINAL_TRAIN"; exit 1
fi

echo "==== run.sh finished OK ===="
