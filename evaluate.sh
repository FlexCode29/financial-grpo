#!/bin/bash

# --- USAGE ---
# ./evaluate.sh <path_to_model_adapters> <base_model_name> [wandb_run_name] [k_folds] [fold_index]
#
# --- EXAMPLES ---
# 1. Evaluate the final model:
#    ./evaluate.sh "/app/outputs/final_model" "unsloth/Qwen2-1.5B-Instruct-GGUF"
#
# 2. Evaluate a specific cross-validation fold:
#    ./evaluate.sh "/app/outputs/cv_fold_3" "unsloth/Qwen2-1.5B-Instruct-GGUF" "eval_of_fold_3" 5 3

# --- CORE PARAMETERS (Read from command line) ---
# The local path inside the container to the trained LoRA adapters. (Required)
MODEL_PATH=${1}

# The Hugging Face name of the base model the adapters were trained on. (Required)
BASE_MODEL=${2}

# --- OPTIONAL PARAMETERS ---
# A custom name for the Weights & Biases run.
RUN_NAME=${3:-"manual-eval-$(date +%s)"}

# The K-Fold parameters used during training (if evaluating a CV fold).
# For a model trained on the full dataset, these should be 1 and 0.
K_FOLDS=${4:-1}
FOLD_INDEX=${5:-0}

# --- LOGGING PARAMETERS ---
WANDB_PROJECT="financial-grpo-llm"

# =================================================================

# --- Sanity Checks ---
if [ -z "$MODEL_PATH" ] || [ -z "$BASE_MODEL" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: ./evaluate.sh <path_to_model_adapters> <base_model_name>"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found at '$MODEL_PATH'"
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY environment variable is not set."
    exit 1
fi

echo "--- Starting Standalone Evaluation ---"
echo "Model Adapters: $MODEL_PATH"
echo "Base Model:     $BASE_MODEL"
echo "W&B Run Name:   $RUN_NAME"
echo "K-Folds Mode:   Evaluating fold $FOLD_INDEX of $K_FOLDS"
echo "--------------------------------------"

# --- Launch Evaluation ---
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --k_folds $K_FOLDS \
    --fold_index $FOLD_INDEX \
    --report_to "wandb" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME"

echo "Evaluation Script Finished"