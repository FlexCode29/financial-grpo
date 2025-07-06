#!/bin/bash
# --- Core Parameters ---

# The model to train. Can be overridden by passing an argument to the script.
# Example: ./run.sh "unsloth/Qwen2-7B-Instruct-GGUF"
MODEL_NAME=${1:-"unsloth/Qwen2-1.5B-Instruct-GGUF"}

# --- Key Hyperparameters ---
# These are the main "knobs" to turn for experiments.
MAX_STEPS=500
LEARNING_RATE=5e-6
LORA_RANK=16
NUM_GENERATIONS=4 # Number of completions per prompt for GRPO

# =================================================================

# --- Sanity Checks ---
# Ensure the W&B API key is provided, otherwise the run will fail to log.
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY environment variable is not set."
    echo "Please set it with -e WANDB_API_KEY='your_key' in your docker run command."
    exit 1
fi

echo "--- Starting Training ---"
echo "Model to train: $MODEL_NAME"
echo "Max Steps: $MAX_STEPS"
echo "Learning Rate: $LEARNING_RATE"
echo "LoRA Rank: $LORA_RANK"
echo "Logging to W&B"
echo "---------------------------"

# --- Launch Training ---
# Use 'accelerate launch' to handle multi-GPU training correctly.
# It passes our variables to the command-line arguments of train_grpo.py.
accelerate launch train_grpo.py \
    --model_name "$MODEL_NAME" \
    --output_dir "/app/outputs/$MODEL_NAME" \
    --report_to "wandb" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate $LEARNING_RATE \
    --max_steps $MAX_STEPS \
    --lora_rank $LORA_RANK \
    --num_generations $NUM_GENERATIONS