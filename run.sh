#!/bin/bash

MODE="CV"

MODEL_NAME=${1}
HF_REPO_ID=${2}

# --- CROSS-VALIDATION PARAMETERS (used in CV mode) ---
K_FOLDS=5

# --- KEY TRAINING HYPERPARAMETERS (Tunable) ---
MAX_STEPS=500
LEARNING_RATE=5e-6
LORA_RANK=16
NUM_GENERATIONS=4
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4

# --- LOGGING PARAMETERS ---
WANDB_PROJECT="financial-grpo-llm"

# =================================================================
# ======================== END OF CONFIG ========================
# =================================================================

# --- Sanity Checks ---
if [ -z "$MODEL_NAME" ]; then
    echo "Error: You must provide a model name as the first argument."
    echo "Usage: ./run.sh <model_name> [hf_repo_id]"
    exit 1
fi
if [ "$MODE" = "FINAL_TRAIN" ] && [ -z "$HF_REPO_ID" ]; then
    echo "Error: MODE is FINAL_TRAIN, but no Hugging Face repo ID was provided as the second argument."
    exit 1
fi
if [ -z "$WANDB_API_KEY" ] || [ -z "$HF_TOKEN" ]; then
    echo "Error: WANDB_API_KEY and/or HF_TOKEN not found in .env.local file."
    exit 1
fi

# --- Main Logic ---
if [ "$MODE" = "CV" ]; then
    # --- WORKFLOW 1: K-Fold Cross-Validation ---
    echo "--- Starting K-Fold Cross-Validation ---"
    echo "Model: $MODEL_NAME, Folds: $K_FOLDS"
    echo "----------------------------------------"

    for i in $(seq 0 $((K_FOLDS - 1))); do
        FOLD_INDEX=$i
        OUTPUT_DIR="/app/outputs/cv_fold_${FOLD_INDEX}"
        echo -e "\n\n=============== RUNNING FOLD ${FOLD_INDEX} ================"
        
        # 1. Train on the current fold
        echo "--- Training Fold ${FOLD_INDEX} ---"
        accelerate launch train_grpo.py \
            --model_name "$MODEL_NAME" \
            --output_dir "$OUTPUT_DIR" \
            --learning_rate $LEARNING_RATE \
            --max_steps $MAX_STEPS \
            --lora_rank $LORA_RANK \
            --num_generations $NUM_GENERATIONS \
            --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
            --k_folds $K_FOLDS \
            --fold_index $FOLD_INDEX \
            --report_to "wandb" \
            --wandb_project "$WANDB_PROJECT"

        # 2. Evaluate on the current fold's test set
        echo "--- Evaluating Fold ${FOLD_INDEX} ---"
        python evaluate.py \
            --model_path "$OUTPUT_DIR" \
            --base_model "$MODEL_NAME" \
            --k_folds $K_FOLDS \
            --fold_index $FOLD_INDEX \
            --report_to "wandb" \
            --wandb_project "$WANDB_PROJECT" \
            --run_name "eval_fold_${FOLD_INDEX}_of_${K_FOLDS}"
    done

    # 3. Aggregate all results
    echo -e "\n\n--- Aggregating All Cross-Validation Results ---"
    python aggregate_results.py \
        --base_dir "/app/outputs" \
        --k_folds $K_FOLDS

elif [ "$MODE" = "FINAL_TRAIN" ]; then
    # --- WORKFLOW 2: Final Training and Hugging Face Upload ---
    echo "--- Starting Final Training Run for Hugging Face Upload ---"
    echo "Model: $MODEL_NAME, Repo: $HF_REPO_ID"
    echo "----------------------------------------------------------"
    
    OUTPUT_DIR="/app/outputs/final_model"

    # Train on the full dataset (k_folds=1 uses the default train/test split)
    accelerate launch train_grpo.py \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --learning_rate $LEARNING_RATE \
        --max_steps $MAX_STEPS \
        --lora_rank $LORA_RANK \
        --num_generations $NUM_GENERATIONS \
        --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --k_folds 1 \
        --report_to "wandb" \
        --wandb_project "$WANDB_PROJECT" \
        --push_to_hub \
        --hf_repo_id "$HF_REPO_ID"

    # Optionally, run a final evaluation on the held-out test set
    echo "--- Evaluating Final Model ---"
    python evaluate.py \
        --model_path "$OUTPUT_DIR" \
        --base_model "$MODEL_NAME" \
        --k_folds 1 \
        --report_to "wandb" \
        --wandb_project "$WANDB_PROJECT" \
        --run_name "final_model_eval"

else
    echo "Error: Invalid MODE set. Choose 'CV' or 'FINAL_TRAIN'."
    exit 1
fi

echo "--- Script Finished ---"