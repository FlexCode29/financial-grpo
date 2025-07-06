#!/bin/bash

# This script launches the evaluation job inside the Docker container.
# It requires the path to the trained model adapters as an argument.

MODEL_PATH=${1}
RUN_NAME=${2:-"eval-$(date +%s)"} # Default run name is 'eval' + current timestamp

# --- Sanity Checks ---
if [ -z "$MODEL_PATH" ]; then
    echo "Error: You must provide the path to the trained model adapters."
    echo "Usage: ./evaluate.sh /path/to/your/model"
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY environment variable is not set."
    exit 1
fi

echo "--- Starting Evaluation ---"
echo "Evaluating model at: $MODEL_PATH"

# Launch the Python evaluation script
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --run_name "$RUN_NAME" \
    --report_to "wandb"