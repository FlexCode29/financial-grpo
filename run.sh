#!/usr/bin/env bash
# --------------------------------------------------------------------
# run.sh – unified launcher for vLLM server or GRPO training
# Usage:
#   ./run.sh server [extra vllm args...]
#   ./run.sh train  [extra train_grpo.py args...]
#
# Examples
#   # Terminal 1 – vLLM on GPU 0
#   HIP_VISIBLE_DEVICES=0 ./run.sh server
#
#   # Terminal 2 – training on GPUs 1-7
#   export HIP_VISIBLE_DEVICES=1,2,3,4,5,6,7
#   ./run.sh train --use_vllm --vllm_device rocm:0 --report_to wandb
# --------------------------------------------------------------------

set -euo pipefail

ROLE=${1:-help}               # server | train | help
shift || true                 # shift off the role, leave the rest in "$@"

# default model used by both roles (override via env var)
MODEL_NAME=${MODEL_NAME:-"meta-llama/meta-Llama-3.1-8B-Instruct"}

case "$ROLE" in
  server)
    echo ">>> Starting vLLM server"
    trl vllm-serve --model "$MODEL_NAME"
    ;;

  train)
    echo ">>> Starting GRPO fine-tuning"
    accelerate launch train_grpo.py \
      --model_name "$MODEL_NAME" \
      "$@"
    ;;

  help | *)
    cat <<EOF
Usage:
  ./run.sh server [vLLM extra args]
  ./run.sh train  [train_grpo.py extra args]

Env overrides:
  MODEL_NAME=...   checkpoint to load (default: $MODEL_NAME)

Typical workflow:
  # terminal 1
  HIP_VISIBLE_DEVICES=0 ./run.sh server

  # terminal 2
  export HIP_VISIBLE_DEVICES=1,2,3,4,5,6,7
  ./run.sh train --use_vllm --vllm_device rocm:0
EOF
    ;;
esac
