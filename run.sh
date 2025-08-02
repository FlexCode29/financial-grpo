#!/usr/bin/env bash
###############################################################################
# run.sh – unified launcher for vLLM server or GRPO training
#
# New in this version
# -------------------
# • --gpus flag (or GPU_LIST env) sets HIP_VISIBLE_DEVICES automatically.
# • Defaults: server → GPU 0,   train → all visible GPUs.
#
# Examples
# --------
#   # vLLM on GPU 0 (default)
#   ./run.sh server
#
#   # vLLM on GPU 3
#   ./run.sh server --gpus 3
#
#   # Training on GPUs 1-7
#   ./run.sh train --gpus 1,2,3,4,5,6,7 \
#        --use_remote_vllm --vllm_remote_url http://localhost:8000
#
#   # You can also export GPU_LIST
#   export GPU_LIST=0
#   ./run.sh server
###############################################################################
set -euo pipefail

# -------- helper: parse --gpus flag -----------------------------------------
GPU_LIST="${GPU_LIST:-}"   # optional env override
ROLE="${1:-help}"          # server | train | help
shift || true



# -------- sensible defaults --------------------------------------------------
if [[ -z "$GPU_LIST" ]]; then
  if [[ "$ROLE" == "server" ]]; then
    GPU_LIST="0"                 # inference server default → GPU 0
  elif [[ "$ROLE" == "train" ]]; then
    GPU_LIST="1,2,3,4,5,6,7"  # training default → all visible GPUs
  fi
fi

export HIP_VISIBLE_DEVICES="$GPU_LIST"  # set HIP_VISIBLE_DEVICES for the launched process

# -------- model checkpoint override -----------------------------------------
MODEL_NAME="${MODEL_NAME:-meta-llama/meta-Llama-3.1-8B-Instruct}"

# -------- main switch --------------------------------------------------------
case "$ROLE" in
  server)
    echo ">>> Starting vLLM server on GPU(s): $GPU_LIST"
    HIP_VISIBLE_DEVICES=$GPU_LIST CUDA_VISIBLE_DEVICES=$GPU_LIST trl vllm-serve --model "$MODEL_NAME"
    ;;

  train)
    echo ">>> Starting GRPO training on GPU(s): $GPU_LIST"
    HIP_VISIBLE_DEVICES=$GPU_LIST CUDA_VISIBLE_DEVICES=$GPU_LIST accelerate launch --multi_gpu --num-processes=7 train_grpo.py --model_name "$MODEL_NAME" --report_to wandb
    ;;

  help | *)
    cat <<EOF
run.sh – GPU-aware wrapper for vLLM server / GRPO trainer

Usage:
  ./run.sh server [--gpus <list>] [extra vLLM args]
  ./run.sh train  [--gpus <list>] [extra train_grpo.py args]

--gpus LIST    Comma-separated GPU indices, e.g. "0" or "1,2,3".
               Overrides HIP_VISIBLE_DEVICES for the launched process.
               You can also pre-set GPU_LIST env instead of using --gpus.

ENV overrides:
  MODEL_NAME    HF checkpoint to load (default: $MODEL_NAME)
  GPU_LIST      Same as --gpus

Examples:
  ./run.sh server                 # GPU 0
  ./run.sh server --gpus 3        # GPU 3
  ./run.sh train  --gpus 1,2,3,4  # GPUs 1-4
EOF
    ;;
esac
