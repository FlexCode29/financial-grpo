# Financial GRPO LLM Fine-Tuning Pipeline

## 1. Project Overview

This project provides a complete, containerized pipeline for fine-tuning a Large Language Model (LLM) on a complex financial prediction task using Group Relative Policy Optimization (GRPO). The primary goal is to train a model that can analyze 20 years of a company's financial history and supplementary data to solve a dual objective:

1.  **Return Prediction:** Predict the stock's return band (e.g., "good", "neutral", "bad") over 1 and 5-year horizons.
2.  **Volatility Prediction:** Predict the stock's volatility band (e.g., "high", "medium", "low") over the same horizons.

The pipeline is built to be scalable and efficient, leveraging high-performance tools and running on AMD's ROCm software stack.

### Core Technologies
*   **Containerization:** Docker
*   **GPU Acceleration:** AMD ROCm
*   **ML Framework:** PyTorch
*   **RL Library:** Hugging Face TRL (Transformer Reinforcement Learning)
*   **Model Optimization:** Unsloth
*   **Inference Acceleration:** vLLM
*   **Experiment Tracking:** Weights & Biases (W&B)

## 2. Prerequisites

Log in: 

sudo docker run   --rm -it   --device=/dev/kfd   --device=/dev/dri   --ipc=host   --shm-size=16g   --env-file ./.env.local   --entrypoint /bin/bash   financial-grpo

Install torch first:

/opt/conda/bin/python -m pip install --no-cache-dir --pre torch torchvision  --index-url https://download.pytorch.org/whl/nightly/rocm6.2/

This pipeline is designed for a specific hardware and software environment. Before you begin, ensure you have the following:

1.  **A Linux Host Machine:** The Docker container must be run on a Linux OS. It will not work on a Windows or macOS host due to driver limitations.
2.  **AMD GPU with ROCm Drivers:** The host machine must have a ROCm-compatible AMD GPU (e.g., MI250x) and the corresponding ROCm drivers installed.
3.  **Docker Engine:** The Docker daemon must be installed and running.
4.  **Git:** For cloning the project repository.
5.  **Accounts:** You will need accounts for:
    *   [Weights & Biases](https://wandb.ai/) for logging results.
    *   [Hugging Face](https://huggingface.co/) for accessing base models.

## 3. Setup Instructions

Follow these steps on your Linux host machine to set up the project.

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd financial-grpo
```

### Step 2: Configure Secrets

This project requires API keys for W&B and Hugging Face. These are managed securely using a local environment file that is **never** committed to Git.

1.  Create a file named `.env.local` in the root of the project directory:
    ```bash
    touch .env.local
    ```

2.  Open the `.env.local` file and add your secrets in the following format:
    ```
    # .env.local
    # This file stores all secrets for the project.

    WANDB_API_KEY="your_actual_wandb_key_goes_here"
    HF_TOKEN="hf_YourHuggingFaceTokenGoesHere"
    ```

The project's `.gitignore` file is already configured to prevent this file from being tracked by Git.

### Step 3: Build the Docker Image

This command builds the self-contained Docker image with all necessary dependencies specified in the `Dockerfile`. This process may take several minutes the first time.

```bash
docker build -t financial-grpo .
```

You are now ready to run the training and evaluation pipeline.

## 4. Running the Pipeline

All commands should be run from the root of the `financial-grpo` project directory on the Linux host machine.

### Step 1: Run Training

This command starts the training process. It mounts the local `./outputs` directory into the container to save the final model adapters.

```bash
docker run --rm -it --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size=16g --env-file ./.env.local -v "$(pwd)/outputs:/app/outputs" financial-grpo ./run.sh
```
*   **To train a different model**, pass it as an argument to the script:
    ```bash
    docker run ... financial-grpo ./run.sh "unsloth/Qwen2-7B-Instruct-GGUF"
    ```

### Step 2: Run Evaluation

After the training command has finished, run this command to evaluate the performance of your newly trained model.

```bash
docker run --rm -it --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size=16g --env-file ./.env.local -v "$(pwd)/outputs:/app/outputs" financial-grpo ./evaluate.sh "/app/outputs/unsloth/Qwen2-1.5B-Instruct-GGUF"
```
*   **Note:** Make sure the path passed to `evaluate.sh` matches the model name that was just trained.

This script will output performance metrics (Accuracy, MSE, VaR, CVaR) and benchmark results to the console and log them to your W&B project.

## 5. Project Structure

```
/financial-grpo
|-- .env.local              # (You create this) Stores secret API keys.
|-- .gitignore              # Ensures secrets and outputs are not committed.
|-- Dockerfile              # Blueprint for building the reproducible Docker environment.
|-- README.md               # This file.
|-- requirements.txt        # Python package dependencies.
|-- run.sh                  # Entrypoint script to launch training.
|-- evaluate.sh             # Entrypoint script to launch evaluation.
|-- train_grpo.py           # The main training script.
|-- evaluate.py             # The main evaluation script.
|-- src/
|   |-- data_loader.py      # Handles loading, merging, and processing all datasets.
|   |-- reward.py           # Defines the custom financial reward function for GRPO.
|-- data/                   # (Optional) For storing small, local data samples for offline testing.
|-- outputs/                # (Created on run) Where the trained model adapters are saved.
```

## 6. Customization

The easiest way to run experiments is to modify the `run.sh` script. You can change hyperparameters like `MAX_STEPS`, `LEARNING_RATE`, and `LORA_RANK` in the "Key Hyperparameters" section at the top of the file.

## 7. CPU-Only Smoke Testing

If you are on a machine without a compatible AMD GPU (like Windows/WSL) and want to verify that the code runs without crashing, you can perform a CPU-only smoke test. This test is **extremely slow** and is only for validating code integrity.

1.  Ensure the `train_grpo.py` script contains the CPU fallback logic.
2.  Run the following `docker run` command from your WSL terminal (note the **absence** of the `--device` flags):
    ```bash
    docker run --rm -it --ipc=host --shm-size=16g --env-file ./.env.local -v "$(pwd)/outputs:/app/outputs" financial-grpo ./run.sh
    ```
Wait for the first few training steps to log to the console to confirm the pipeline is working, then stop the container with `Ctrl+C`.