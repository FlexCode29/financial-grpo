from transformers import TrainerCallback, TrainerState, TrainerControl
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
import wandb # Import wandb to log metrics


class PeriodicEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, eval_steps=100, num_samples=32):
        print("init")

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Check if it's time to evaluate based on the global step
        print("step")