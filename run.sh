#!/bin/bash

# Launch the training script using Hugging Face Accelerate
accelerate launch --num_processes 8 train_easy.py