import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Install and import required packages
# (Assume requirements.txt is used for installation)

import torch
import wandb
import PyPDF2
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Login to Hugging Face
if HF_TOKEN:
    os.system(f"huggingface-cli login --token {HF_TOKEN}")
else:
    print("HF_TOKEN not found in environment variables.")

# Login to WandB
if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
else:
    print("WANDB_API_KEY not found in environment variables.")

# Your main code logic goes here
