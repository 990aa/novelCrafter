import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

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

if HF_TOKEN:
    os.system(f"huggingface-cli login --token {HF_TOKEN}")
else:
    print("HF_TOKEN not found in environment variables.")

if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
else:
    print("WANDB_API_KEY not found in environment variables.")

# ------------------- SETUP COMPLETE -------------------

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

book_text = extract_text_from_pdf("/input/cocodile/The_crocodile.pdf")
print(f"Extracted text length: {len(book_text)} characters")

