import os
import re
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
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from peft.utils import prepare_model_for_kbit_training

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

book_text = extract_text_from_pdf("input/The_crocodile.pdf")
print(f"Extracted text length: {len(book_text)} characters")

# ------------------- DATA PREPARATION -------------------

def clean_and_chunk_text(text, chunk_size=10):
    """
    Clean text and split into chunks of sentences
    chunk_size: number of sentences per training example
    """
    # Clean text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'\n', ' ', text)   # Replace newlines with spaces
    
    # Split into sentences (simple approach)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Create chunks of sentences
    chunks = []
    for i in range(0, len(sentences) - chunk_size, chunk_size):
        chunk = " ".join(sentences[i:i+chunk_size])
        chunks.append(chunk)
    
    return chunks





# --- Model and Tokenizer Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
hf_repo = "a-01a/novelCrafter"
tokenizer = None
model = None

# Try to load from HF repo if available (resume training)
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoConfig

def hf_model_exists(repo_id):
    try:
        files = list_repo_files(repo_id)
        return any(f.endswith("pytorch_model.bin") or f.endswith("adapter_model.bin") for f in files)
    except Exception:
        return False

if hf_model_exists(hf_repo):
    print(f"Resuming from Hugging Face repo: {hf_repo}")
    tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    try:
        # Try to load as PEFT model
        model = AutoModelForCausalLM.from_pretrained(hf_repo, trust_remote_code=True)
        # If adapter weights exist, load as PEFT
        if os.path.exists(os.path.join(hf_repo, "adapter_config.json")):
            model = PeftModel.from_pretrained(model, hf_repo)
        print("Loaded model from Hugging Face repo.")
    except Exception as e:
        print(f"Error loading model from HF repo: {e}")
        # fallback to base
        model = None
        tokenizer = None

#Models 

if tokenizer is None or model is None:
    try:
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Successfully loaded Llama-3.2-3B-Instruct tokenizer")
    except Exception as e:
        print(f"Error loading distill model tokenizer: {e}")
        print("Falling back to Llama-3.2-1B-Instruct tokenizer")
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Using model: {model_name}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        print(f"Successfully loaded {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to Llama-3.2-1B-Instruct")
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# --- Incremental Training Setup ---
import json
from datasets import Dataset

def tokenize_function(examples):
    """Tokenize the text chunks"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=1024,
        return_tensors=None
    )

chunk_size = 10
text_chunks = clean_and_chunk_text(book_text, chunk_size=chunk_size)
print(f"Created {len(text_chunks)} training chunks")

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

num_parts = 10
parts = split_list(text_chunks, num_parts)

progress_file = "train_progress.json"
if os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        progress = json.load(f)
    start_part = progress.get("last_completed_part", 0)
else:
    start_part = 0

for part_idx in range(start_part, num_parts):
    print(f"\n--- Training on part {part_idx+1}/{num_parts} ---")
    part_chunks = parts[part_idx]
    dataset = Dataset.from_dict({"text": part_chunks})
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.train_test_split(test_size=0.1)

    print(f"Training samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")

    training_args = TrainingArguments(
        output_dir=f"./book_model_part_{part_idx+1}",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=2,
        prediction_loss_only=False,
        report_to=["none"],
        logging_dir=f"./logs_part_{part_idx+1}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(f"./book_model_part_{part_idx+1}")

    # Upload to Hugging Face Hub
    model.push_to_hub("a-01a/novelCrafter", commit_message=f"Trained on part {part_idx+1}")
    tokenizer.push_to_hub("a-01a/novelCrafter", commit_message=f"Trained on part {part_idx+1}")

    # Update progress
    with open(progress_file, "w") as f:
        json.dump({"last_completed_part": part_idx+1}, f)

    # Ask user if should continue
    user_input = input(f"Continue to next part? (y/n): ").strip().lower()
    if user_input != "y":
        print("Training stopped by user. Model saved and uploaded.")
        break

print("✅ Incremental training complete!")

