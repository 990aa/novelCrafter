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
    os.system(f"hf auth login --token {HF_TOKEN}")
    print("✅ Logged in to Hugging Face")
else:
    print("⚠️ HF_TOKEN not found in environment variables.")

if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
    print("✅ Logged in to WandB")
else:
    print("⚠️ WANDB_API_KEY not found in environment variables.")

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
print(f"✅ Extracted text length: {len(book_text)} characters")
print("="*60)

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
print("\n" + "="*60)
print("MODEL SETUP")
print("="*60)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
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
    # Choose model based on device - 3B is too large for CPU training
    if device == "cuda":
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        print("Using CUDA - attempting to load 3B model")
    else:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        print("⚠️ Running on CPU - using smaller 1B model for better performance")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ Successfully loaded {model_name} tokenizer")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        if "3B" in model_name:
            print("Falling back to Llama-3.2-1B-Instruct tokenizer")
            model_name = "meta-llama/Llama-3.2-1B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}...")

    try:
        # Optimize loading for CPU
        load_kwargs = {
            "trust_remote_code": True,
        }
        
        if device == "cuda":
            load_kwargs["dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            # CPU-specific optimizations
            load_kwargs["dtype"] = torch.float32
            load_kwargs["low_cpu_mem_usage"] = True
            print("Using CPU optimizations: low_cpu_mem_usage=True")
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        print(f"✅ Successfully loaded {model_name}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        if "3B" in model_name:
            print("Falling back to Llama-3.2-1B-Instruct")
            model_name = "meta-llama/Llama-3.2-1B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            load_kwargs_fallback = {
                "trust_remote_code": True,
                "dtype": torch.float32,
                "low_cpu_mem_usage": True
            }
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs_fallback)
            print(f"✅ Successfully loaded {model_name}")
        else:
            raise

    model = prepare_model_for_kbit_training(model)
    print("Preparing LoRA configuration...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print("Applying LoRA to model...")
    model = get_peft_model(model, lora_config)
    print("\n" + "="*60)
    model.print_trainable_parameters()
    print("="*60)
    model.resize_token_embeddings(len(tokenizer))
    print(f"✅ Model setup complete!\n")

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
print(f"\n{'='*60}")
print(f"✅ Created {len(text_chunks)} training chunks")
print(f"{'='*60}")

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
    print(f"\n{'='*60}")
    print(f"--- Training on part {part_idx+1}/{num_parts} ---")
    print(f"{'='*60}")
    part_chunks = parts[part_idx]
    dataset = Dataset.from_dict({"text": part_chunks})
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.train_test_split(test_size=0.1)

    print(f"Training samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # Adjust steps based on dataset size
    train_size = len(dataset['train'])
    # Calculate steps per epoch
    steps_per_epoch = max(1, train_size // (1 * 8))  # batch_size * gradient_accumulation_steps
    # Set reasonable save/eval steps
    save_steps = max(10, steps_per_epoch // 2)
    eval_steps = max(10, steps_per_epoch // 2)
    
    print(f"Steps per epoch: ~{steps_per_epoch}")
    print(f"Save/Eval every {save_steps} steps")

    training_args = TrainingArguments(
        output_dir=f"./book_model_part_{part_idx+1}",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=min(100, steps_per_epoch),
        logging_steps=max(1, steps_per_epoch // 4),
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        load_best_model_at_end=False,  # Disable to avoid issues with small datasets
        fp16=torch.cuda.is_available(),
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=2,
        prediction_loss_only=False,
        report_to=["wandb"] if WANDB_API_KEY else ["none"],
        logging_dir=f"./logs_part_{part_idx+1}",
        dataloader_pin_memory=False,  # Fix the pin_memory warning on CPU
        use_cpu=not torch.cuda.is_available(),  # Explicitly use CPU if no CUDA
        no_cuda=not torch.cuda.is_available(),  # Disable CUDA if not available
        disable_tqdm=False,  # Keep progress bars enabled
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print(f"Starting training for part {part_idx+1}...")
    print(f"Device: {device}")
    print(f"Model device: {next(model.parameters()).device}")
    
    try:
        trainer.train()
        print(f"✅ Training completed for part {part_idx+1}")
    except Exception as e:
        print(f"❌ Training failed for part {part_idx+1}: {e}")
        import traceback
        traceback.print_exc()
        break

    # Save model and tokenizer
    print(f"Saving model for part {part_idx+1}...")
    trainer.save_model()
    tokenizer.save_pretrained(f"./book_model_part_{part_idx+1}")
    print(f"✅ Model saved locally")

    # Upload to Hugging Face Hub
    try:
        print(f"Uploading to Hugging Face Hub...")
        model.push_to_hub("a-01a/novelCrafter", commit_message=f"Trained on part {part_idx+1}")
        tokenizer.push_to_hub("a-01a/novelCrafter", commit_message=f"Trained on part {part_idx+1}")
        print(f"✅ Model uploaded to HuggingFace")
    except Exception as e:
        print(f"⚠️ Failed to upload to HuggingFace: {e}")
        print("Continuing with local training...")

    # Update progress
    with open(progress_file, "w") as f:
        json.dump({"last_completed_part": part_idx+1}, f)
    print(f"Progress saved: {part_idx+1}/{num_parts} parts completed")

    # Ask user if should continue
    if part_idx < num_parts - 1:  # Don't ask on the last part
        print(f"\n{'='*60}")
        user_input = input(f"Continue to next part? (y/n): ").strip().lower()
        if user_input != "y":
            print("Training stopped by user. Model saved and uploaded.")
            break
    else:
        print(f"✅ All {num_parts} parts completed!")

print("\n{'='*60}")
print("✅ Incremental training complete!")
print(f"Final model saved in: ./book_model_part_{part_idx+1}")
print("="*60)

