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

# Prepare training data
chunk_size = 10  # Input + output sentences
text_chunks = clean_and_chunk_text(book_text, chunk_size=chunk_size)
print(f"Created {len(text_chunks)} training chunks")


# Initialize tokenizer
model_name = "deepseek-ai/deepseek-llm-7b"  # Using the 7B version as fallback
try:
    # Try to load the distill model first
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Successfully loaded DeepSeek-R1-Distill-Llama-8B tokenizer")
except Exception as e:
    print(f"Error loading distill model tokenizer: {e}")
    print("Falling back to deepseek-llm-7b tokenizer")
    model_name = "deepseek-ai/deepseek-llm-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Using model: {model_name}")

def tokenize_function(examples):
    """Tokenize the text chunks"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=1024,
        return_tensors=None
    )

# Create dataset
dataset = Dataset.from_dict({"text": text_chunks})
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.train_test_split(test_size=0.1)

print(f"Training samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")

# Load model with error handling
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    print(f"Successfully loaded {model_name}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Falling back to deepseek-llm-7b")
    model_name = "deepseek-ai/deepseek-llm-7b"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

# Resize token embeddings
model.resize_token_embeddings(len(tokenizer))

# Data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling, not masked LM
)

# Training arguments - FIXED VERSION
from transformers import TrainingArguments

# Detect TPU
try:
    import torch_xla.core.xla_model as xm
    tpu = True
except ImportError:
    tpu = False

training_args = TrainingArguments(
    output_dir="./book_model",
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
    
    # Precision handling
    fp16=torch.cuda.is_available() and not tpu,  # only for GPU
    bf16=tpu,                                    # only for TPU
    
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    prediction_loss_only=False,
    report_to=["none"],        # disable wandb/tensorboard
    logging_dir="./logs",
)



# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
