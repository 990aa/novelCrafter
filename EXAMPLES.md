# 📚 Examples

This file contains example code snippets for common tasks with NovelCrafter.

## Table of Contents

1. [Basic Training](#basic-training)
2. [Loading and Using Models](#loading-and-using-models)
3. [Text Generation](#text-generation)
4. [Custom Processing](#custom-processing)
5. [Advanced Configuration](#advanced-configuration)

---

## Basic Training

### Training on a Single Book

```python
# main.py is already configured for this!
# Just place PDF in input/ and run:
python main.py
```

### Training with Custom Parameters

```python
# In main.py, modify these lines:

# Chunk configuration (line ~180)
chunk_size = 15      # More sentences per chunk
num_parts = 5        # Fewer parts = faster training

# LoRA configuration (line ~170)
lora_config = LoraConfig(
    r=16,            # Higher rank = more capacity
    lora_alpha=64,   # Typically 2-4x rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # More modules
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training configuration (line ~210)
training_args = TrainingArguments(
    num_train_epochs=5,                # More epochs
    per_device_train_batch_size=2,     # Larger batch
    learning_rate=2e-5,                # Lower learning rate
    # ... other args
)
```

---

## Loading and Using Models

### Load from Local Checkpoint

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load LoRA adapter from local checkpoint
model = PeftModel.from_pretrained(base_model, "./book_model_part_10")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./book_model_part_10")

print("✅ Model loaded from local checkpoint")
```

### Load from Hugging Face Hub

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map="auto"
)

# Load your fine-tuned adapter from HF Hub
model = PeftModel.from_pretrained(base_model, "your-username/your-model")
tokenizer = AutoTokenizer.from_pretrained("your-username/your-model")

print("✅ Model loaded from Hugging Face")
```

### Merge LoRA Weights (for faster inference)

```python
from peft import PeftModel

# Load model as before
model = PeftModel.from_pretrained(base_model, "path/to/adapter")

# Merge LoRA weights into base model
model = model.merge_and_unload()

# Now model can be used without PEFT overhead
model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")

print("✅ LoRA weights merged and saved")
```

---

## Text Generation

### Basic Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model (as shown above)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "your-username/your-model")
tokenizer = AutoTokenizer.from_pretrained("your-username/your-model")

# Generate
prompt = "Once upon a time, in a land far away,"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.8,
    do_sample=True
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

### Creative Writing (High Temperature)

```python
# For creative, diverse text
outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=1.0,        # High creativity
    top_p=0.95,            # Nucleus sampling
    top_k=50,              # Top-k sampling
    do_sample=True,
    repetition_penalty=1.15
)
```

### Consistent Style (Low Temperature)

```python
# For consistent, focused text
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,        # Lower creativity
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1
)
```

### Streaming Generation

```python
from transformers import TextIteratorStreamer
from threading import Thread

# Setup streamer
streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

# Generate in thread
generation_kwargs = dict(
    **inputs,
    max_new_tokens=200,
    temperature=0.8,
    streamer=streamer
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Stream output
print("Generating: ", end="")
for text in streamer:
    print(text, end="", flush=True)
print()
```

### Batch Generation

```python
# Generate multiple outputs at once
prompts = [
    "Chapter 1: The Beginning",
    "The old house stood",
    "In the depths of the forest"
]

inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.8,
    num_return_sequences=1,  # Outputs per prompt
    do_sample=True
)

# Decode all outputs
for i, output in enumerate(outputs):
    text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"\n--- Output {i+1} ---")
    print(text)
```

---

## Custom Processing

### Extract Text from Multiple PDFs

```python
import PyPDF2
import os

def extract_all_pdfs(directory):
    """Extract text from all PDFs in directory"""
    all_text = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            print(f"Processing {filename}...")
            
            text = ""
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
            
            all_text.append({
                'filename': filename,
                'text': text,
                'length': len(text)
            })
    
    return all_text

# Usage
books = extract_all_pdfs("input/")
for book in books:
    print(f"{book['filename']}: {book['length']} characters")
```

### Custom Text Cleaning

```python
import re

def advanced_clean_text(text):
    """Advanced text cleaning"""
    
    # Remove page numbers
    text = re.sub(r'\n\d+\n', '\n', text)
    
    # Remove chapter headings if needed
    text = re.sub(r'Chapter \d+', '', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (keep basic punctuation)
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"]', '', text)
    
    # Fix common OCR errors
    text = text.replace('rn', 'm')  # Example
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text.strip()

# Usage
book_text = extract_text_from_pdf("input/book.pdf")
book_text = advanced_clean_text(book_text)
```

### Smart Chunking by Paragraphs

```python
def chunk_by_paragraphs(text, min_length=500, max_length=1000):
    """Chunk text by paragraphs with length constraints"""
    
    # Split by double newlines (paragraphs)
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If adding this paragraph exceeds max, save current chunk
        if len(current_chunk) + len(para) > max_length and len(current_chunk) >= min_length:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n\n" + para
    
    # Add remaining text
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# Usage
text_chunks = chunk_by_paragraphs(book_text)
print(f"Created {len(text_chunks)} chunks")
```

---

## Advanced Configuration

### Multi-Book Training Script

```python
import os
from main import extract_text_from_pdf, clean_and_chunk_text

books = [
    ("input/book1.pdf", "author-book1"),
    ("input/book2.pdf", "author-book2"),
]

for book_path, model_name in books:
    print(f"\n{'='*60}")
    print(f"Training on: {book_path}")
    print(f"{'='*60}\n")
    
    # Extract and process
    book_text = extract_text_from_pdf(book_path)
    text_chunks = clean_and_chunk_text(book_text)
    
    # Update HF repo name
    hf_repo = f"your-username/{model_name}"
    
    # ... continue with training
    # (Copy training loop from main.py)
```

### Custom Training Callback

```python
from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    """Custom callback for monitoring training"""
    
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"✅ Epoch {state.epoch} completed")
        print(f"   Loss: {state.log_history[-1].get('loss', 'N/A')}")
    
    def on_step_end(self, args, state, control, **kwargs):
        # Log every N steps
        if state.global_step % 10 == 0:
            print(f"Step {state.global_step}")

# Add to Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    processing_class=tokenizer,
    callbacks=[CustomCallback()]  # Add custom callback
)
```

### GPU Memory Optimization

```python
# For large models on limited GPU memory
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",                    # Automatic device mapping
    torch_dtype=torch.float16,           # Half precision
    load_in_8bit=True,                   # 8-bit quantization
    low_cpu_mem_usage=True,              # Reduce CPU memory
    offload_folder="offload",            # CPU offload directory
    offload_state_dict=True              # Offload state dict
)

# Or use 4-bit quantization (requires bitsandbytes)
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### Hyperparameter Search

```python
from transformers import TrainingArguments
import itertools

# Define search space
learning_rates = [1e-5, 5e-5, 1e-4]
lora_ranks = [4, 8, 16]

best_loss = float('inf')
best_config = None

for lr, rank in itertools.product(learning_rates, lora_ranks):
    print(f"\nTrying lr={lr}, rank={rank}")
    
    # Configure LoRA
    lora_config = LoraConfig(r=rank, lora_alpha=rank*4, ...)
    model = get_peft_model(base_model, lora_config)
    
    # Configure training
    training_args = TrainingArguments(learning_rate=lr, ...)
    
    # Train
    trainer = Trainer(model=model, args=training_args, ...)
    trainer.train()
    
    # Evaluate
    eval_loss = trainer.evaluate()['eval_loss']
    
    if eval_loss < best_loss:
        best_loss = eval_loss
        best_config = {'lr': lr, 'rank': rank}

print(f"Best config: {best_config}")
```

---

## Integration Examples

### Simple API with Flask

```python
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = Flask(__name__)

# Load model once at startup
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "your-username/your-model")
tokenizer = AutoTokenizer.from_pretrained("your-username/your-model")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 200)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({'generated_text': text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Gradio Interface

```python
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "your-username/your-model")
tokenizer = AutoTokenizer.from_pretrained("your-username/your-model")

def generate_text(prompt, max_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Create interface
interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", lines=3),
        gr.Slider(50, 500, value=200, label="Max Tokens"),
        gr.Slider(0.1, 2.0, value=0.8, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="NovelCrafter Text Generator",
    description="Generate text in the style of your trained model"
)

interface.launch()
```

---

## Tips & Best Practices

1. **Start Small**: Test with 2 epochs and 5 parts first
2. **Monitor Loss**: Should decrease steadily
3. **Save Checkpoints**: Keep multiple versions
4. **Test Generation**: After each part, generate samples
5. **Adjust Temperature**: Lower for consistency, higher for creativity
6. **Use WandB**: Track experiments systematically
7. **Version Control**: Keep track of what works

---

For more examples, check out:
- [USAGE.md](USAGE.md) - Detailed usage guide
- [README.md](README.md) - Project overview
- [MODEL_CARD.md](MODEL_CARD.md) - Model documentation
