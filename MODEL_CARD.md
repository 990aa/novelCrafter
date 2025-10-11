# Model Card: NovelCrafter Fine-Tuned Model

## Model Details

### Model Description

This model is a fine-tuned version of Meta's Llama 3.2 (1B or 3B) using LoRA (Low-Rank Adaptation) on literary text. It has been trained incrementally on book content to capture writing style, narrative patterns, and literary conventions.

- **Developed by**: [990aa](https://github.com/990aa)
- **Model type**: Causal Language Model (CLM)
- **Base Model**: 
  - `meta-llama/Llama-3.2-1B-Instruct` (CPU training)
  - `meta-llama/Llama-3.2-3B-Instruct` (GPU training)
- **Language(s)**: English (primarily)
- **License**: MIT License (training code), Llama 3.2 License (base model)
- **Finetuned from**: Meta Llama 3.2 Instruct
- **Training Method**: LoRA (Parameter-Efficient Fine-Tuning)

### Model Sources

- **Repository**: [https://github.com/990aa/novelCrafter](https://github.com/990aa/novelCrafter)
- **Model Hub**: [https://huggingface.co/a-01a/novelCrafter](https://huggingface.co/a-01a/novelCrafter)

## Uses

### Direct Use

This model can be used for:
- **Text Generation**: Generate text in the style of the training book
- **Story Continuation**: Continue narratives with consistent style
- **Creative Writing Assistance**: Help authors write in specific literary styles
- **Literary Analysis**: Understand patterns in specific works
- **Educational Purposes**: Learn about fine-tuning and literary AI

### Downstream Use

Can be further fine-tuned on:
- Additional literary works
- Specific genres or authors
- Creative writing tasks
- Dialogue generation
- Scene description

### Out-of-Scope Use

This model should NOT be used for:
- Medical, legal, or financial advice
- Generating harmful, toxic, or biased content
- Impersonating specific real individuals
- Producing academic work without proper attribution
- Any application requiring factual accuracy without verification

## Bias, Risks, and Limitations

### Known Limitations

1. **Training Data Bias**: The model reflects biases present in the training literature
2. **Factual Accuracy**: Not trained for factual tasks; may generate plausible but incorrect information
3. **Context Length**: Limited to the base model's context window (~8k tokens for Llama 3.2)
4. **Style Specificity**: Most effective for generating text similar to training material
5. **Language**: Primarily trained on English text

### Risks

- **Copyright Concerns**: Generated text may inadvertently reproduce training data
- **Harmful Content**: Despite instruction tuning, may generate inappropriate content
- **Over-reliance**: Users should not rely solely on model outputs for critical decisions
- **Hallucination**: May generate confident but false information

### Recommendations

Users should:
- Review and edit all generated content
- Add appropriate disclaimers for AI-generated text
- Not use for high-stakes decisions without human oversight
- Be aware of potential copyright issues
- Test thoroughly for their specific use case

## Training Details

### Training Data

- **Source**: PDF book(s) placed in `input/` directory
- **Preprocessing**: 
  - Text extracted from PDF
  - Cleaned and normalized (whitespace, newlines)
  - Split into sentence chunks (10 sentences per chunk by default)
  - Tokenized with Llama tokenizer
  - 90/10 train/test split per training part

### Training Procedure

#### Training Hyperparameters

**LoRA Configuration:**
```python
rank (r) = 8
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "v_proj"]
bias = "none"
task_type = "CAUSAL_LM"
```

**Training Arguments:**
```python
num_train_epochs = 3 (per part)
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
learning_rate = 5e-5
weight_decay = 0.01
warmup_steps = 100 (adjusted per part)
fp16 = True (GPU only)
optimizer = AdamW
lr_scheduler = Linear with warmup
```

#### Training Process

1. **Text Extraction**: PDF → plain text
2. **Chunking**: Split into 10 parts for incremental training
3. **Tokenization**: Llama tokenizer with max_length=1024
4. **LoRA Application**: Add trainable adapters to base model
5. **Incremental Training**: Train on each part sequentially
6. **Checkpoint Saving**: Save after each part
7. **Hub Upload**: Push to Hugging Face after each part

**Trainable Parameters:**
- Total parameters: ~1.2B (1B model) or ~3.2B (3B model)
- Trainable parameters: ~2.3M (0.07% of total)
- LoRA enables efficient training with minimal memory

#### Compute Infrastructure

**Hardware:**
- CPU training: Any modern CPU with 8GB+ RAM
- GPU training: NVIDIA GPU with 8GB+ VRAM recommended
- Tested on: Consumer-grade hardware

**Software:**
```
Python 3.8+
PyTorch 2.0+
Transformers 4.56+
PEFT 0.17+
```

**Training Time:**
- CPU (1B model): ~2-4 hours per part (30-40 hours total)
- GPU (3B model): ~15-30 minutes per part (3-5 hours total)

## Evaluation

### Testing Data

- 10% of each training part held out for evaluation
- Evaluated using perplexity on held-out test set
- Real-time evaluation during training

### Metrics

- **Training Loss**: Cross-entropy loss on training data
- **Validation Loss**: Cross-entropy loss on test data
- **Perplexity**: exp(validation_loss)

Note: Specific metrics depend on the training run and can be viewed in WandB logs or training outputs.

## Environmental Impact

- **Hardware Type**: CPU or GPU (varies by user)
- **Hours Used**: 3-40 hours (depending on hardware)
- **Cloud Provider**: N/A (local training)
- **Compute Region**: User-dependent
- **Carbon Emitted**: Varies by location and power source

We encourage users to:
- Use energy-efficient hardware when possible
- Train during off-peak hours
- Consider renewable energy sources
- Reuse and share trained models

## Technical Specifications

### Model Architecture

- **Base Architecture**: Llama 3.2 (Transformer decoder)
- **Attention Type**: Multi-head attention with GQA
- **Hidden Size**: 2048 (1B) or 3072 (3B)
- **Num Layers**: 16 (1B) or 28 (3B)
- **Num Attention Heads**: 32
- **Vocabulary Size**: 128,256
- **Position Embeddings**: RoPE (Rotary Position Embedding)

### Fine-Tuning Method

**LoRA (Low-Rank Adaptation):**
- Adds trainable low-rank matrices to attention layers
- Freezes original model weights
- Reduces memory and compute requirements
- Enables efficient multi-task learning

## How to Use

### Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "a-01a/novelCrafter")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("a-01a/novelCrafter")
```

### Generating Text

```python
# Prepare input
prompt = "Once upon a time, in a distant land,"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1
)

# Decode
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### Inference Parameters

Recommended generation parameters:
```python
max_new_tokens = 100-500      # Length of generation
temperature = 0.7-0.9         # Creativity (lower = more focused)
top_p = 0.9                   # Nucleus sampling
top_k = 50                    # Top-k sampling
repetition_penalty = 1.1-1.2  # Prevent repetition
do_sample = True              # Enable sampling
```

## Model Card Contact

For questions or concerns about this model:
- **GitHub Issues**: [https://github.com/990aa/novelCrafter/issues](https://github.com/990aa/novelCrafter/issues)
- **Email**: Via GitHub profile

## Changelog

### Version 1.0.0 (October 2025)
- Initial release
- Incremental training on literary works
- LoRA fine-tuning implementation
- CPU/GPU optimization
- Hugging Face integration

---

**Model Card Authors**: 990aa  
**Model Card Date**: October 2025  
**Model Card Version**: 1.0.0
