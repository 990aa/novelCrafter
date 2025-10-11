# 📚 NovelCrafter

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**NovelCrafter** is an incremental fine-tuning framework for training language models on literary works. It enables efficient training of LLMs on books and novels using LoRA (Low-Rank Adaptation) with automatic incremental training, progress tracking, and Hugging Face integration.

## 🌟 Features

- **📖 PDF Text Extraction**: Automatically extracts and processes text from PDF books
- **🔄 Incremental Training**: Splits large texts into manageable chunks for progressive training
- **⚡ LoRA Fine-Tuning**: Memory-efficient training using Parameter-Efficient Fine-Tuning (PEFT)
- **💾 Auto-Save & Resume**: Automatic progress tracking and ability to resume training
- **☁️ HuggingFace Integration**: Automatic model uploads to Hugging Face Hub
- **📊 WandB Logging**: Optional Weights & Biases integration for experiment tracking
- **🖥️ CPU/GPU Support**: Automatic device detection with optimized configurations
- **🔧 Smart Model Selection**: Uses 1B model for CPU, 3B model for GPU

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- 8GB+ RAM (16GB recommended for CPU training)
- GPU with 8GB+ VRAM (optional, but recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/990aa/novelCrafter.git
   cd novelCrafter
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   HF_TOKEN=your_huggingface_token_here
   WANDB_API_KEY=your_wandb_api_key_here  # Optional
   ```
   
   Get your tokens:
   - **Hugging Face Token**: https://huggingface.co/settings/tokens (needs write permission)
   - **WandB API Key**: https://wandb.ai/authorize (optional)

### Usage

1. **Place your book PDF** in the `input/` directory
   ```
   input/
   └── your_book.pdf
   ```

2. **Update the PDF path** in `main.py` (line 48):
   ```python
   book_text = extract_text_from_pdf("input/your_book.pdf")
   ```

3. **Run the training script**
   ```bash
   python main.py
   ```

4. **Training Progress**
   - The script will train on 10 incremental parts
   - After each part, you'll be asked to continue
   - Models are automatically saved and uploaded to HuggingFace
   - Progress is saved in `train_progress.json`

5. **Resume Training** (if interrupted)
   - Simply run `python main.py` again
   - It will automatically resume from the last completed part

## 📁 Project Structure

```
novelCrafter/
├── main.py                    # Main training script
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
├── .gitignore                # Git ignore rules
├── LICENSE.md                # MIT License
├── README.md                 # This file
├── MODEL_CARD.md             # Model documentation
├── USAGE.md                  # Detailed usage guide
├── train_progress.json       # Training progress tracker
├── input/                    # Place your PDF books here
│   └── The_crocodile.pdf
├── book_model_part_1/        # Saved model checkpoints
├── book_model_part_2/
└── wandb/                    # WandB logs (if enabled)
```

## ⚙️ Configuration

### Model Selection

The script automatically selects the appropriate model based on your hardware:

- **GPU (CUDA)**: `meta-llama/Llama-3.2-3B-Instruct` (3 billion parameters)
- **CPU**: `meta-llama/Llama-3.2-1B-Instruct` (1 billion parameters)

You can modify this in `main.py` around line 115.

### Training Parameters

Key training parameters (in `main.py`):

```python
# Text chunking
chunk_size = 10              # Sentences per chunk
num_parts = 10               # Number of training parts

# LoRA Configuration
r=8                          # LoRA rank
lora_alpha=32               # LoRA alpha
lora_dropout=0.05           # Dropout rate

# Training Arguments
num_train_epochs=3          # Epochs per part
per_device_train_batch_size=1
gradient_accumulation_steps=8
learning_rate=5e-5
weight_decay=0.01
```

### Hugging Face Repository

Update the repository name in `main.py` (line 83):
```python
hf_repo = "your-username/your-model-name"
```

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Use the 1B model instead of 3B
- Close other applications

**2. Slow Training on CPU**
- Expected behavior - CPU training is 10-100x slower than GPU
- Consider using Google Colab or cloud GPU
- Reduce `num_train_epochs` for faster iterations

**3. Model Not Uploading to HuggingFace**
- Check your HF_TOKEN has write permissions
- Ensure repository exists: `huggingface-cli repo create your-model-name`
- Check internet connection

**4. Import Errors**
- Reinstall requirements: `pip install -r requirements.txt --upgrade`
- Check Python version: `python --version` (needs 3.8+)

## 📊 Monitoring Training

### Local Logs
- Training progress is printed to console
- Models saved in `book_model_part_X/` directories
- Progress tracked in `train_progress.json`

### WandB (Optional)
If you set `WANDB_API_KEY`, view training metrics at:
```
https://wandb.ai/your-username/huggingface
```

## 🎯 Use Cases

- **Style Transfer**: Train models to write in the style of specific authors
- **Book Continuation**: Generate text that continues a book's narrative
- **Literary Analysis**: Fine-tune models for book-specific Q&A
- **Creative Writing**: Use as a writing assistant trained on specific genres
- **Educational**: Learn about LLM fine-tuning and PEFT techniques

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- **Meta AI** for the Llama models
- **Hugging Face** for transformers and PEFT libraries
- **Microsoft** for DeepSpeed optimizations
- **Weights & Biases** for experiment tracking

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{novelcrafter2025,
  author = {990aa},
  title = {NovelCrafter: Incremental Fine-Tuning Framework for Literary LLMs},
  year = {2025},
  url = {https://github.com/990aa/novelCrafter}
}
```

## 📧 Contact

- **GitHub**: [@990aa](https://github.com/990aa)
- **Hugging Face**: [a-01a](https://huggingface.co/a-01a)

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ for the AI and Literature community**
