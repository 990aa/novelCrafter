# Changelog

All notable changes to NovelCrafter will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- EPUB file support
- Better PDF extraction (handle images, tables)
- Web UI for training
- Multi-GPU training support
- Quantization support (4-bit, 8-bit)
- Automated testing framework

## [1.0.0] - 2025-10-09

### Added
- Initial release
- PDF text extraction using PyPDF2
- Incremental training on book content (10 parts)
- LoRA fine-tuning implementation
- CPU/GPU automatic detection and optimization
- Hugging Face Hub integration for model uploads
- WandB integration for experiment tracking
- Progress tracking and resume capability
- Automatic model selection (1B for CPU, 3B for GPU)
- Comprehensive error handling and logging
- Status indicators (✅, ⚠️, ❌) in console output

### Features
- **Text Processing**: Clean and chunk text into sentences
- **Model Support**: Llama 3.2 1B and 3B Instruct models
- **Training**: 3 epochs per part with configurable hyperparameters
- **LoRA Config**: r=8, alpha=32, targets=["q_proj", "v_proj"]
- **Monitoring**: Real-time progress bars and loss tracking
- **Saving**: Local checkpoints + HuggingFace uploads
- **Resume**: Automatic resume from last completed part

### Configuration
- Adjustable chunk size and number of parts
- Customizable LoRA parameters
- Flexible training arguments
- Environment variable support for tokens

### Documentation
- Comprehensive README.md
- Detailed MODEL_CARD.md
- Complete USAGE.md guide
- CONTRIBUTING.md guidelines
- Example .env file

### Optimizations
- CPU-specific optimizations (low_cpu_mem_usage)
- Dynamic step calculation for small datasets
- Memory-efficient gradient accumulation
- Automatic device mapping for GPU

### Bug Fixes
- Fixed `torch_dtype` deprecation (changed to `dtype`)
- Fixed `tokenizer` deprecation (changed to `processing_class`)
- Fixed `huggingface-cli login` deprecation (changed to `hf auth login`)
- Fixed pin_memory warning on CPU
- Fixed training hang on small datasets
- Fixed model loading hang on CPU (auto-select 1B model)

### Dependencies
```
transformers >= 4.56.2
torch >= 2.8.0
peft >= 0.17.1
accelerate >= 1.10.1
datasets >= 4.1.1
PyPDF2 >= 3.0.1
wandb >= 0.22.1
python-dotenv >= 1.1.1
huggingface_hub[hf_xet] >= 0.35.3
```

## [0.1.0] - 2025-10-08

### Added
- Initial project structure
- Basic training script
- PDF extraction functionality

---

## Release Notes

### Version 1.0.0 Highlights

This is the first stable release of NovelCrafter! 🎉

**Key Features:**
- ✅ Production-ready incremental training pipeline
- ✅ Full CPU and GPU support
- ✅ Automatic model selection based on hardware
- ✅ Comprehensive documentation
- ✅ HuggingFace integration
- ✅ Resume capability for interrupted training

**What's Working:**
- Training on PDF books works reliably
- Models upload successfully to HuggingFace
- Progress tracking and resume functionality
- Clear error messages and status updates

**Known Limitations:**
- Only PDF format supported (EPUB coming soon)
- Single-GPU training only (multi-GPU planned)
- No automated tests yet
- Manual prompt engineering needed for best results

**Performance:**
- CPU (1B model): ~2-4 hours per part
- GPU (3B model): ~15-30 minutes per part
- Memory: 8GB+ RAM, 8GB+ VRAM (GPU)

**Breaking Changes:**
- None (initial release)

---

## Migration Guide

### From Pre-release to 1.0.0

This is the initial release!

---

## Support

For issues, questions, or feature requests:
- GitHub Issues: https://github.com/990aa/novelCrafter/issues
- Discussions: https://github.com/990aa/novelCrafter/discussions

---

**Legend:**
- `Added` - New features
- `Changed` - Changes to existing functionality
- `Deprecated` - Soon-to-be removed features
- `Removed` - Removed features
- `Fixed` - Bug fixes
- `Security` - Security fixes
