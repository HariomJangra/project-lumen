# Project Lumen

A 128M Parameter Language Model Built from Scratch

Project Lumen is a foundational language model created entirely from scratch for learning and research purposes.
It explores every step of modern LLM development — from data preprocessing and tokenization to architecture design, training, evaluation and so on...

## 🎯 Overview

This project implements a GPT-style transformer model from scratch using PyTorch, featuring grouped multi-query attention (GQA), SwiGLU activation, and RMSNorm. The **128M parameter** model is trained on custom datasets and evaluated on standard NLP benchmarks.

## 📁 Project Structure

```
PreTraining/
├── Implementation/       # Model training and data preparation
├── Benchmark/           # Model evaluation on benchmarks
├── Inference/          # Text generation and inference
```

## ✨ Key Features

- **Custom Transformer Architecture**
  - Grouped Multi-Query Attention (GQA) for efficient inference
  - SwiGLU feed-forward networks
  - RMSNorm for layer normalization
  - Rotary Position Embeddings (RoPE)
  - Weight tying between embedding and output layers

- **Training Pipeline**
  - Mixed precision training (FP16/BF16)
  - Gradient accumulation and clipping
  - Cosine annealing with linear warmup
  - Automatic checkpointing and resume support

- **Benchmarking**
  - ARC (AI2 Reasoning Challenge) - Easy & Challenge
  - HellaSwag commonsense reasoning

## 🚀 Quick Start

### 1. Data Preparation

Prepare your dataset using the provided notebook:

```bash
jupyter notebook PreTraining/Implementation/01_Dataset-Prepration.ipynb
```

### 2. Train Custom Tokenizer

Train a BPE tokenizer on your dataset:

```bash
jupyter notebook PreTraining/Implementation/02_Training-Tokenizer.ipynb
```

### 3. Tokenize Dataset

Convert text data to token IDs:

```bash
jupyter notebook PreTraining/Implementation/03_Tokenizing-Dataset.ipynb
```

### 4. Pre-train Model

Launch the training process:

```bash
jupyter notebook PreTraining/Implementation/PreTraining.ipynb
```

### 5. Run Inference

Generate text with your trained model:

```bash
jupyter notebook PreTraining/Inference/Inference.ipynb
```

## 📊 Model Configuration

```python
ModelConfig(
    vocab_size=32000,
    hidden_size=768,
    n_heads=12,
    n_kv_heads=4,           # Grouped Query Attention
    n_kv_groups=3,
    head_dim=64,
    n_layers=12,
    intermediate_size=3072,
    max_position_embeddings=2048,
    dropout=0.1,
    tie_weights=True
)
```

## 🎛️ Training Configuration

- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.1)
- **Scheduler**: Linear warmup + Cosine annealing
- **Batch Size**: 12 with 4 gradient accumulation steps
- **Sequence Length**: 2048 tokens
- **Mixed Precision**: Automatic (BF16/FP16/FP32)

### Training Progress

![Training Loss Curve](images/training_loss_curve.png)

*Training vs Validation Loss over time*

## 📈 Benchmarks

Evaluate your model on standard benchmarks:

```bash
jupyter notebook PreTraining/Benchmark/Benchmark.ipynb
```

Supported benchmarks:
- ARC-Easy & ARC-Challenge
- HellaSwag

## 🚧 Post-Training

**Coming Soon**: Fine-tuning capabilities for instruction following, chat models, and task-specific adaptations.

## 🔧 Requirements

```bash
pip install torch numpy tqdm tokenizers datasets huggingface_hub matplotlib
```

## 📝 Model Checkpoints

The training pipeline automatically saves:
- Regular checkpoints every N steps
- Best model based on validation loss
- Training history and loss curves

Checkpoints are saved in: `PreTraining/Implementation/checkpoints/`

## 🎨 Text Generation

The model supports various sampling strategies:

- **Greedy decoding** (temperature=0)
- **Top-k sampling**
- **Nucleus (top-p) sampling**
- **Temperature scaling**

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements.

## 📄 License

This project is open source and licensed under the MIT License. See the LICENSE file for details.

---

**Note**: This is a research/educational implementation. For production use, consider established frameworks like Hugging Face Transformers.
