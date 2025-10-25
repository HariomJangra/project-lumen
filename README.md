# Project Lumen

A 128M Parameter Language Model Built from Scratch

Project Lumen is a foundational language model created entirely from scratch for learning and research purposes.
It explores every step of modern LLM development — from data preprocessing and tokenization to architecture design, training, evaluation and so on...

## Live PlayGround

> **Instruct Model**: [![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-black?logo=vercel)](https://lumenchat.vercel.app/)   [![Hugging Face](https://img.shields.io/badge/Instruct_Model-HuggingFace-green?logo=huggingface)](https://huggingface.co/spaces/VirtualInsight/Lumen-Instruct)
>
> **Base Model**: [![Hugging Face](https://img.shields.io/badge/Base_Model-HuggingFace-blue?logo=huggingface)](https://huggingface.co/spaces/VirtualInsight/LumenBase)


## 🎯 Overview

This project implements a GPT-style transformer model from scratch using PyTorch, featuring grouped multi-query attention (GQA), SwiGLU activation, and RMSNorm. The **128M parameter** model is trained on custom datasets and evaluated on standard NLP benchmarks.

## 🤖 Model Variants

### Lumen Base
The foundational pre-trained model trained on diverse text data, capable of general language understanding and generation. Primarily intended for research and development purposes.

### ⭐ Lumen Instruct
A fine-tuned version of the base model optimized for following instructions and engaging in conversational AI tasks. This variant has been trained on instruction-following datasets to provide more helpful, accurate, and contextually appropriate responses.


## 📈 Benchmarks (Base Model)

Benchmarks: ARC-Easy, ARC-Challenge, HellaSwag

| Benchmark | Accuracy | Correct/Total |
|-----------|----------|---------------|
| ARC-Easy | 39.48% | 938/2,376 |
| ARC-Challenge | 23.55% | 276/1,172 |
| HellaSwag | 32.62% | 334/1,024 |

Run detailed evaluation in: `PreTraining/Benchmark/Benchmark.ipynb`

## 📉 Training Loss

![Training Loss Curve](PreTraining/images/training_loss_curve.png)

## 📁 Project Structure

```
PreTraining/
├── Implementation/       # Base model training and data preparation
├── Benchmark/           # Model evaluation on benchmarks
├── Inference/          # Text generation and inference

PostTraining/
├── Implementation/      # Supervised fine-tuning for Instruct model
├── Datasets/           # Instruction-following datasets
├── Inference/          # Instruct model inference
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

### ⭐ For End Users (Recommended)
**Try Lumen Instruct directly:**
- 🌐 **Live Demo**: [lumenchat.vercel.app](https://lumenchat.vercel.app/)
- 🤗 **Hugging Face**: [VirtualInsight/Lumen-Instruct](https://huggingface.co/spaces/VirtualInsight/Lumen-Instruct)

### For Developers & Researchers

**Base Model Training Pipeline:**
1. **Data Preparation** → `PreTraining/Implementation/01_Dataset-Prepration.ipynb`
2. **Train Tokenizer** → `PreTraining/Implementation/02_Training-Tokenizer.ipynb` (BPE, 32K vocab)
3. **Tokenize Dataset** → `PreTraining/Implementation/03_Tokenizing-Dataset.ipynb`
4. **Pre-train Model** → `PreTraining/Implementation/PreTraining.ipynb`
5. **Run Inference** → `PreTraining/Inference/Inference.ipynb`

**Instruct Model Fine-tuning Pipeline:**
1. **Prepare Instruction Datasets** → `PostTraining/Implementation/Dataset-Prepration.ipynb`
2. **Supervised Fine-tuning** → `PostTraining/Implementation/SupervisedFineTuning.ipynb`
3. **Instruct Model Inference** → `PostTraining/Inference/Inference.ipynb`

**Using the Models:**

- **Base Model**: See `PreTraining/Inference/Inference.ipynb` for complete usage examples
- **Instruct Model**: See `PostTraining/Inference/Inference.ipynb` for complete usage examples

## 📊 Model Configuration

```python
vocab_size: 32000          # BPE tokenizer vocabulary
hidden_size: 768           # Model dimension
n_heads: 12                # Query heads
n_kv_heads: 4              # Key-Value heads (GQA)
n_layers: 12               # Transformer layers
intermediate_size: 3072    # FFN dimension
max_position_embeddings: 2048
```

## 🎛️ Training Setup

- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.1)
- **Batch**: 12 × 4 accumulation = 48 effective
- **Precision**: Mixed (BF16/FP16/FP32)
- **Scheduler**: Linear warmup + Cosine annealing


 

## 🔧 Requirements

```bash
pip install torch numpy tqdm tokenizers datasets huggingface_hub matplotlib
```

## 🎨 Sampling Strategies

- **Greedy**: temperature=0
- **Top-k**: Sample from k most likely tokens
- **Top-p (Nucleus)**: Sample from cumulative probability p
- **Temperature**: Control randomness (lower = deterministic)

## 📄 License

Apache License 2.0 - See LICENSE file for details.

---

**Note**: Educational/research implementation. For production, use established frameworks like Hugging Face Transformers.
