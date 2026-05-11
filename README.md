# 🇹🇭 ThaiLLM — Continual Pre-Training Pipeline

> A production-ready pipeline for Continual Pre-Training (CPT) of large language models on Thai-language corpora, built on top of [LlamaFactory](https://github.com/hiyouga/LlamaFactory) and optimised for NSTDA's **Lanta HPC cluster**.

---

## Table of Contents
- [Lanta Guild](https://www.notion.so/Lanta-Guild-31da7787e2728022960cf1648729e37d?source=copy_link)
- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Tokenization](#1-data-tokenization)
  - [2. Training](#2-training)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

---

## Overview

This repository provides a complete, HPC-ready workflow to continually pre-train an LLM on Thai text data. The pipeline covers:

- **Data preparation** — tokenizing raw corpora into packed, model-ready datasets via SLURM batch jobs
- **Distributed training** — multi-node, multi-GPU training with DeepSpeed ZeRO, Flash Attention 2, and Liger Kernel optimisations
- **Experiment tracking** — integrated Weights & Biases (wandb) logging

The stack is designed for NVIDIA H100/A100 nodes on Lanta but can be adapted to any SLURM-managed GPU cluster.

---

## Architecture

```
thaillm-cpt/
├── LlamaFactory/          # Submodule — training framework (hiyouga/LlamaFactory)
└── example/
    ├── tokenized-data/    # Data preparation job (data-prep.sh)
    └── train/             # Training job (submit.sh)
```

The pipeline follows two sequential stages:

```
Raw Text Corpus
      │
      ▼
[data-prep.sh]  ──► Tokenised & Packed Dataset
      │
      ▼
[submit.sh]     ──► Fine-tuned / CPT Checkpoint
```

---

## Prerequisites

### Cluster Access

This pipeline targets the **Lanta HPC cluster**. Ensure you have a valid Lanta account and have reviewed the onboarding guide:

📖 [Lanta Guild — Getting Started](https://www.notion.so/Lanta-Guild-31da7787e2728022960cf1648729e37d)

### Required Modules (Lanta Environment)

Load the following modules before proceeding:

```bash
ml purge
ml cuda
ml gcc
ml Mamba
```

---

## Installation

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/ainrg-llm-dev/thaillm-cpt
cd thaillm-cpt/model-training-cpt
```

> The `--recursive` flag is required to pull the LlamaFactory submodule.

### 2. Create the Conda Environment

```bash
# Set a writable temporary directory (important on shared filesystems)
export TMPDIR=./.cache

conda create -p ./env python=3.11 -y
conda activate ./env/
```

### 3. Install Python Dependencies

```bash
# PyTorch with CUDA 12.8 support
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# LlamaFactory (editable install)
pip install -e LlamaFactory/.

# Training utilities
pip install wandb deepspeed==0.18.7 liger_kernel==0.7.0 huggingface_hub

# Flash Attention 2 (no build isolation required for pre-built wheel)
pip install flash_attn==2.8.3 --no-build-isolation
```

> **Note:** `flash_attn` installation may take several minutes to compile on first run.

---

## Usage

### 1. Data Tokenization

Navigate to the tokenization example and submit the SLURM job:

```bash
cd example/tokenized-data/
sbatch data-prep.sh
```

Monitor the job queue and tail logs until completion:

```bash
# Check your job status
myqueue          # Lanta alias
# or
squeue -u <your_username>

# Stream live logs
tail -f <JOB_ID>
```

Wait for the job to reach `COMPLETED` status before proceeding to training.

### 2. Training

Navigate to the training directory and submit:

```bash
# If currently in tokenized-data/
cd ../train/

# Submit the training job
sbatch submit.sh
```

Monitor training progress:

```bash
tail -f logs/<JOB_ID>/node_log/node-0.out
```

Training metrics (loss, learning rate, throughput) are logged in real time to **Weights & Biases** if `WANDB_API_KEY` is set in your environment.

---

## Project Structure

```
thaillm-cpt/
│
├── LlamaFactory/                  # Submodule: training framework
│
├── example/
│   ├── tokenized-data/
│   │   └── data-prep.sh           # SLURM script for tokenizing the corpus
│   │
│   └── train/
│       └── submit.sh              # SLURM script for launching distributed training
│
├── .gitmodules                    # Submodule configuration
└── .gitignore
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | 3.11 | Runtime |
| PyTorch | 2.8.0 | Deep learning framework |
| CUDA | 12.8 | GPU compute |
| LlamaFactory | 95ac3f2 | Training orchestration |
| DeepSpeed | 0.18.7 | ZeRO distributed optimisation |
| Flash Attention | 2.8.3 | Memory-efficient attention |
| Liger Kernel | 0.7.0 | Fused CUDA kernels for throughput |
| wandb | latest | Experiment tracking |
| huggingface_hub | latest | Model & dataset management |

---
