import argparse
import sys
import os
from os import path

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator

from datasets import Dataset as HFDataset

from litespecformer.dataset import SpectrumLibraryDataset
from litespecformer.pipeline import LiteSpecFormerPipeline

parser = argparse.ArgumentParser("LiteSpecFormer Fine-tuning")

# Path to the fine-tuning dataset (expected to be in Hugging Face datasets format).
parser.add_argument(
    "--dataset_name_or_path",
    type=str,
    required=True,
    help=(
        "Path to the fine-tuning dataset in Hugging Face `datasets` format "
        "(e.g., a directory loadable by `datasets.load_from_disk`)."
    ),
)

# Fine-tuning strategy:
# - "full": update all trainable model parameters.
# - "lora": parameter-efficient fine-tuning using LoRA adapters.
parser.add_argument(
    "--finetune_mode",
    type=str,
    required=True,
    choices=["full", "lora"],
    help=(
        "Fine-tuning mode. "
        "`full` updates the whole model, while `lora` performs parameter-efficient adaptation."
    ),
)

# Input context length used during fine-tuning.
# This controls how many historical time steps are provided to the model.
parser.add_argument(
    "--context_length",
    type=int,
    default=768,
    help="Length of the input context window used for fine-tuning.",
)

# Optimizer learning rate for fine-tuning updates.
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-6,
    help="Learning rate for fine-tuning optimization.",
)

# Number of optimization steps (not epochs) for the fine-tuning loop.
parser.add_argument(
    "--num_steps",
    type=int,
    default=2560,
    help="Total number of gradient update steps during fine-tuning.",
)

# Training batch size used by `pipeline.fit`.
parser.add_argument(
    "--batch_size",
    type=int,
    default=1024,
    help="Mini-batch size used during fine-tuning.",
)

# Optional output path for saving fine-tuned artifacts/checkpoints.
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Optional directory to save the fine-tuned model/checkpoints.",
)

# Minimum required history length for constructing training samples.
# Samples shorter than this threshold may be filtered out by the dataset pipeline.
parser.add_argument(
    "--min_past",
    type=int,
    default=64,
    help="Minimum number of past observations required for each training sample.",
)

# Train/validation split ratio used when creating `SpectrumLibraryDataset`.
# Example: 0.6 means 60% train and 40% test/validation split.
parser.add_argument(
    "--split_ratio",
    type=float,
    default=0.6,
    help="Train split ratio used when partitioning the spectrum dataset.",
)

# Forecast horizon length used in the final evaluation phase (after fine-tuning).
parser.add_argument(
    "--prediction_length",
    type=int,
    default=96,
    help="Prediction horizon length used for post-fine-tuning evaluation.",
)

args = parser.parse_args()

# -----------------------------------------------------------------------------
# 1) Load and preprocess the spectrum dataset for fine-tuning
# -----------------------------------------------------------------------------
# `SpectrumLibraryDataset` returns data arranged for sequence forecasting.
# Here `prediction_length=16` is used for training sample construction.
train_dataset = SpectrumLibraryDataset(
    seq_length=args.context_length,
    dataset_name_or_path=args.dataset_name_or_path,
    prediction_length=16,
    split_ratio=args.split_ratio,
    flag="train",
)

# -----------------------------------------------------------------------------
# 2) Convert dataset into Hugging Face Dataset format expected by pipeline.fit
# -----------------------------------------------------------------------------
# `data_x` has shape (seq_length, num_channels).
# Each channel is converted into one sample: {"target": 1D series}.
data_x = train_dataset.data_x
seq_length, num_channels = data_x.shape
hf_dataset = HFDataset.from_list(
    [{"target": data_x[:, i]} for i in range(num_channels)]
)
hf_dataset.set_format("torch")

# Basic sanity logs for quick verification.
print(hf_dataset)
print(
    f"Dataset loaded with {len(hf_dataset)} samples, each sample has shape {hf_dataset[0]['target'].shape}"
)

# -----------------------------------------------------------------------------
# 3) Load pretrained LiteSpecFormer pipeline as initialization
# -----------------------------------------------------------------------------
# Fine-tuning starts from a pretrained checkpoint from Hugging Face Hub.
pipeline = LiteSpecFormerPipeline.from_pretrained(
    "FlowVortex/LiteSpecFormer", device_map="cuda"
)

# -----------------------------------------------------------------------------
# 4) Run fine-tuning
# -----------------------------------------------------------------------------
# Note:
# - `inputs` is the per-channel Hugging Face dataset.
# - `prediction_length=16` is the training horizon used in this script.
pipeline.fit(
    inputs=hf_dataset,
    prediction_length=16,
    context_length=args.context_length,
    finetune_mode=args.finetune_mode,
    learning_rate=args.learning_rate,
    num_steps=args.num_steps,
    batch_size=args.batch_size,
    min_past=args.min_past,
    output_dir=args.output_dir,
)

# -----------------------------------------------------------------------------
# 5) Evaluate fine-tuned model on spectrum library benchmark
# -----------------------------------------------------------------------------
# Evaluation uses a fixed context length (`seq_length=512`) and configurable horizon.
pipeline.evaluate_spectrum_library(
    dataset_name_or_path=args.dataset_name_or_path,
    seq_length=512,
    prediction_length=args.prediction_length,
    batch_size=16,
)
