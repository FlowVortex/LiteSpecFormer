import argparse

import os
import sys
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from safetensors.torch import load_file

from litespecformer import (
    LiteSpecFormerModel,
    LiteSpecFormerConfig,
    LiteSpecFormerPipeline,
)

parser = argparse.ArgumentParser(description="Zero-Shot Prediction with LiteSpecFormer")

# Dataset path (local file / directory) used for zero-shot evaluation.
parser.add_argument(
    "--dataset_name_or_path",
    type=str,
    help="Path or identifier of the dataset used for zero-shot evaluation.",
)

# Input context window length fed to the model.
# The model uses this historical segment to forecast future spectrum values.
parser.add_argument(
    "--seq_length",
    type=int,
    default=336,
    help="Length of the historical context window provided to the model.",
)

# Forecast horizon length.
# The model predicts this many future time steps after each input context.
parser.add_argument(
    "--prediction_length",
    type=int,
    default=96,
    help="Number of future time steps to predict for each sample.",
)

# Optional local checkpoint path.
# If omitted, the script falls back to a pretrained public model from Hugging Face Hub.
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None,
    help=(
        "Path to a local model checkpoint (.safetensors / compatible state dict). "
        "If not provided, a default pretrained model is loaded from the Hub."
    ),
)

# Evaluation batch size.
# Larger values improve throughput but require more GPU memory.
parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="Batch size used during evaluation/prediction.",
)

# Optional output directory for saving prediction artifacts or metrics
# (if downstream evaluation utilities support writing outputs).
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Optional directory for saving evaluation outputs/results.",
)

args = parser.parse_args()

# Model loading strategy:
# 1) No checkpoint path provided:
#    Load an official pretrained LiteSpecFormer pipeline directly from the Hub.
# 2) Checkpoint path provided:
#    Build model from config, load local weights, then wrap with pipeline.
if args.checkpoint_path is None:
    pipeline = LiteSpecFormerPipeline.from_pretrained(
        "https://huggingface.co/FlowVortex/LiteSpecFormer-1.0-36M", device_map="cuda"
    )
else:
    # Load serialized parameters from local checkpoint.
    params = load_file(args.checkpoint_path)

    # Instantiate model architecture with default LiteSpecFormer config.
    model = LiteSpecFormerModel(LiteSpecFormerConfig())

    # Load checkpoint weights into model.
    # strict=False allows partial compatibility (e.g., missing/unexpected keys).
    model.load_state_dict(params, strict=False)

    # Build inference pipeline for zero-shot prediction/evaluation.
    pipeline = LiteSpecFormerPipeline(model=model)

# Run zero-shot evaluation on the spectrum library benchmark/data.
# The pipeline handles dataset loading, inference, and metric computation internally.
pipeline.evaluate_spectrum_library(
    dataset_name_or_path=args.dataset_name_or_path,
    seq_length=args.seq_length,
    prediction_length=args.prediction_length,
    batch_size=args.batch_size,
)
