"""Shared helpers for the LiteSpecFormer unit tests."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from litespecformer.config import LiteSpecFormerConfig
from litespecformer.model import LiteSpecFormerModel


def tiny_config(**forecasting_overrides) -> LiteSpecFormerConfig:
    """A one-layer model small enough to construct and run on CPU."""
    forecasting = {
        "context_length": 32,
        "output_patch_size": 8,
        "input_patch_size": 8,
        "input_patch_stride": 8,
        "quantiles": [0.1, 0.5, 0.9],
        "max_output_patches": 2,
        "min_past": 8,
        "use_arcsinh": False,
    }
    forecasting.update(forecasting_overrides)
    return LiteSpecFormerConfig(
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=1,
        num_heads=4,
        dropout_rate=0.0,
        forecasting_config=forecasting,
    )


def tiny_model(**forecasting_overrides) -> LiteSpecFormerModel:
    model = LiteSpecFormerModel(tiny_config(**forecasting_overrides))
    model.eval()
    return model
