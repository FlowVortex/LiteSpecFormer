from dataclasses import dataclass

import torch
from transformers.file_utils import ModelOutput


@dataclass
class AttentionOutput(ModelOutput):
    """The output of the time attention layer in `LiteSpecFormer`"""

    hidden_states: torch.Tensor | None = None
    attn_weights: torch.Tensor | None = None


@dataclass
class LiteSpecFormerEncoderBlockOutput(ModelOutput):
    """The output of a single encoder block in `LiteSpecFormer`"""

    hidden_states: torch.Tensor | None = None
    attn_weights: torch.Tensor | None = None


@dataclass
class LiteSpecFormerEncoderOutput(ModelOutput):
    """The output of the encoder in `LiteSpecFormer`"""

    last_hidden_state: torch.Tensor | None = None
    all_attn_weights: tuple[torch.Tensor, ...] | None = None


@dataclass
class LiteSpecFormerOutput(ModelOutput):
    """The final output of `LiteSpecFormer` model for zero-shot quantile forecasting"""

    loss: torch.Tensor | None = None
    context: torch.Tensor | None = None
    quantile_preds: torch.Tensor | None = None
    enc_attn_weights: tuple[torch.Tensor, ...] | None = None
