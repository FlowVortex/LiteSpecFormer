from typing import Optional

import torch
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from .config import LiteSpecFormerConfig
from .output import AttentionOutput


class RoPE(nn.Module):
    """Applies rotary position embeddings (RoPE) to input tensors.

    Implementation adapted from:
    https://github.com/huggingface/transformers/blob/965cf677695dd363285831afca8cf479cf0c600c/src/transformers/models/llama/modeling_llama.py#L95
    """

    def __init__(self, dim: int, base: float = 10000):
        super().__init__()

        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.inv_freq: torch.Tensor  # type hint for type checker
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attentio, num_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (RoPE.rotate_half(q) * sin)
        k_embed = (k * cos) + (RoPE.rotate_half(k) * sin)
        return q_embed, k_embed


class RMSNorm(nn.Module):
    """
    The RMSNorm layer implementation from the paper `Root Mean Square Layer Normalization`.
    This layer normalizes the inputs based on the root mean square (RMS) of the input values.
    The code is taken from the `huggingface transformers library`.

    Parameters
    ----------
    hidden_size
        The size of the hidden layer.
    eps
        A small value to avoid division by zero, default is 1e-6.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super(RMSNorm, self).__init__()

        # Learnable weight parameter
        self.weight = nn.Parameter(torch.ones(hidden_size))

        # Epsilon value for numerical stability
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the RMSNorm layer.

        :param x: (Tensor) Input tensor of shape (..., hidden_size).

        :return: (Tensor) Normalized tensor of the same shape as input.
        """
        input_dtype = x.dtype
        x = x.to(torch.float32)

        # Compute the variance (mean of squares) along the last dimension
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)

        # Scale the normalized tensor with the learnable weight parameter
        return self.weight * x.to(input_dtype)


class Chronos2LayerNorm(nn.Module):
    """
    The layer normalization module from Chronos2 for the finally outputs of the encoder backbone.

    Parameters
    ----------
    hidden_size
        The size of the hidden layer.
    eps
        A small value to avoid division by zero, default is 1e-6.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


# This is how transformers keeps track of LayerNorm classes ¯\_(ツ)_/¯
ALL_LAYERNORM_LAYERS.append(Chronos2LayerNorm)  # type: ignore


class SEAttention(nn.Module):
    """
    The Squeeze-and-Excitation Attention for Time Series (1D) or Image (2D) Analysis.
    This module adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels.

    Reference: "Squeeze-and-Excitation Networks" by Jie Hu, Li Shen, et al.

    URL: https://arxiv.org/abs/1709.01507

    Parameters
    ----------
    n_dims
        The dimension of input data, either 1 (time series) or 2 (image).
    n_channels
        The number of input channels of time series data.
    reduction
        The reduction ratio for the intermediate layer in the SE block.
    bias
        Whether to include bias terms in the linear layers.
    """

    def __init__(
        self,
        n_dims: int,
        n_channels: int,
        reduction: Optional[int] = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()

        # Validate the input dimension
        assert n_dims in [1, 2], "The dimension of input data must be either 1 or 2."

        # The dimension of inputs data
        self.n_dims = n_dims

        # Global average pooling layer to squeeze the spatial dimensions
        self.avg_pool = (
            nn.AdaptiveAvgPool2d(1) if n_dims == 2 else nn.AdaptiveAvgPool1d(1)
        )

        # Fully connected layers for the excitation operation
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=bias),
            nn.Sigmoid(),
        )

        # View shape for reshaping the excitation output
        self.view_shape = (1, 1) if n_dims == 2 else (1,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SEAttention module.

        The input tensor is expected to be of shape [batch_size, channels, seq_len] for time series data,
        and [batch_size, channels, height, width] for image data.

        Parameters
        ----------
        x
            The input tensor with shape [batch_size, channels, seq_len] for time series data,
            or [batch_size, channels, height, width] for image data.

        Returns
        -------
        torch.Tensor
            The output tensor with shape [batch_size, channels, seq_len] for time series data,
            or [batch_size, channels, height, width] for image data.
        """
        # Get the batch size, number of channels
        batch_size, channels = x.size()[:2]

        # Perform the Squeeze operation
        y = self.avg_pool(x).view(batch_size, channels)

        # Perform the Excitation operation
        y = self.fc(y).view(batch_size, channels, *self.view_shape)

        # Scale the input tensor with the recalibrated weights
        return x * y.expand_as(x)


class Transpose(nn.Module):
    """Transpose the dimensions of the input tensor"""

    def __init__(self, *dims, contiguous: bool = False) -> None:
        super().__init__()

        # Get the dimensions and whether to make the tensor contiguous
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (Tensor) Input tensor of shape (..., *dims).
        :return: (Tensor) Transposed tensor of shape (*dims, ...).
        """
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class DepthWiseConv(nn.Module):
    """
    Depthwise Separable Convolution to reduce the number of parameters and computation.
    The depthwise convolution is applied to the input tensor, followed by a pointwise convolution.
    This module is used in the Feedforward Network of the Transformer Encoder.
    Compared to standard convolution, depthwise separable convolution significantly reduces the number of parameters and computational cost while maintaining performance.

    Parameters
    ----------
    in_channels
        The number of input channels for the deep wise convolution.
    out_channels
        The number of output channels for the point wise convolution.
        Default is None, which means the number of output channels is equal to the number of input channels.
    kernel_size
        The size of the convolutional kernel. Default is 3.
    padding
        The padding to apply to the input. Default is 1.
    bias
        Whether to include a bias term in the convolutional layers. Default is True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Optional[int] = 3,
        padding: Optional[int] = 1,
        bias: Optional[bool] = True,
    ) -> None:
        super(DepthWiseConv, self).__init__()

        if out_channels is None:
            # When out_channels is not specified, set it to the same as in_channels
            out_channels = in_channels

        # Create the depthwise convolutional layer
        self.depth_wise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        # Create the pointwise convolutional layer
        self.point_wise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the DepthWiseConv module."""
        out = self.depth_wise(x)
        out = self.point_wise(out)

        return out


class ConvNet(nn.Module):
    """
    A convolutional network module for processing sequential data.

    Parameters
    ----------
    config
        The configuration object for the ConvNet module from the `LiteSpecFormerConfig` class,
        which contains the necessary hyperparameters for constructing the convolutional layers.
    """

    def __init__(self, config: LiteSpecFormerConfig) -> None:
        super().__init__()

        # Get the kernel size and whether to use channel attention
        self.kernel_size = config.kernel_size
        self.use_channel_attention = config.use_channel_attention

        # Whether to use depthwise convolution
        self.use_dw_cnn = config.use_dw_cnn

        # Create the inputs projection and output projection layers,
        # which can be either depthwise separable convolution or standard convolution based on the configuration
        self.wi = nn.Sequential(
            Transpose(1, 2, contiguous=True),
            (
                DepthWiseConv(
                    in_channels=config.d_model,
                    out_channels=config.d_ff,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    bias=False,
                )
                if self.use_dw_cnn
                else nn.Conv1d(
                    in_channels=config.d_model,
                    out_channels=config.d_ff,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    bias=False,
                )
            ),
        )

        # Create the outputs projection layer,
        # which can be either depthwise separable convolution or standard convolution based on the configuration
        self.wo = nn.Sequential(
            (
                DepthWiseConv(
                    in_channels=config.d_ff,
                    out_channels=config.d_model,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    bias=False,
                )
                if self.use_dw_cnn
                else nn.Conv1d(
                    in_channels=config.d_ff,
                    out_channels=config.d_model,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    bias=False,
                )
            ),
            Transpose(1, 2, contiguous=True),
        )

        # Create the channel attention module
        self.sea = (
            nn.Sequential(
                SEAttention(
                    n_dims=1,
                    n_channels=config.d_ff,
                    reduction=config.reduction,
                    bias=False,
                ),
                ACT2FN[config.dense_act_fn],
            )
            if self.use_channel_attention
            else None
        )

        # Create the activation function and dropout layer
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """The forward pass of the ConvNet module in the feedforward network of the Transformer Encoder."""

        # Pass through the inputs projection layer
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if self.use_channel_attention:
            # Pass through the channel attention module
            hidden_states = self.sea(hidden_states)

        # Pass through the outputs projection layer
        hidden_states = self.wo(hidden_states)

        return hidden_states


class FeedForward(nn.Module):
    """
    The feedforward network module in the Transformer block,
    consisting of two linear layers with an activation function in between.
    """

    def __init__(self, config: LiteSpecFormerConfig) -> None:
        super().__init__()

        assert not config.is_gated_act, "gated activations are unsupported"

        # Create the convolutional network
        self.mlp: nn.Module = ConvNet(config)

        # Create the layer normalization layer and dropout layer
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """
        The forward pass of the feedforward network.

        :param hidden_states: (torch.FloatTensor) Input tensor of shape (batch_size, seq_len, d_model).

        :return: (torch.FloatTensor) Output tensor of shape (batch_size, seq_len, d_model).
        """
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class MHA(nn.Module):
    """
    The Multi-Head Attention module in the Transformer Encoder for `LiteSpecFormer`,
    which supports both eager attention and SDPA implementations, as well as optional RoPE and gating mechanisms.

    To better learn the nonlinear dynamic trends of spectral data, we introduced a nonlinear gating mechanism based on the original linear attention mechanism.

    We primarily used a head-wise approach, adding only one parameter to each head for computation.

    This parameter is added within the mapping of the query vector.

    Finally, a nonlinear activation is achieved using the sigmoid function.

    Parameters
    ----------
    config
        The configuration object for the MHA module from the `LiteSpecFormerConfig` class,
        which contains the necessary hyperparameters for constructing the multi-head attention layers.

    Reference
    ---------
        Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free

    URL
    -----
        https://arxiv.org/abs/2505.06708
    """

    def __init__(self, config: LiteSpecFormerConfig, use_rope: bool = True) -> None:
        super().__init__()
        self.d_model: int = config.d_model  # hidden_size
        self.kv_proj_dim: int = config.d_kv
        self.num_heads: int = config.num_heads
        self.dropout: float = config.dropout_rate

        self.config = config

        # Gate for the query and outputs projections
        self.attn_output_gate = config.attn_output_gate
        self.headwise_attn_output_gate = False
        self.elementwise_attn_output_gate = False

        self.inner_dim: int = self.num_heads * self.kv_proj_dim

        # Select the Gate type and Create the query projection layer for Gate
        if self.attn_output_gate == "headwise":
            self.headwise_attn_output_gate = True
            self.q = nn.Linear(
                in_features=self.d_model,
                out_features=self.inner_dim + self.num_heads,
                bias=False,
            )
        elif self.attn_output_gate == "elementwise":
            self.elementwise_attn_output_gate = True
            self.q = nn.Linear(
                in_features=self.d_model,
                out_features=self.inner_dim * 2,
                bias=False,
            )
        elif self.attn_output_gate == "none":
            self.q = nn.Linear(
                in_features=self.d_model, out_features=self.inner_dim, bias=False
            )
        else:
            raise ValueError(f"Invalid attention output gate: {self.attn_output_gate}.")

        self.k = nn.Linear(
            in_features=self.d_model, out_features=self.inner_dim, bias=False
        )
        self.v = nn.Linear(
            in_features=self.d_model, out_features=self.inner_dim, bias=False
        )

        # The output projection layer
        self.o = nn.Linear(
            in_features=self.inner_dim, out_features=self.d_model, bias=False
        )

        self.use_rope = use_rope
        if use_rope:
            self.rope_embed = RoPE(dim=self.kv_proj_dim, base=config.rope_theta)

    def _eager_attention(
        self,
        query_states: torch.FloatTensor,
        key_states: torch.FloatTensor,
        value_states: torch.FloatTensor,
        mask: torch.Tensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Eager attention implementation using manual matmul.

        Parameters
        ----------
        query_states
            The query states tensor with shape [batch, num_heads, seq_len, kv_proj_dim].
        key_states
            The key states tensor with shape [batch, num_heads, seq_len, kv_proj_dim].
        value_states
            The value states tensor with shape [batch, num_heads, seq_len, kv_proj_dim].
        mask
            The attention mask tensor with shape [batch, num_heads, q_len, kv_len].

        Returns
        -------
        attn_output
            The outputs of the attention value with shape of [batch, num_heads, seq_len, kv_proj_dim]
        attn_weights
            The outputs of the attention weights with shape of [batch, num_heads, q_len, kv_len]
        """
        # Compute attention weights (no scaling - this is the original Chronos-2 implementation)
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # "bnqd,bnkd->bnqk"
        scores += mask
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        return attn_output, attn_weights

    def _sdpa_attention(
        self,
        query_states: torch.FloatTensor,
        key_states: torch.FloatTensor,
        value_states: torch.FloatTensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """SDPA attention implementation using torch.nn.functional.scaled_dot_product_attention.

        Parameters
        ----------
        query_states
            The query states tensor with shape [batch, num_heads, seq_len, kv_proj_dim].
        key_states
            The key states tensor with shape [batch, num_heads, seq_len, kv_proj_dim].
        value_states
            The value states tensor with shape [batch, num_heads, seq_len, kv_proj_dim].
        mask
            The attention mask tensor with shape [batch, num_heads, q_len, kv_len].

        Returns
        -------
        attn_output
            The outputs of the attention value with shape of [batch, num_heads, seq_len, kv_proj_dim]
        attn_weights
            The outputs of the attention weights with shape of [batch, num_heads, q_len, kv_len]
        """
        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0,  # Match eager implementation (no scaling)
        )

        return attn_output, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        """
        The forward processing for the multi-head attention layer.

        Parameters
        ----------
        hidden_states
            The input hidden states tensor with shape [batch, seq_len, d_model].
        mask
            The attention mask tensor with shape [batch, num_heads, q_len, kv_len].
        position_ids
            The position IDs tensor with shape [batch, seq_len].
        output_attentions
            Whether to return the attention weights.

        Returns
        -------
        attn_output
            The outputs of the attention value with shape of [batch, seq_len, d_model]
        attn_weights
            The outputs of the attention weights with shape of [batch, num_heads, q_len, kv_len] if output_attentions=True, else None
        """
        if self.use_rope:
            assert (
                position_ids is not None
            ), "position_ids must be provided when self.use_rope=True"

        # Force eager attention if output_attentions is True (only eager returns weights)
        attn_implementation = self.config._attn_implementation
        if output_attentions:
            attn_implementation = "eager"

        batch_size, seq_length, _ = hidden_states.shape

        def shape(states: torch.Tensor) -> torch.Tensor:
            """(batch, seq_len, inner_dim) -> (batch, num_heads, seq_len, kv_proj_dim)"""
            return rearrange(
                states,
                "b s (h d) -> b h s d",
                h=self.num_heads,
                s=seq_length,
                d=self.kv_proj_dim,
            )

        def unshape(states: torch.Tensor) -> torch.Tensor:
            """(batch, num_heads, seq_len, kv_proj_dim) -> (batch, seq_len, inner_dim)"""
            return rearrange(
                states,
                "b h s d -> b s (h d)",
                h=self.num_heads,
                s=seq_length,
                d=self.kv_proj_dim,
            )

        # Construct query states
        query_states = self.q(
            hidden_states
        )  # [batch_size, n_token, d_model] -> [batch_size, n_token, num_heads * (d_model / num_heads)]

        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))

        # Perform the Granularity for the query projection
        if self.headwise_attn_output_gate:
            # Headwise attention output gate
            query_states = query_states.view(batch_size, seq_length, self.num_heads, -1)

            query_states, gate_score = torch.split(
                query_states,
                [self.kv_proj_dim, 1],
                dim=-1,
            )

            query_states = query_states.reshape(
                batch_size, seq_length, -1, self.kv_proj_dim
            ).transpose(1, 2)

            # Reshape the attention scores
            gate_score = gate_score.reshape(batch_size, seq_length, -1, 1)

        elif self.elementwise_attn_output_gate:
            # Elementwise attention output gate
            query_states = query_states.view(batch_size, seq_length, self.num_heads, -1)

            query_states, gate_score = torch.split(
                query_states,
                [self.kv_proj_dim * self.num_heads, self.kv_proj_dim * self.num_heads],
                dim=-1,
            )

            query_states = query_states.reshape(
                batch_size, seq_length, -1, self.kv_proj_dim
            ).transpose(1, 2)

            gate_score = gate_score.reshape(
                batch_size, seq_length, -1, self.kv_proj_dim
            )
        else:
            query_states = shape(query_states)

        if self.use_rope:
            cos, sin = self.rope_embed(value_states, position_ids)
            query_states, key_states = RoPE.apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if attn_implementation == "sdpa":
            attn_output, attn_weights = self._sdpa_attention(
                query_states, key_states, value_states, mask
            )
        else:  # eager
            attn_output, attn_weights = self._eager_attention(
                query_states, key_states, value_states, mask
            )

        # Transpose back to (batch_size, seq_len, num_heads, head_dim) before non-linearity gating
        attn_output = attn_output.transpose(1, 2).contiguous()

        # If using gating mechanism, apply gating
        if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate_score)

        # Project attention output
        attn_output = rearrange(
            attn_output,
            "b s h d -> b s (h d)",
            s=seq_length,
            h=self.num_heads,
            d=self.kv_proj_dim,
        )
        attn_output = self.o(attn_output)

        return AttentionOutput(
            hidden_states=attn_output,
            attn_weights=attn_weights if output_attentions else None,
        )


class TimeSelfAttention(nn.Module):
    """
    The self-attention module in the Transformer block for time series data, which applies multi-head attention with RoPE and a residual connection.
    """

    def __init__(self, config: LiteSpecFormerConfig):
        super().__init__()

        # Create the multi-head attention layer with RoPE
        self.self_attention = MHA(config, use_rope=True)

        # Create the layer normalization layer and dropout layer for the residual connection
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

        # Create the dropout layer for the residual connection
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.FloatTensor,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        """
        The forward pass of the TimeSelfAttention module in the Transformer block for time series data from `LiteSpecFormer`.

        Parameters
        ----------
        hidden_states
            The input hidden states tensor with shape [batch, seq_len, d_model].
        attention_mask
            The attention mask tensor with shape [batch, num_heads, q_len, kv_len].
        position_ids
            The position IDs tensor with shape [batch, seq_len].
        output_attentions
            Whether to return the attention weights.

        Returns
        -------
        attn_output
            The attention output class from `AttentionOutput`.
        """
        # Perform the pre-normalization before the self-attention
        normed_hidden_states = self.layer_norm(hidden_states)

        # Forward pass through the self-attention layer
        attention_output: AttentionOutput = self.self_attention(
            hidden_states=normed_hidden_states,
            position_ids=position_ids,
            mask=attention_mask,
            output_attentions=output_attentions,
        )
        # Add the attention output to the input hidden states with a residual connection
        hidden_states = hidden_states + self.dropout(attention_output[0])

        return AttentionOutput(
            hidden_states=hidden_states, attn_weights=attention_output.attn_weights
        )


class FrequencyFilter(nn.Module):
    """
    Frequency-domain filter with learnable complex weights and optional adaptive high-frequency suppression.

    Transforms the input along the time axis with a real FFT, applies per-channel complex
    multipliers, and optionally routes high-energy frequency bins through a separate learnable
    branch masked by :meth:`create_adaptive_high_freq_mask`. The filtered spectrum is mapped
    back to the time domain with an inverse real FFT.

    When ``adaptive_filter`` is enabled, the module combines a global frequency weighting path
    (``complex_weight``) with a high-frequency path (``complex_weight_high``) gated by a
    learnable, energy-based soft mask controlled by ``threshold_param``.

    Parameters
    ----------
    adaptive_filter
        If ``True``, apply adaptive high-frequency masking and add the masked branch to the
        base frequency-weighted output.
    d_model
        Number of input channels; each channel has its own learnable complex weight.
    norm
        FFT normalization mode (e.g. ``"ortho"``). Passed to ``torch.fft.rfft`` and
        ``torch.fft.irfft`` in :meth:`forward`.
    """

    def __init__(
        self, adaptive_filter: bool, d_model: int, norm: Optional[str] = "ortho"
    ) -> None:
        super().__init__()

        self.complex_weight_high = nn.Parameter(
            torch.randn(d_model, 2, dtype=torch.float32) * 0.02
        )
        self.complex_weight = nn.Parameter(
            torch.randn(d_model, 2, dtype=torch.float32) * 0.02
        )

        nn.init.trunc_normal_(self.complex_weight_high, std=0.02)
        nn.init.trunc_normal_(self.complex_weight, std=0.02)

        self.threshold_param = nn.Parameter(torch.rand(1))

        self.adaptive_filter = adaptive_filter
        self.norm = norm

    def create_adaptive_high_freq_mask(
        self, x_fft: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Build a soft, learnable mask that emphasizes high-energy frequency bins.

        Per-batch median energy is used to normalize the magnitude-squared spectrum.
        Bins whose normalized energy exceeds ``threshold_param`` receive a mask near 1;
        weaker bins are scaled down. A straight-through estimator keeps
        ``threshold_param`` differentiable.

        Parameters
        ----------
        x_fft
            Complex FFT coefficients with shape ``(batch, num_freq_bins, d_model)``.

        Returns
        -------
        torch.FloatTensor
            Adaptive mask with shape ``(batch, num_freq_bins, d_model, 1)``, broadcastable
            over the complex frequency dimension.
        """
        B, _, _ = x_fft.shape

        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(B, 1)

        normalized_energy = energy / (median_energy + 1e-6)

        adaptive_mask = (
            (normalized_energy > self.threshold_param).float() - self.threshold_param
        ).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply frequency-domain filtering and return time-domain features.

        Parameters
        ----------
        x_in
            Input tensor with shape ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.FloatTensor
            Filtered tensor with the same shape as ``x_in``.
        """
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        x_fft = torch.fft.rfft(x, dim=1, norm=self.norm)
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm=self.norm)

        x = x.to(dtype)
        x = x.view(B, N, C)

        return x


class ResidualBlock(nn.Module):
    """
    A generic residual block which can be used for input and output embedding layers from the Time Series Foundation Models.

    Parameters
    ----------
    in_dim
        The dimension of the input tensor.
    h_dim
        The dimension of the hidden layer.
    out_dim
        The dimension of the output tensor.
    act_fn_name
        The name of the activation function to use.
    dropout_p
        The dropout probability, by default 0.0.
    use_layer_norm
        Whether to use layer normalization, by default False.
    """

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = Chronos2LayerNorm(hidden_size=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the ResidualBlock for the input and output embedding layers."""
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out
