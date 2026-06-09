import copy
from dataclasses import dataclass
from typing import cast, Union, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat
from transformers.modeling_utils import PreTrainedModel

from .config import LiteSpecFormerConfig, LiteSpecFormerForecastingConfig
from .layers import (
    MHA,
    RMSNorm,
    Chronos2LayerNorm,
    FeedForward,
    TimeSelfAttention,
    DepthWiseConv,
    ResidualBlock,
)
from .output import (
    AttentionOutput,
    LiteSpecFormerEncoderBlockOutput,
    LiteSpecFormerEncoderOutput,
    LiteSpecFormerOutput,
)
from .utils import Patch, InstanceNorm, calculate_batch_autocorrelation_function


class LiteSpecFormerEncoderBlock(nn.Module):
    """The encoder block in `LiteSpecFormer`, consisting of a time attention layer and a feed forward layer."""

    def __init__(self, config: LiteSpecFormerConfig) -> None:
        super().__init__()

        # Create the attention layer
        self.attention = TimeSelfAttention(config)

        # Create the feed forward layer
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> LiteSpecFormerEncoderBlockOutput:
        """
        The forward pass of a single encoder block in `LiteSpecFormer`.

        Parameters
        ----------
        hidden_states
            The input hidden states of shape (batch_size, seq_length, d_model)
        position_ids
            The position ids of shape (batch_size, seq_length) for the time attention layer
        attention_mask
            The attention mask of shape (batch_size, 1, 1, seq_length) for the time attention layer
        output_attentions
            Whether to return attention weights, by default False

        Returns
        -------
        LiteSpecFormerEncoderBlockOutput containing:
            - hidden_states: The output hidden states of shape (batch_size, seq_length, d_model)
            - attn_weights: The attention weights of shape (batch_size, num_heads, seq_length, seq_length) if output_attentions=True, else None
        """
        # Forward the attention layer
        attention_output: AttentionOutput = self.attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = attention_output[0]

        # Forward the feed forward layer
        hidden_states = self.feed_forward(hidden_states)

        return LiteSpecFormerEncoderBlockOutput(
            hidden_states=hidden_states,
            attn_weights=attention_output.attn_weights,
        )


class LiteSpecFormerEncoder(nn.Module):
    """The encoder in `LiteSpecFormer`, consisting of a stack of `LiteSpecFormerEncoderBlock`."""

    def __init__(self, config: LiteSpecFormerConfig) -> None:
        super().__init__()

        # The encoder in `LiteSpecFormer` is not a decoder
        assert not config.is_decoder

        # Create a stack of encoder blocks
        self.block = nn.ModuleList(
            [
                LiteSpecFormerEncoderBlock(config=config)
                for _ in range(config.num_layers)
            ]
        )

        # The final layer norm after the stack of encoder blocks
        self.final_layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

        # The dropout layer after the final layer norm
        self.dropout = nn.Dropout(config.dropout_rate)

    @staticmethod
    def _expand_and_invert_time_attention_mask(
        attention_mask: torch.Tensor, floating_type: torch.dtype
    ) -> torch.Tensor:
        """
        Expand the time attention mask from shape (batch_size, seq_length) to
        (batch_size, 1, 1, seq_length) and invert it for use in attention scores.

        Parameters
        ----------
        attention_mask
            The time attention mask of shape (batch_size, seq_length) where 1 indicates valid tokens and 0 indicates masked tokens.
        floating_type
            The floating point type to which the attention mask should be converted, typically the same as the model.dtype for compatibility with attention score calculations.

        Returns
        -------
        The expanded and inverted attention mask of shape (batch_size, 1, 1, seq_length)
        where valid tokens have a value of 0.0 and masked tokens have a large negative value (e.g., -inf)
        that can be added to attention scores to effectively mask them out.
        """
        assert (
            attention_mask.ndim == 2
        ), "attention_mask must have shape (batch, seq_len)"

        # Add new dims for attention heads and q_len
        attention_mask = attention_mask[:, None, None, :]

        # Invert binary mask to float mask which can be added to attention scores
        attention_mask = attention_mask.to(dtype=floating_type)
        attention_mask = (1.0 - attention_mask) * torch.finfo(floating_type).min

        return attention_mask

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        *,
        attention_mask: Union[torch.Tensor, None] = None,
        position_ids: Union[torch.Tensor, None] = None,
        output_attentions: bool = False,
    ) -> LiteSpecFormerEncoderOutput:
        """
        The forward pass of the encoder backbone in `LiteSpecFormer`.

        Parameters
        ----------
        inputs_embeds
            The input embeddings of shape (batch_size, seq_length, d_model) to the encoder.
        attention_mask
            The time attention mask of shape (batch_size, seq_length) where 1 indicates valid tokens and 0 indicates masked tokens.
            If None, no tokens will be masked.
        position_ids
            The position ids of shape (batch_size, seq_length) for the time attention layer. If None, position ids will be generated as a range from 0 to seq_length-1.
        output_attentions
            Whether to return attention weights, by default False

        Returns
        -------
        LiteSpecFormerEncoderOutput containing:
        """
        batch_size, seq_length = inputs_embeds.size()[:-1]

        if position_ids is None:
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0)

        all_attn_weights: tuple[torch.Tensor, ...] = ()

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                seq_length,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )

        # make the time attention mask broadcastable to attention scores (batch, n_heads, q_len, kv_len) and invert
        extended_attention_mask = self._expand_and_invert_time_attention_mask(
            attention_mask, inputs_embeds.dtype
        )

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module) in enumerate(self.block):
            layer_outputs: LiteSpecFormerEncoderBlockOutput = layer_module(
                hidden_states,
                position_ids=position_ids,
                attention_mask=extended_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                assert layer_outputs.attn_weights is not None

                all_attn_weights = (
                    *all_attn_weights,
                    layer_outputs.attn_weights,
                )

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return LiteSpecFormerEncoderOutput(
            last_hidden_state=hidden_states,
            all_attn_weights=all_attn_weights,
        )


class LiteSpecFormerModel(PreTrainedModel):
    """
    The `LiteSpecFormerModel` class implements the main architecture of the `LiteSpecFormer` model for time series forecasting.
    It consists of an input patch embedding layer, a stack of `LiteSpecFormerEncoder` blocks, and an output patch embedding layer for quantile prediction.
    The model is designed to handle univariate time series data in a channels-independent format and supports patch-based inputs for efficient modeling of long sequences.
    """

    config_class = LiteSpecFormerConfig  # type: ignore[assignment]
    _supports_long_horizon: bool = True
    _supports_sdpa: bool = True
    _supports_flash_attn_2 = True

    def __init__(self, config: LiteSpecFormerConfig) -> None:
        super().__init__(config)
        assert hasattr(
            config, "forecasting_config"
        ), "Not a valid huggingface config for LiteSpecFormer model"

        # The forecasting_config contains the specific configurations for the time series forecasting task,
        # such as patch size, context length, quantiles, etc.
        self.forecasting_config: LiteSpecFormerForecastingConfig = (
            config.forecasting_config
        )

        # The context length is the maximum length of the historical time series data that the model can take as input.
        self.context_length = self.forecasting_config.context_length

        # To ensure that the predicted autoregressive paradigm works correctly
        # The input and output dimensions of the model must be equal.
        assert (
            self.forecasting_config.input_patch_size
            == self.forecasting_config.output_patch_size
        ), (
            "input_patch_size and output_patch_size sizes must be equal, "
            f"but found {self.forecasting_config.input_patch_size} and {self.forecasting_config.output_patch_size}"
        )

        # Only [PAD] token (and [REG] token)
        if self.forecasting_config.use_reg_token:
            config.reg_token_id = 1
        config.vocab_size = 2 if self.forecasting_config.use_reg_token else 1
        if self.forecasting_config.use_reg_token:
            self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Input patch embedding layer
        self.input_patch_embedding = ResidualBlock(
            # x3 for [time_embedding, patch, patch_mask]
            in_dim=self.forecasting_config.input_patch_size * 3,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # patching layer
        self.patch = Patch(
            patch_size=self.forecasting_config.input_patch_size,
            patch_stride=self.forecasting_config.input_patch_stride,
        )

        # instance normalization, also referred to as "scaling" in GluonTS
        self.instance_norm = InstanceNorm(
            use_arcsinh=self.forecasting_config.use_arcsinh
        )

        # The encoder backbone of the model, consisting of a stack of `LiteSpecFormerEncoderBlock`.
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = LiteSpecFormerEncoder(encoder_config)

        # The quantiles for prediction, registered as a buffer since they are not learnable parameters
        # but should be part of the model state for saving and loading.
        quantiles = torch.tensor(
            self.forecasting_config.quantiles, dtype=self.dtype
        ).squeeze(dim=0)
        self.quantiles: torch.Tensor
        self.register_buffer("quantiles", quantiles, persistent=False)
        # Get the length of the quantiles for later use in the output patch embedding layer and loss calculation
        self.num_quantiles = len(self.quantiles)

        # Output patch embedding layer for quantile prediction,
        # which takes the hidden states from the encoder and produces quantile predictions for the future time steps.
        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            # The shape of the output of this module is the quantile multiplied by the output patch_len.
            out_dim=self.num_quantiles * self.forecasting_config.output_patch_size,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # Whether to use the Autocorrelation loss for model training
        self.use_acf_loss = config.use_acf_loss

        # The index of the median quantile, used for the ACF loss calculation when use_acf_loss is True
        self.medium_index = self.forecasting_config.medium_index
        self.history_token_number = self.config.history_token_number
        self.n_lags = self.config.n_lags

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module) -> None:
        """
        Initialize the weights of the `LiteSpecFormer` model.
        Including the initialization of the weights of the LayerNorm layer in the Transformer backbone,
        the weights of the RMSNorm layer in the Feedforward Network,
        and the weights of the depthwise and pointwise CNN in the Feedforward Network.
        """
        super()._init_weights(module)

        # Upload the factor for model initialization
        factor = self.config.initializer_factor

        if isinstance(module, Chronos2LayerNorm):
            # Initialize the weights of the LayerNorm in the Transformer backbone
            module.weight.data.fill_(factor * 1.0)

        elif isinstance(module, RMSNorm):
            # Initialize the weights of the RMSNorm in the Transformer backbone
            module.weight.data.fill_(factor * 1.0)

        elif isinstance(module, DepthWiseConv):
            # Initialize the depthwise and pointwise CNN in the Feedforward Network

            # Initialize the weights of the depthwise convolution
            module.depth_wise.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
            )
            # Initialize the biases of the depthwise convolution
            if hasattr(module, "bias") and module.depth_wise.bias is not None:
                module.depth_wise.bias.data.zero_()

            # Initialize the weights of the pointwise convolution
            module.point_wise.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
            )
            # Initialize the biases of the pointwise convolution
            if hasattr(module, "bias") and module.point_wise.bias is not None:
                module.point_wise.bias.data.zero_()

        elif isinstance(module, MHA):
            # Initialize the weights of the query, key, and value layers

            # Upload the factor for model initialization
            d_model = self.config.d_model
            kv_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads

            # Initialize the weights of the query, key, and value layers
            module.q.weight.data.normal_(
                mean=0.0, std=factor * ((d_model * kv_proj_dim) ** -0.5)
            )
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            # The finnal projection layer
            module.o.weight.data.normal_(
                mean=0.0, std=factor * ((n_heads * kv_proj_dim) ** -0.5)
            )

        elif isinstance(module, LiteSpecFormerModel):
            # Initialize the weights of the learnable embedding layer
            if self.forecasting_config.use_reg_token:
                module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)

        elif isinstance(module, ResidualBlock):
            # Initialize the weights of the embedding and output layers
            module.hidden_layer.weight.data.normal_(
                mean=0.0,
                std=factor * (module.hidden_layer.weight.size(-1) ** -0.5),
            )
            if (
                hasattr(module.hidden_layer, "bias")
                and module.hidden_layer.bias is not None
            ):
                module.hidden_layer.bias.data.zero_()

            # The hidden residual layer
            module.residual_layer.weight.data.normal_(
                mean=0.0,
                std=factor * (module.residual_layer.weight.size(-1) ** -0.5),
            )
            if (
                hasattr(module.residual_layer, "bias")
                and module.residual_layer.bias is not None
            ):
                module.residual_layer.bias.data.zero_()

            # The final output layer for the time series forecasting
            module.output_layer.weight.data.normal_(
                mean=0.0, std=factor * (module.output_layer.weight.size(-1) ** -0.5)
            )
            if (
                hasattr(module.output_layer, "bias")
                and module.output_layer.bias is not None
            ):
                module.output_layer.bias.data.zero_()

    def _validate_input(
        self,
        context: torch.Tensor,
        context_mask: Union[torch.Tensor, None],
        num_output_patches: int,
        future_target: Union[torch.Tensor, None],
        future_target_mask: Union[torch.Tensor, None],
    ) -> None:
        """
        This method is called at the beginning of the model's forward propagation to check the validity of the input data.
        We mainly check whether the shapes of the model's input and output, as well as the shape of their masks, are valid.
        A ValueError will be triggered when invalid input is found.
        This check is crucial for ensuring the accuracy of each autoregressive prediction.

        Parameters
        ----------
        context
            Input tensor of shape (batch_size, context_length) containing the historical values in univariate channels-independent format
        context_mask
            Binary mask tensor of same shape as context indicating which values are valid (1) vs missing (0)
            If missing, the context_mask will be automatically constructed based on the NaN values in context.
        num_output_patches
            Number of output patches to generate predictions for, by default 1
        future_target
            Target tensor of shape (batch_size, future_length) used during training.
        future_target_mask
            Binary mask tensor of same shape as `future_target` indicating which values are valid (1) vs missing (0)
            If missing, the `future_target_mask` will be automatically constructed based on the NaN values in `future_target`.
        output_attentions
            Whether to return attention weights, by default False

        Returns
        -------
        None
        """

        # Get the output patch size from config
        output_patch_size = self.forecasting_config.output_patch_size

        # Check the shape of the input time series data
        if context.ndim != 2:
            # The input time series data must be univariate in channels-independent format
            raise ValueError(
                f"context must have shape (batch_size, context_length), found: {tuple(context.shape)}"
            )

        if context_mask is not None and context_mask.shape != context.shape:
            # Check the shape of the context mask for the time series data
            raise ValueError(
                f"mask must have shape {tuple(context.shape)}, found: {tuple(context_mask.shape)}"
            )

        # Check the shape of the future target
        if future_target is not None:
            if future_target.shape[0] != context.shape[0] or future_target.ndim != 2:
                # The batch size of the future target must match the batch size of the context
                raise ValueError(
                    f"future_target must have shape (batch_size={context.shape[0]}, future_length), found: {tuple(future_target.shape)}"
                )

            if future_target.shape[-1] > output_patch_size * num_output_patches:
                # The number of output patches must be large enough to accommodate the length of future_target
                raise ValueError(
                    f"{num_output_patches=} must be large enough to accommodate the length of future_target, "
                    f"found: {future_target.shape[-1]} > {num_output_patches} * {output_patch_size}"
                )

        if future_target_mask is not None:
            # Check the shape of the future target mask

            if future_target is None:
                raise ValueError(
                    "future_target must be provided if future_target_mask is provided"
                )
            if future_target_mask.shape != future_target.shape:
                raise ValueError(
                    f"future_target_mask must have the same shape as future_target, found: {tuple(future_target_mask.shape)} and {tuple(future_target.shape)}"
                )

    def _prepare_patched_context(
        self, context: torch.Tensor, context_mask: Union[torch.Tensor, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        This method prepares the input time series data for the encoder by performing patching,
        instance normalization, and time encoding.

        Parameters
        ----------
        context
            Input tensor of shape (batch_size, context_length) containing the historical values in univariate channels-independent format
        context_mask
            Binary mask tensor of same shape as context indicating which values are valid (1) vs missing (0)
            If missing, the context_mask will be automatically constructed based on the NaN values in context.

        Returns
        -------
        patched_context
            The patched and processed context tensor of shape (batch_size, num_patches, input_patch_size) ready for input to the encoder.
            The last dimension is the concatenation of time encoding, patch values, and patch mask.
        attention_mask
            The attention mask for the encoder of shape (batch_size, num_patches)
            where 1 indicates valid patches and 0 indicates masked patches.
        loc_scale
            A tuple of tensors (loc, scale) containing the location and scale parameters from instance normalization,
            which can be used for denormalization of predictions later.
        """

        # 检查是否有掩码，如果没有则创建它
        context_mask = (
            context_mask.to(context.dtype)
            if context_mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )

        # Get the shape of the input time series data
        batch_size, context_length = context.shape

        # truncate context if it's longer than model's context length
        if context_length > self.context_length:
            context = context[..., -self.context_length :]
            context_mask = context_mask[..., -self.context_length :]

        # scaling from the ReVIN to the inputs
        context, loc_scale = self.instance_norm(context)

        # scaling is done in 32-bit precision, then the context is moved to model's dtype
        context = context.to(self.dtype)
        context_mask = context_mask.to(self.dtype)

        # DO patching operation for the channels-independent time series data
        patched_context = self.patch(context)

        # Zero-fill missing values
        patched_mask = torch.nan_to_num(self.patch(context_mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)

        # attention_mask = 1 if at least one item in the patch is observed
        attention_mask = patched_mask.sum(dim=-1) > 0  # (batch_size, num_patches)
        num_context_patches = attention_mask.shape[-1]

        # context time encoding: every observation is assigned a sequential time index,
        # scaled by model's context length = [-C, -(C-1), ..., -1] / context_length
        # It provides explicit information about temporal ordering to the model which
        # is beneficial when using patch-based inputs.
        final_context_length = (
            num_context_patches * self.forecasting_config.input_patch_size
        )
        context_time_enc = torch.arange(
            start=-final_context_length, end=0, device=self.device, dtype=torch.float32
        )
        context_time_enc = (
            repeat(
                context_time_enc,
                "(n p) -> b n p",
                b=batch_size,
                n=num_context_patches,
                p=self.forecasting_config.input_patch_size,
            )
            .div(cast(int, self.forecasting_config.time_encoding_scale))
            .to(self.dtype)
        )

        # concat time encoding, context and mask along the last (feature) dim
        patched_context = torch.cat(
            [context_time_enc, patched_context, patched_mask], dim=-1
        )

        return patched_context, attention_mask, loc_scale

    def _compute_loss(
        self,
        context: torch.Tensor,  # (batch_size,  num_patches, context_length)
        quantile_preds: torch.Tensor,  # [batch_size, num_quantiles, pred_length]
        future_target: torch.Tensor,  # [batch_size, future_length]
        future_target_mask: Union[torch.Tensor, None],
        loc_scale: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.FloatTensor:
        """
        Compute the training loss for the `LiteSpecFormer` model,
        which consists of a quantile regression loss and an optional autocorrelation function (ACF) loss.

        If the model has future_targets in the forward propagation,
        its output will automatically include the calculated loss.

        Parameters
        ----------
        context
            The input look-back windows for the model to make predictions, of shape (batch_size, num_patches, context_length)
        quantile_preds
            The quantile predictions from the model of shape (batch_size, num_quantiles, pred_length)
        future_target
            The ground truth future target values of shape (batch_size, future_length) used for calculating the loss during training.
        future_target_mask
            The mask for the future target values of shape (batch_size, future_length).
        loc_scale
            The location and scale parameters for instance normalization.

        Returns
        -------
        loss
            The computed loss as a single scalar tensor.
        """
        batch_size = future_target.shape[0]
        # output_patch_size = self.forecasting_config.output_patch_size

        # Check the batch_size shape of quantile_preds
        assert (
            quantile_preds.shape[0] == batch_size
            and quantile_preds.shape[-1] >= future_target.shape[-1]
        )

        # normalize target and mask
        future_target, _ = self.instance_norm(future_target, loc_scale)

        if self.use_acf_loss:
            # Here, the autocorrelation function of the model is calculated.
            history_context = context[:, -self.history_token_number :, :]
            history_context = rearrange(
                tensor=history_context, pattern="b n p -> b (n p)"
            )
            output = quantile_preds[:, self.medium_index, :]

            # Combine historical inputs and real labels
            combined_targets = torch.cat([history_context, future_target], dim=-1)

            # The historical input and the predicted sequence are concatenated.
            combined_predictions = torch.cat([history_context, output], dim=-1)

            # Calculate the autocorrelation functions of the prediction and the label separately.
            target_acf = calculate_batch_autocorrelation_function(
                combined_targets, max_lag=self.n_lags
            )
            pred_acf = calculate_batch_autocorrelation_function(
                combined_predictions, max_lag=self.n_lags
            )
            # Calculate the ACF loss as 1 - cosine similarity between the predicted ACF
            # and the target ACF, averaged over the batch.
            acf_loss = 1 - F.cosine_similarity(pred_acf, target_acf, dim=-1).mean()

        else:
            # If ACF loss is not used, set it to 0 so that it does not contribute to the total loss.
            acf_loss = 0

        # reshape future_target and future_target_mask to (batch_size, 1, future_length) for broadcasting with quantile_preds
        future_target = future_target.unsqueeze(1)
        future_target = future_target.to(self.device)

        # create future_target_mask if it's not provided, where valid values are 1 and missing values are 0
        future_target_mask = (
            future_target_mask.unsqueeze(1).to(self.device)
            if future_target_mask is not None
            else ~torch.isnan(future_target)
        )
        future_target = torch.where(future_target_mask > 0.0, future_target, 0.0)

        # pad target and target_mask if they are shorter than model's prediction
        if quantile_preds.shape[-1] > future_target.shape[-1]:
            padding_shape = (
                *future_target.shape[:-1],
                quantile_preds.shape[-1] - future_target.shape[-1],
            )
            future_target = torch.cat(
                [future_target, torch.zeros(padding_shape).to(future_target)], dim=-1
            )
            future_target_mask = torch.cat(
                [future_target_mask, torch.zeros(padding_shape).to(future_target_mask)],
                dim=-1,
            )

        # Reshape the quantile predictions and future target to
        # (batch_size, num_quantiles, num_output_patches * output_patch_size)
        quantiles = rearrange(self.quantiles, "num_quantiles -> 1 num_quantiles 1")
        quantile_loss = 2 * torch.abs(
            (future_target - quantile_preds)
            * ((future_target <= quantile_preds).float() - quantiles)
        )
        # mean over prediction horizon, sum over quantile levels and mean over batch
        quantile_loss = quantile_loss.mean(dim=-1).sum(dim=-1).mean()

        # FIXME: the first components masks any missing targets and the second component masks known future values
        # loss_mask = future_target_mask.float()
        # loss = quantile_loss * loss_mask

        # mean over prediction horizon, sum over quantile levels and mean over batch
        loss = quantile_loss + acf_loss

        return loss

    def encode(
        self,
        context: Union[torch.Tensor, None] = None,  # (batch_size, context_length)
        context_mask: Union[torch.Tensor, None] = None,
        num_output_patches: Union[torch.Tensor, None, int] = 1,
        future_target: Union[torch.Tensor, None] = None,
        future_target_mask: Union[torch.Tensor, None] = None,
        output_attentions: bool = False,
    ) -> Tuple[LiteSpecFormerEncoderOutput]:
        """
        Encode the input context and future target (if provided) into hidden states.

        For a deep learning model with `Transformer` backbone like `LiteSpecFormer`,
        the encoding of the input time series data is a crucial step that directly affects the model's ability to learn and make accurate predictions.

        To facilitate the embedding of time series data,
        we specifically designed an encoder function to process the input time series data
        and convert it into a format suitable for the Transformer encoder.

        Parameters
        ----------
        context
            Input tensor of shape (batch_size, context_length) containing the historical values in univariate channels-independent format
        context_mask
            Binary mask tensor of same shape as context indicating which values are valid (1) vs missing (0)
            If missing, the context_mask will be automatically constructed based on the NaN values in context.
        num_output_patches
            Number of output patches to generate predictions for, by default 1
        future_target
            Target tensor of shape (batch_size, future_length) used during training.
        future_target_mask
            Binary mask tensor of same shape as `future_target` indicating which values are valid (1) vs missing (0)
            If missing, the `future_target_mask` will be automatically constructed based on the NaN values in `future_target`.
        output_attentions
            Whether to return attention weights, by default False

        Returns
        -------
        encoder_outputs
            The output of the encoder backbone, containing the last hidden state and optionally the attention weights.
        patched_context
            The patched and processed context tensor of shape (batch_size, num_patches, input_patch_size) ready for input to the encoder.
            The last dimension is the concatenation of time encoding, patch values, and patch mask.
        loc_scale
            A tuple of tensors (loc, scale) containing the location and scale parameters from instance normalization,
            which can be used for denormalization of predictions later.
        num_context_patches
            The number of context patches after patching, which is used for slicing the encoder outputs for prediction.
        """
        # Validate the input data to ensure it has the correct shape and format for processing.
        self._validate_input(
            context=context,
            context_mask=context_mask,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
        )

        batch_size = context.shape[0]

        # Get the patched context, attention mask, and loc_scale for normalization
        patched_context, attention_mask, loc_scale = self._prepare_patched_context(
            context=context, context_mask=context_mask
        )
        num_context_patches = attention_mask.shape[-1]

        # get input embeddings of shape (batch, num_context_patches, d_model)
        input_embeds: torch.FloatTensor = self.input_patch_embedding(patched_context)

        # append [REG] special token embedding, if needed
        if self.forecasting_config.use_reg_token:
            reg_input_ids = torch.full(
                (batch_size, 1), self.config.reg_token_id, device=input_embeds.device
            )

            reg_embeds = self.shared(reg_input_ids)

            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)

            attention_mask = torch.cat(
                [
                    attention_mask.to(self.dtype),
                    torch.ones_like(reg_input_ids).to(self.dtype),
                ],
                dim=-1,
            )

        # Forward the encoder backbone to get the encoder outputs
        encoder_outputs: LiteSpecFormerEncoderOutput = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            output_attentions=output_attentions,
        )

        return encoder_outputs, patched_context, loc_scale, num_context_patches

    def forward(
        self,
        context: torch.Tensor,
        context_mask: Union[torch.Tensor, None] = None,
        num_output_patches: int = 1,
        future_target: Union[torch.Tensor, None] = None,
        future_target_mask: Union[torch.Tensor, None] = None,
        output_attentions: bool = False,
    ) -> LiteSpecFormerOutput:
        """The forward pass of the `LiteSpecFormer` model for zero-shot confidence spectrum prediction.

        Parameters
        ----------
        context
            Input tensor of shape (batch_size, context_length) containing the historical values
        context_mask
            Binary mask tensor of same shape as context indicating which values are valid (1) vs missing (0)
            If missing, the context_mask will be automatically constructed based on the NaN values in context.
        num_output_patches
            Number of output patches to generate predictions for, by default 1
        future_target
            Target tensor of shape (batch_size, future_length) used during training.
        future_target_mask
            Binary mask tensor of same shape as `future_target` indicating which values are valid (1) vs missing (0)
            If missing, the `future_target_mask` will be automatically constructed based on the NaN values in `future_target`.
        output_attentions
            Whether to return attention weights, by default False

        Returns
        -------
        LiteSpecFormerOutput containing:
        - loss: Training loss, if `future_target` is provided
        - quantile_preds: Quantile predictions of shape (batch_size, num_quantiles, num_output_patches * output_patch_size).
            quantile_preds will contain an entry for every time series in the context batch regardless of whether it was a
            known future covariate.
        """

        batch_size = context.shape[0]

        # Encode the input context and future target (if provided) to get the encoder outputs,
        # patched context, loc_scale for normalization, and number of context patches.
        encoder_outputs, patched_context, loc_scale, num_context_patches = self.encode(
            context=context,
            context_mask=context_mask,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
            output_attentions=output_attentions,
        )

        hidden_states: torch.Tensor = encoder_outputs[0]

        assert hidden_states.shape == (
            batch_size,
            num_context_patches,
            self.config.d_model,
        )

        # slice the last num_output_patches hidden states to be input into the output_patch_embedding
        forecast_embeds = hidden_states[
            :, -num_output_patches:
        ]  # Obtain the output of the last patch in the encoder architecture.
        quantile_preds: torch.Tensor = self.output_patch_embedding(
            forecast_embeds
        )  # [batch_sized, num_patches, num_quantiles * patch_size]

        # reshape quantile_preds to (batch, num_output_patches, num_quantiles * output_patch_size)
        quantile_preds = rearrange(
            quantile_preds,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.num_quantiles,
            p=self.forecasting_config.output_patch_size,
        )  # [batch_size, num_quantiles, num_patches * patch_size]

        # Compute the loss if future_target is provided, otherwise set loss to None for inference.
        loss = (
            self._compute_loss(
                context=patched_context,
                quantile_preds=quantile_preds,
                future_target=future_target,
                future_target_mask=future_target_mask,
                loc_scale=loc_scale,
            )
            if future_target is not None
            else None
        )

        # Unscale predictions
        quantile_preds = rearrange(
            quantile_preds,
            "b q h -> b (q h)",
            b=batch_size,
            q=self.num_quantiles,
            h=num_output_patches * self.forecasting_config.output_patch_size,
        )  # [batch_size, num_quantiles * pred_length]

        # Doing the reverse scaling to get the predictions back to the original scale of the input time series data.
        quantile_preds = self.instance_norm.inverse(quantile_preds, loc_scale)

        # reshape quantile_preds back to (batch_size, num_quantiles, num_output_patches * output_patch_size) for output
        quantile_preds = rearrange(
            quantile_preds,
            "b (q h) -> b q h",
            q=self.num_quantiles,
            h=num_output_patches * self.forecasting_config.output_patch_size,
        )

        return LiteSpecFormerOutput(
            loss=loss,
            context=context,
            quantile_preds=quantile_preds,
            enc_attn_weights=encoder_outputs.all_attn_weights,
        )
