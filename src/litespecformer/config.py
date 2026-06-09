from dataclasses import dataclass
from typing import List, Literal, Optional, Dict
from enum import Enum

from transformers.configuration_utils import PretrainedConfig


class DatasetMode(str, Enum):
    # Dataset split modes used across training and evaluation.
    TRAIN = "train"  # Training split
    VALIDATION = "validation"  # Validation split
    TEST = "test"  # Test split


@dataclass
class LiteSpecFormerConfig(PretrainedConfig):
    """
    The Huggingface transformer-style pre-trained model config for `LiteSpecFormer` model
    for zero-shot spectrum confidence prediction.

    Parameters
    ----------
    d_model
        Size model's hidden states of `LiteSpecFormer`, by default 512
    d_kv
        Size of the key, query, value projections per attention head, by default 64
    d_ff
        Size of the intermediate feed forward layers, by default 2048
    num_layers
        Number of hidden layers in the encoder, by default 6
    num_heads
        Number of attention heads for each attention layer, by default 8
    dropout_rate
        The ratio for all dropout layers, by default 0.1
    kernel_size
        The kernel size for the (depth-wise) convolution in the feed-forward network, by default 3
    attn_output_gate
        The type of the gate for the attention output, by default "headwise". Options: "headwise", "elementwise", "none"
    use_channel_attention
        Whether to use the channel attention in the feed forward network, by default False
    use_dw_cnn
        Whether to use the depth-wise convolution in the feed forward network, by default False
    use_acf_loss
        Whether to use the Autocorrelation loss, by default False
    layer_norm_epsilon
        The epsilon used by the layer normalization layers, by default 1e-6
    initializer_factor
        A factor for initializing all weight matrices, by default 0.05
    feed_forward_proj
        Type of feed forward layer to be used, by default "gelu"
    vocab_size
        Size of vocabulary for special tokens, by default 2
    pad_token_id
        Token ID for padding/missing value token, by default 0
    rope_theta
        The base theta for rotary position embedding (RoPE), by default 10000.0
    attn_implementation
        The attention implementation to use. Options: "eager" or "sdpa", by default None (uses "sdpa")
    """

    # Create the attribute map for backward compatibility with the original config keys used in the paper and codebase
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "head_dim": "d_kv",
    }
    model_type = "time_series_transformer"

    def __init__(
        self,
        d_model: int = 512,
        d_kv: int = 64,
        d_ff: int = 1536,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        kernel_size: int = 3,
        attn_output_gate: Optional[str] = "headwise",
        use_channel_attention: Optional[bool] = False,
        use_dw_cnn: Optional[bool] = False,
        use_acf_loss: Optional[bool] = False,
        layer_norm_epsilon: float = 1e-6,
        initializer_factor: float = 0.05,
        feed_forward_proj: str = "gelu",
        vocab_size: int = 2,
        pad_token_id: int = 0,
        rope_theta: float = 10000.0,
        attn_implementation: Literal["eager", "sdpa"] | None = None,
        forecasting_config: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        # Set the attributes for the model configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.kernel_size = kernel_size

        # Whether to use the channel attention in the feed forward network
        self.use_channel_attention = use_channel_attention

        # The selection of the gate attention
        self.attn_output_gate = attn_output_gate
        assert self.attn_output_gate in [
            "headwise",
            "elementwise",
            "none",
        ], f"attn_output_gate {self.attn_output_gate} not supported"

        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.rope_theta = rope_theta

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = False
        assert not self.is_gated_act, "gated activation is not supported"

        # Get the forecasting config
        self.forecasting_config = LiteSpecFormerForecastingConfig(
            **(forecasting_config or {})
        )

        # Attention implementation - default to "sdpa" if not specified
        attn_implementation = attn_implementation or "sdpa"
        assert attn_implementation in [
            "eager",
            "sdpa",
        ], f"attn_implementation {attn_implementation} not supported"

        # Wheather to use the dwcnn
        self.use_dw_cnn = use_dw_cnn

        # Whether to use the Autocorrelation loss
        self.use_acf_loss = use_acf_loss
        self.n_lags: int = 48
        self.history_token_number: int = 3

        self.reduction: int = 64

        # unused
        kwargs.pop("is_encoder_decoder", None)
        kwargs.pop("eos_token_id", None)

        super().__init__(
            pad_token_id=pad_token_id,
            is_encoder_decoder=False,
            attn_implementation=attn_implementation,
            **kwargs,
        )


@dataclass
class LiteSpecFormerForecastingConfig(object):
    """
    The forecasting configuration for `LiteSpecFormer` model,
    which contains the parameters related to the forecasting task,
    such as the input and output sequence lengths, the quantiles to predict, etc.

    Parameters
    ----------
    context_length
        The length of the input time series sequence, by default 768
    output_patch_size
        The length of the output time series patch or token, by default 16
    input_patch_size
        The length of the input time series patch or token, by default 16
    input_patch_stride
        The stride of the input time series patch or token, by default 16
    quantiles
        The quantiles to predict (a list of floats), by default None
    use_reg_token
        Whether to use the regression token, by default False
    """

    def __init__(
        self,
        context_length: int = 768,
        output_patch_size: int = 16,
        input_patch_size: int = 16,
        input_patch_stride: int = 16,
        quantiles: List[float] = None,
        use_reg_token: bool = False,
        max_output_patches: int = 1,
        use_arcsinh: bool = True,
        min_past: int = 32,
        time_encoding_scale: Optional[int] = None,
        medium_index: Optional[int] = None,
    ) -> None:
        # The length of the input time series sequence
        self.context_length: int = context_length

        # The length of the output time series patch or token
        self.output_patch_size: int = output_patch_size

        # The length of the input time series patch or token
        self.input_patch_size: int = input_patch_size

        # The stride of the input time series patch or token
        self.input_patch_stride: int = input_patch_stride

        # The quantiles to predict (a list of floats)
        self.quantiles: List[float] = (
            quantiles
            if quantiles is not None
            else [
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.95,
            ]
        )
        # The index of the median quantile (0.5) in the quantiles list,
        # used for loss calculation and evaluation
        self.medium_index: int = (
            self.quantiles.index(0.5)
            if 0.5 in self.quantiles
            else len(self.quantiles) // 2
        )

        # Whether to use the regression token
        self.use_reg_token: bool = use_reg_token

        self.use_arcsinh: bool = use_arcsinh
        self.max_output_patches: int = max_output_patches
        self.time_encoding_scale: int | None = context_length
        self.min_past = min_past


@dataclass
class LiteSpecFormerTrainingConfig(object):
    """
    The configuration for the pre-training the `LiteSpecFormer` model,
    which contains the parameters related to the training process,

    Parameters
    ----------
    batch_size
        The batch size for model training per device (GPU/TPU), by default 4096
    shuffle
        Whether to shuffle the training data, by default True
    test_batch_size
        The batch size for model evaluation, by default 256
    num_workers
        The number of worker processes for data loading, by default 0
    mixed_precision
        The mixed precision setting for training, by default "fp16". Options: "fp16", "bf16", "no"
    gradient_accumulation_steps
        The number of steps to accumulate gradients before performing an optimizer step, by default 1
    optimizer
        The optimizer to use for training, by default "AdamW"
    learning_rate
        The learning rate for the optimizer, by default 4e-4
    weight_decay
        The weight decay for the optimizer, by default 0.0
    beta1
        The beta1 parameter for the AdamW optimizer, by default 0.9
    beta2
        The beta2 parameter for the AdamW optimizer, by default 0.999
    epsilon
        The epsilon parameter for the AdamW optimizer, by default 1e-8
    num_training_steps
        The total number of training steps, by default 100000
    num_test_steps
        The number of steps between each evaluation on the test set, by default 500
    num_warmup_steps
        The number of warmup steps for learning rate scheduling, by default 1000
    num_subset_steps
        The number of steps after which to switch the training data subset, by default 10000000
    anneal_strategy
        The strategy for annealing the learning rate, by default "cos". Options: "cos", "linear"
    """

    def __init__(
        self,
        batch_size: int = 4096,
        shuffle: bool = True,
        test_batch_size: int = 1024,
        num_workers: int = 0,
        mixed_precision: str = "fp16",
        gradient_accumulation_steps: int = 1,
        optimizer: str = "AdamW",
        learning_rate: float = 4e-4,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        num_training_steps: int = 500000,
        num_test_steps: int = 500,
        num_warmup_steps: int = 1280,
        anneal_strategy: str = "cos",
    ) -> None:
        # The batch size for model training per device (GPU/TPU)
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle

        # The batch size for model evaluation
        self.test_batch_size: int = test_batch_size

        self.num_workers: int = num_workers

        self.mixed_precision: str = mixed_precision
        self.gradient_accumulation_steps: int = gradient_accumulation_steps

        # Parameters related to creating optimizer
        self.optimizer: str = optimizer
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon

        # Parameters for creating dynamic learning rate adjustment
        self.num_warmup_steps: int = num_warmup_steps
        self.anneal_strategy: str = anneal_strategy

        # Total batches of model training
        self.num_training_steps: int = num_training_steps
        # Test every how many steps
        self.num_test_steps: int = num_test_steps

        # Address where the model and training state are saved.
        self.checkpoint_path: str = "./checkpoints"

        # Address for storing training and validation data
        self.data_path: str = "data/Large-Spectrum-Prediction-Dataset-split/train"

        # Training data types
        self.train_data_class: str = "all"

        # Different context lengths for testing
        self.eval_context_lengths: List[int] = [128, 192, 256, 336, 512]
