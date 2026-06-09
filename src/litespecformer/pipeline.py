import logging
from copy import deepcopy
from datetime import datetime
import time
import math
import os
from os import path

from enum import Enum
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Sequence,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Callable,
    Literal,
    Mapping,
    Sequence,
)

from accelerate import Accelerator
from colorama import Fore, Style

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from einops import rearrange, repeat
from transformers import AutoConfig, PreTrainedModel
from transformers.utils.import_utils import is_peft_available
from transformers.utils.peft_utils import find_adapter_config_file

if TYPE_CHECKING:
    import pandas as pd
    from peft import LoraConfig
    from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)

from litespecformer.config import LiteSpecFormerConfig
from litespecformer.model import LiteSpecFormerModel
from litespecformer.dataset import (
    LiteSpecFormerDataset,
    SpectrumLibraryDataset,
    DatasetMode,
    TensorOrArray,
)
from litespecformer.utils import (
    left_pad_and_stack_1D,
    get_num_output_patches,
    interpolate_quantiles,
    weighted_quantile,
    logging_results,
)
from litespecformer.metrics import calculate_metrics


class ForecastType(Enum):
    SAMPLES = "samples"
    QUANTILES = "quantiles"


class PipelineRegistry(type):
    REGISTRY: Dict[str, "PipelineRegistry"] = {}

    def __new__(cls, name, bases, attrs):
        """See, https://github.com/faif/python-patterns."""
        new_cls = type.__new__(cls, name, bases, attrs)
        if name is not None:
            cls.REGISTRY[name] = new_cls

        return new_cls


class BaseForecastPipeline(metaclass=PipelineRegistry):
    """
    The BaseForecastPipeline class provides a common interface for
    forecasting tasks. It defines the basic methods that all forecasting
    pipelines must implement.

    The code is adapted from the Chronos2Pipeline class in the Chronos library
    in https://github.com/amazon-science/chronos-forecasting
    with modifications to fit the LiteSpecFormer model and use case.
    """

    forecast_type: ForecastType
    dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}

    def __init__(self, inner_model: "PreTrainedModel"):
        """
        Parameters
        ----------
        inner_model : PreTrainedModel
            A hugging-face transformers PreTrainedModel, e.g., T5ForConditionalGeneration
        """
        # for easy access to the inner HF-style model
        self.inner_model = inner_model

    def _prepare_and_validate_context(
        self, context: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Prepare and validate the context tensor.

        Parameters
        ----------
        context
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.

        Returns
        -------
        context
            The prepared and validated context tensor.
            Is a 2D time series tensor.
        """
        if isinstance(context, list):
            # Iterate through all the tensors in this list and fill them with the longest tensor.
            context: torch.FloatTensor = left_pad_and_stack_1D(context)

        # Handling the case where the input data is a single sample
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            # duplicate the series to make it a batch of size 1
            context = context.unsqueeze(0)

        # Ensure the input data is two-dimensional.
        assert context.ndim == 2

        return context

    def predict(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
    ):
        """
        Get forecasts for the given time series. Predictions will be
        returned in fp32 on the cpu.

        Parameters
        ----------
        inputs
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.
        prediction_length
            Time steps to predict. Defaults to a model-dependent
            value if not given.

        Returns
        -------
        forecasts
            Tensor containing forecasts. The layout and meaning
            of the forecasts values depends on ``self.forecast_type``.
        """
        raise NotImplementedError()

    def predict_quantiles(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get quantile and mean forecasts for given time series.
        Predictions will be returned in fp32 on the cpu.

        Parameters
        ----------
        inputs : Union[torch.Tensor, List[torch.Tensor]]
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.
        prediction_length : Optional[int], optional
            Time steps to predict. Defaults to a model-dependent
            value if not given.
        quantile_levels : List[float], optional
            Quantile levels to compute, by default [0.1, 0.2, ..., 0.9]

        Returns
        -------
        quantiles
            Tensor containing quantile forecasts. Shape
            (batch_size, prediction_length, num_quantiles)
        mean
            Tensor containing mean (point) forecasts. Shape
            (batch_size, prediction_length)
        """
        raise NotImplementedError()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        *model_args,
        **kwargs,
    ):
        """
        Load the model, either from a local path, S3 prefix, or from the HuggingFace Hub.
        Supports the same Parameters as ``AutoConfig`` and ``AutoModel`` from ``transformers``.
        """
        from transformers import AutoConfig

        torch_dtype = kwargs.get("torch_dtype", "auto")
        if torch_dtype != "auto" and isinstance(torch_dtype, str):
            kwargs["torch_dtype"] = cls.dtypes[torch_dtype]

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        is_valid_config = hasattr(config, "litespecformer_pipeline_class") or hasattr(
            config, "forecasting_config"
        )

        if not is_valid_config:
            raise ValueError("Not a LiteSpecFormer config file")

        pipeline_class_name = getattr(
            config, "litespecformer_pipeline_class", "LiteSpecFormerPipeline"
        )
        class_ = PipelineRegistry.REGISTRY.get(pipeline_class_name)
        if class_ is None:
            raise ValueError(
                f"Trying to load unknown pipeline class: {pipeline_class_name}"
            )

        return class_.from_pretrained(  # type: ignore[attr-defined]
            pretrained_model_name_or_path, *model_args, **kwargs
        )


class LiteSpecFormerPipeline(BaseForecastPipeline):
    """
    Forecasting pipeline for the LiteSpecFormer spectrum prediction model.

    This pipeline is designed for inference with a pretrained model:
    1) Load a pretrained checkpoint via `from_pretrained(...)`.
    2) Run forecasting through `predict(...)`.

    By default, the pipeline outputs quantile forecasts.
    """

    # The forecast output format used by this pipeline (quantile-based prediction).
    forecast_type: ForecastType = ForecastType.QUANTILES

    # Fallback context window length used when no explicit context length is provided.
    default_context_length: int = 768

    def __init__(self, model: LiteSpecFormerModel):
        """
        Initialize the LiteSpecFormer inference pipeline.

        Parameters
        ----------
        model : LiteSpecFormerModel
            A constructed LiteSpecFormer model instance, typically restored from
            pretrained weights before running inference.
        """
        # Initialize BaseForecastPipeline with the same model as the internal model.
        super().__init__(inner_model=model)

        # Keep a direct reference to the LiteSpecFormer model for pipeline methods.
        self.model = model

    @staticmethod
    def _get_prob_mass_per_quantile_level(
        quantile_levels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes normalized probability masses for quantile levels using trapezoidal rule approximation.

        Each quantile receives probability mass proportional to the width of its surrounding interval,
        creating a piecewise uniform distribution. The mass for quantile q_i is computed as
        (q_{i+1} - q_{i-1}) / 2, where q_0 = 0 and q_{n+1} = 1.

        Parameters
        ----------
        quantile_levels : torch.Tensor
            The quantile levels, must be strictly in (0, 1)

        Returns
        -------
        torch.Tensor
            The normalized probability mass per quantile
        """
        assert quantile_levels.ndim == 1
        assert quantile_levels.min() > 0.0 and quantile_levels.max() < 1.0

        device = quantile_levels.device
        boundaries = torch.cat(
            [
                torch.tensor([0.0], device=device),
                quantile_levels,
                torch.tensor([1.0], device=device),
            ]
        )
        prob_mass = (boundaries[2:] - boundaries[:-2]) / 2
        return prob_mass / prob_mass.sum()

    @property
    def model_context_length(self) -> int:
        """Get the context length of the model in the forecasting pipline."""
        return self.model.context_length

    @property
    def model_output_patch_size(self) -> int:
        """Get the output patch size of the model in the forecasting pipeline."""
        return self.model.forecasting_config.output_patch_size

    @property
    def model_prediction_length(self) -> int:
        """Get the prediction length of the model in the forecasting pipeline."""
        return (
            self.model.forecasting_config.max_output_patches
            * self.model.forecasting_config.output_patch_size
        )

    @property
    def quantiles(self) -> list[float]:
        """Get the the list of quantile levels for the model in the forecasting pipeline."""
        return self.model.forecasting_config.quantiles

    @property
    def max_output_patches(self) -> int:
        """Get the maximum number of output patches for the model in the forecasting pipeline."""
        return self.model.forecasting_config.max_output_patches

    def fit(
        self,
        inputs: (
            TensorOrArray
            | Sequence[TensorOrArray]
            | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]]
        ),
        prediction_length: int | None = None,
        validation_inputs: (
            TensorOrArray
            | Sequence[TensorOrArray]
            | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]]
            | None
        ) = None,
        finetune_mode: Literal["full", "lora"] = "lora",
        lora_config: "LoraConfig | dict | None" = None,
        context_length: int | None = None,
        learning_rate: float = 1e-6,
        num_steps: int = 2560,
        batch_size: int = 256,
        output_dir: Path | str | None = None,
        min_past: int | None = 64,
        callbacks: list["TrainerCallback"] | None = None,
        remove_printer_callback: bool = False,
        disable_data_parallel: bool = True,
        **extra_trainer_kwargs,
    ) -> "LiteSpecFormerPipeline":
        """
        Fine-tune a copy of the current `LiteSpecFormer` model on the given inputs and return a new pipeline.

        Parameters
        ----------
        inputs
            The time series on which the model will be fine-tuned. The allowed formats of inputs are the same as `LiteSpecFormerPipeline.predict()`.
            Note: when `inputs` is a list of dicts, the values inside `future_covariates` are not technically used for training the model;
            however, this key is used to infer which covariates are known into the future. Therefore, if your task contains known future covariates,
            make sure that this key exists in `inputs`. The values of individual future covariates may be set to `None` or an empty array.
        prediction_length
            The prediction horizon for which the model will be fine-tuned
        validation_inputs
            The time series used for validation and model selection. The format of `validation_inputs` is exactly the same as `inputs`, by default None which
            means that no validation is performed. Note that enabling validation may slow down fine-tuning for large datasets.
        finetune_mode
            One of "full" (performs full fine-tuning) or "lora" (performs Low Rank Adaptation (LoRA) fine-tuning), by default "full"
        lora_config
            The configuration to use for LoRA fine-tuning when finetune_mode="lora". Can be a `LoraConfig` object or a dict which is used to initialize `LoraConfig`.
            When unspecified and finetune_mode="lora", a default configuration is used
        context_length
            The maximum context length used during fine-tuning, by default set to the model's default context length
        learning_rate
            The learning rate for the optimizer, by default 1e-6
            When finetune_mode="lora", we recommend using a higher value of the learning rate, such as 1e-5
        num_steps
            The number of steps to fine-tune for, by default 2560
        batch_size
            The batch size used for fine-tuning. Note that the batch size here means the number of time series, including target(s) and covariates,
            which are input into the model. If your data has multiple target and/or covariates, the effective number of time series tasks in a batch
            will be lower than this value, by default 256
        output_dir
            The directory in which outputs from the `Trainer` will be saved, by default set to `litespecformer-finetuned/{%Y-%m-%d_%H-%M-%S}`
        min_past
            The minimum number of time steps the context must have during fine-tuning. All time series shorter than `min_past + prediction_length`
            are filtered out, by default set equal to prediction_length
        callbacks
            A list of `TrainerCallback`s which will be forwarded to the HuggingFace `Trainer`
        remove_printer_callback
            If True, all instances of `PrinterCallback` are removed from callbacks
        disable_data_parallel
            If True, ensures that DataParallel is disabled and training happens on a single GPU
        **extra_trainer_kwargs
            Extra kwargs are directly forwarded to `TrainingParameters`

        Returns
        -------
        A new `Chronos2Pipeline` with the fine-tuned modelchronos_config
        """

        import torch.cuda
        from transformers.trainer_callback import PrinterCallback
        from transformers.training_args import TrainingArguments

        if finetune_mode == "lora":
            if is_peft_available():
                from peft import LoraConfig, get_peft_model
            else:
                warnings.warn(
                    "`peft` is required for `finetune_mode='lora'`. Please install it with `pip install peft`. Falling back to `finetune_mode='full'`."
                )
                finetune_mode = "full"
                lora_config = None

        # Use the trainer from chronos2 to fine-tune the model
        from chronos.chronos2.trainer import (
            Chronos2Trainer,
            EvaluateAndSaveFinalStepCallback,
        )

        # Check that finetune_mode is valid, and set the default lora_config if necessary
        assert finetune_mode in [
            "full",
            "lora",
        ], f"finetune_mode must be one of ['full', 'lora'], got {finetune_mode}"

        # Check that lora_config is not specified when finetune_mode="full"
        if finetune_mode == "full" and lora_config is not None:
            raise ValueError(
                "lora_config should not be specified when `finetune_mode='full'`. To enable LoRA, set `finetune_mode='lora'`."
            )

        # Create a copy of the model to avoid modifying the original
        config: LiteSpecFormerConfig = deepcopy(self.model.config)
        model = LiteSpecFormerModel(config=config).to(self.model.device)  # type: ignore
        # Load the pre-trained model weights into the new model
        model.load_state_dict(self.model.state_dict())

        if finetune_mode == "lora":
            if lora_config is None:
                # If lora_config is not specified, use a default configuration
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=[
                        "self_attention.q",
                        "self_attention.v",
                        "self_attention.k",
                        "self_attention.o",
                        "output_patch_embedding.output_layer",
                    ],
                )
            elif isinstance(lora_config, dict):
                # If lora_config is a dict, use it to initialize a LoraConfig
                lora_config = LoraConfig(**lora_config)
            else:
                assert isinstance(
                    lora_config, LoraConfig
                ), f"lora_config must be an instance of LoraConfig or a dict, got {type(lora_config)}"

            # A pre-trained base model (such as LLaMA, BERT, GPT) is combined with your defined PEFT configuration
            # (such as LoRA, IA3, Prefix Tuning) to generate a PEFT model that "fine-tunes only some parameters".
            model = get_peft_model(model, lora_config)
            n_trainable_params, n_params = model.get_nb_trainable_parameters()
            logger.info(
                f"Using LoRA. Number of trainable parameters: {n_trainable_params}, total parameters: {n_params}."
            )

        if context_length is None:
            # Get the default context length from the model's config if not specified
            context_length = self.model_context_length

        if prediction_length is None:
            # Get the default prediction_length from the model's config if not specified
            prediction_length = self.model_prediction_length

        if min_past is None:
            # Get the default min_past from the model's config if not specified
            min_past = prediction_length

        # Create the training dataset for fine-tuning of LiteSpecFormer model.
        # Note that the dataset will filter out time series shorter than `min_past + prediction_length`.
        train_dataset = LiteSpecFormerDataset.convert_inputs(
            inputs=inputs,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=self.model_output_patch_size,
            min_past=min_past,
            mode=DatasetMode.TRAIN,
        )

        # Create the output directory for the fine-tuned model
        if output_dir is None:
            output_dir = Path("./results/litespecformer-finetuned") / time.strftime(
                "%Y-%m-%d_%H-%M-%S"
            )
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)

        assert isinstance(output_dir, Path)

        # check if the model is on CPU or GPU, and if GPU is available
        use_cpu = str(self.model.device) == "cpu"
        has_sm80 = (
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        )

        # warn user if a cuda device is available and CPU fine-tuning is used
        if use_cpu and torch.cuda.is_available():
            warnings.warn(
                "The model is being fine-tuned on the CPU, but a CUDA device is available. "
                "We recommend using the GPU for faster fine-tuning.",
                category=UserWarning,
                stacklevel=2,
            )

        # Create the training Parameters for the fine-tuning
        training_kwargs: dict = dict(
            output_dir=str(output_dir),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            warmup_ratio=0.0,
            optim="adamw_torch_fused",
            logging_strategy="steps",
            logging_steps=100,
            disable_tqdm=False,
            report_to="none",
            max_steps=num_steps,
            gradient_accumulation_steps=1,
            dataloader_num_workers=0,
            tf32=has_sm80 and not use_cpu,
            bf16=has_sm80 and not use_cpu,
            save_only_model=True,
            prediction_loss_only=True,
            save_total_limit=1,
            save_strategy="no",
            save_steps=None,
            eval_strategy="no",
            eval_steps=None,
            load_best_model_at_end=False,
            metric_for_best_model=None,
            use_cpu=use_cpu,
        )

        # Check if validation inputs are provided
        eval_dataset = None
        callbacks = callbacks or []
        if validation_inputs is not None:
            # construct validation dataset
            eval_dataset = LiteSpecFormerDataset.convert_inputs(
                inputs=validation_inputs,
                context_length=context_length,
                prediction_length=prediction_length,
                batch_size=batch_size,
                output_patch_size=self.model_output_patch_size,
                mode=DatasetMode.VALIDATION,
            )

            # set validation parameters
            training_kwargs["save_strategy"] = "steps"
            training_kwargs["save_steps"] = 100
            training_kwargs["eval_strategy"] = "steps"
            training_kwargs["eval_steps"] = 100
            training_kwargs["load_best_model_at_end"] = True
            training_kwargs["metric_for_best_model"] = "eval_loss"
            training_kwargs["label_names"] = ["future_target"]

            # add callback to ensure that the final model is evaluated
            callbacks.append(EvaluateAndSaveFinalStepCallback())

        # Update training Parameters with extra_trainer_kwargs for model evaluation
        training_kwargs.update(extra_trainer_kwargs)

        if training_kwargs["tf32"]:
            # setting tf32=True changes these global properties, we copy them here so that
            # we can restore them after fine-tuning
            matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
            cudnn_tf32 = torch.backends.cudnn.allow_tf32

        # Create the trainer for fine-tuning
        training_args = TrainingArguments(**training_kwargs)

        if disable_data_parallel and not use_cpu:
            # This is a hack to disable the default `transformers` behavior of using DataParallel
            training_args._n_gpu = 1
            assert training_args.n_gpu == 1  # Ensure that the hack worked

        # Create the trainer for fine-tuning from the chronos2
        trainer = Chronos2Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

        # Remove the PrinterCallback if it was added
        if remove_printer_callback:
            trainer.pop_callback(PrinterCallback)

        # ====== Start fine-tuning ======
        trainer.train()

        # update context_length and max_output_patches, if the model was fine-tuned with larger values
        model.forecasting_config.context_length = max(
            model.forecasting_config.context_length, context_length
        )
        model.forecasting_config.max_output_patches = max(
            model.forecasting_config.max_output_patches,
            math.ceil(prediction_length / self.model_output_patch_size),
        )
        # update forecasting_config in model's config, so it is saved correctly
        model.config.forecasting_config = model.forecasting_config.__dict__

        # Create a new pipeline with the fine-tuned model
        finetuned_pipeline = LiteSpecFormerPipeline(model=model)

        # Save fine-tuned model
        finetuned_path = output_dir
        finetuned_pipeline.save_pretrained(finetuned_path)
        logger.info(f"Finetuned model saved to {finetuned_path}")

        if training_kwargs["tf32"]:
            # restore tf32 settings
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32

        return finetuned_pipeline

    def _prepare_inputs_for_long_horizon_unrolling(
        self,
        context: torch.Tensor,
        # group_ids: torch.Tensor,
        # future_covariates: torch.Tensor,
        unrolled_quantiles: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare expanded context and path weights for long-horizon autoregressive unrolling.

        This helper expands each input context into multiple forecasting paths, where each
        path corresponds to one quantile level in `unrolled_quantiles`. It also computes
        the probability mass weight for each (unrolled path quantile, model output quantile)
        pair, which is later used to merge all path predictions back into the target
        quantile set.

        Parameters
        ----------
        context : torch.Tensor
            Historical input sequence with shape (batch, time_length).
        unrolled_quantiles : torch.Tensor
            Quantile levels used to generate autoregressive paths during unrolling.
            Shape: (n_paths,).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - expanded context with shape (batch, n_paths, time_length)
            - unrolled sample weights with shape (n_paths, n_model_quantiles)

        Notes
        -----
        - `unrolled_sample_weights` is built as an outer product of:
        1) probability mass of each unrolled path quantile, and
        2) probability mass of each model output quantile (`self.quantiles`).
        - Compared with median-only unrolling, this multi-quantile strategy preserves
        more uncertainty and reduces uncertainty collapse in long-range rollout.
        """
        context = repeat(context, "b t -> b q t", q=len(unrolled_quantiles))

        # We unroll the quantiles in unrolled_quantiles to the future and each unrolled quantile gives
        # len(self.quantiles) predictions, so we end up with len(unrolled_quantiles) * len(self.quantiles)
        # "samples". unrolled_sample_weights specifies the amount of probability mass covered by each sample.
        # Note that this effectively leads to shrinking of the probability space but it is better heuristic
        # than just using the median to unroll, which leads to uncertainty collapse.
        unrolled_sample_weights = torch.outer(
            self._get_prob_mass_per_quantile_level(unrolled_quantiles),
            self._get_prob_mass_per_quantile_level(torch.tensor(self.quantiles)),
        )

        # return context, group_ids, future_covariates, unrolled_sample_weights
        return context, unrolled_sample_weights

    def _autoregressive_unroll_for_long_horizon(
        self,
        context: torch.Tensor,
        prediction: torch.Tensor,
        unrolled_quantiles: torch.Tensor,
        unrolled_sample_weights: torch.Tensor,
        num_output_patches: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute one autoregressive rollout step for long-horizon forecasting.

        Workflow
        --------
        1) Interpolate current prediction from model quantiles (`self.quantiles`) to
        unrolled path quantiles (`unrolled_quantiles`).
        2) Append interpolated values to context and keep only the latest
        `self.model_context_length` time steps.
        3) Run `_predict_step` on all paths.
        4) Merge `(n_paths * n_quantiles)` samples back to `n_quantiles` using
        weighted quantile aggregation with `unrolled_sample_weights`.

        Parameters
        ----------
        context : torch.Tensor
            Expanded context for each unrolled path, shape (batch, n_paths, context_length).
        prediction : torch.Tensor
            Current forecast at model quantiles, shape (batch, n_quantiles, horizon_length).
        unrolled_quantiles : torch.Tensor
            Quantile levels used for autoregressive path expansion, shape (n_paths,).
        unrolled_sample_weights : torch.Tensor
            Weight matrix for path/sample aggregation, shape (n_paths, n_quantiles).
        num_output_patches : int
            Number of output patches generated by each `_predict_step` call.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - updated prediction at `self.quantiles`, shape (batch, n_quantiles, horizon_length)
            - updated context after appending rollout values,
            shape (batch, n_paths, self.model_context_length)

        Notes
        -----
        - This function performs a path-wise stochastic-style rollout while preserving
        quantile uncertainty through weighted aggregation.
        - It is intended for long-horizon prediction where one-step direct forecasting
        is iteratively unrolled.
        """
        # Get unrolled_quantiles from prediction and append it to the expanded context
        prediction_unrolled = interpolate_quantiles(
            query_quantile_levels=unrolled_quantiles,
            original_quantile_levels=self.quantiles,
            original_values=rearrange(prediction, "b q h -> b h q"),
        )

        prediction_unrolled = rearrange(prediction_unrolled, "b h q -> b q h")
        context = torch.cat([context, prediction_unrolled], dim=-1)[
            ..., -self.model_context_length :
        ]
        n_paths = len(unrolled_quantiles)

        # Note that the function for single-step prediction still needs to be called here.
        prediction: torch.Tensor = self._predict_step(
            context=rearrange(context, "b n t -> (b n) t"),
            num_output_patches=num_output_patches,
        )

        # Reshape predictions from (batch * n_paths, n_quantiles, length) to (batch, n_paths * n_quantiles, length)
        prediction = rearrange(prediction, "(b n) q h -> b (n q) h", n=n_paths)

        # Reduce `n_paths * n_quantiles` to n_quantiles and transpose back
        prediction = weighted_quantile(
            query_quantile_levels=self.quantiles,
            sample_weights=rearrange(unrolled_sample_weights, "n q -> (n q)"),
            samples=rearrange(prediction, "b (n q) h -> b h (n q)", n=n_paths),
        )
        prediction = rearrange(prediction, "b h q -> b q h")

        return prediction, context

    @torch.no_grad()
    def predict(
        self,
        inputs: (
            TensorOrArray
            | Sequence[TensorOrArray]
            | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray]]]
        ),
        prediction_length: int | None = None,
        batch_size: int = 256,
        context_length: int | None = None,
        limit_prediction_length: bool = False,
        **kwargs,
    ) -> list[torch.Tensor]:
        """
        Generate forecasts for the given time series.

        Parameters
        ----------
        inputs
            The input time series for which forecasts will be generated. The allowed formats of inputs are:
            - A universe time series with shape of (batch_size, series_length).
            - A list of time series with shape of (series_length). Each time series in the list can have different length.
            - A Huggingface Dataset object with a column named "target" containing the time series. Each time series can have different length.

            The input time series must be the 2-dimensional tensor of shape (batch_size, series_length).
            If the input is a multivariate time series with shape of (batch_size, n_variates, series_length), must be reshaped to (batch_size * n_variates, series_length).

            Examples:
            ```python

            # Batch of time series with channel-independence
            inputs = torch.randn(32, 100)

            # List of time series with different lengths and n_variates
            inputs = [
                torch.randn(100),  # univariate series of length 100
                torch.randn(150),  # univariate series of length 150
                torch.randn(120),  # univariate series of length 120
            ]

            # Huggingface Dataset with a "target" column containing time series of different lengths and n_variates
            from datasets import Dataset
            inputs = Dataset.from_list([{"target": torch.randn(100)},
                                        {"target": torch.randn(150)},
                                        {"target": torch.randn(120)}])
            ```
        prediction_length
            The number of time steps to predict for, defaults to the model's default prediction length
        batch_size
            The batch size used for prediction. Note that the batch size here means the number of time series, including target(s) and covariates,
            which are input into the model. If your data has multiple target and/or covariates, the effective number of time series tasks in a batch
            will be lower than this value, by default 256
        context_length
            The maximum context length used during for inference, by default set to the model's default context length
        limit_prediction_length
            If True, an error is raised when prediction_length is greater than model's default prediction length, by default False

        Returns
        -------
        The model's predictions, a list of `torch.Tensor` where each element has shape (n_variates, n_quantiles, prediction_length) and the number of
        elements are equal to the number of target time series (univariate or multivariate) in the `inputs`.

        """
        # Obtain the maximum prediction length of a single autoregression iteration of the model.
        model_prediction_length = self.model_prediction_length
        if prediction_length is None:
            prediction_length = model_prediction_length

        # The maximum number of output patches to generate in a single forward pass before the long-horizon heuristic kicks in. Note: A value larger
        # than the model's default max_output_patches may lead to degradation in forecast accuracy, defaults to a model-specific value
        max_output_patches = kwargs.pop("max_output_patches", self.max_output_patches)
        # The set of quantiles to use when making long-horizon predictions; must be a subset of the model's default quantiles. These quantiles
        # are appended to the historical context and input into the model autoregressively to generate long-horizon predictions. Note that the
        # effective batch size increases by a factor of `len(unrolled_quantiles)` when making long-horizon predictions,

        # by default [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        unrolled_quantiles = kwargs.pop(
            "unrolled_quantiles",
            [
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
            ],
        )

        # A callback which is called after each batch has been processed
        after_batch_callback: Callable = kwargs.pop("after_batch", lambda: None)

        if len(kwargs) > 0:
            raise TypeError(f"Unexpected keyword Augments: {list(kwargs.keys())}.")

        if not set(unrolled_quantiles).issubset(self.quantiles):
            raise ValueError(
                f"Unrolled quantiles must be a subset of the model's quantiles. "
                f"Found: {unrolled_quantiles=}, model_quantiles={self.quantiles}"
            )
        unrolled_quantiles_tensor = torch.tensor(unrolled_quantiles)

        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        # Set the context length for model prediction
        if context_length is None:
            context_length = self.model_context_length

        # If the specified context length is greater than the model's default context length,
        # it may lead to suboptimal forecasts since the model is not optimized for such long contexts.
        # Then a warning will be issued and the context length will be reset to the model's default value.
        if context_length > self.model_context_length:
            warnings.warn(
                f"The specified context_length {context_length} is greater than the model's default context length {self.model_context_length}. "
                f"Resetting context_length to {self.model_context_length}."
            )
            context_length = self.model_context_length

        # Convert the input to a LiteSpecFormerDataset and create a DataLoader.
        test_dataset = LiteSpecFormerDataset.convert_inputs(
            inputs=inputs,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=self.model_output_patch_size,
            mode=DatasetMode.TEST,
        )

        # Build a DataLoader for testing
        test_loader = DataLoader(
            test_dataset,
            batch_size=None,
            # pin_memory=self.model.device.type == "cuda",
            shuffle=False,
            drop_last=False,
        )

        all_predictions: list[torch.Tensor] = []

        for batch in test_loader:
            # assert batch["future_target"] is None
            batch_context = batch["context"]

            # Predict a batch of data
            batch_prediction = self._predict_batch(
                context=batch_context,
                unrolled_quantiles_tensor=unrolled_quantiles_tensor,
                prediction_length=prediction_length,
                max_output_patches=max_output_patches,
            )

            all_predictions.extend(batch_prediction)
            after_batch_callback()

        return all_predictions

    def _predict_batch(
        self,
        context: torch.Tensor,
        unrolled_quantiles_tensor: torch.Tensor,
        prediction_length: int,
        max_output_patches: int,
    ) -> list[torch.Tensor]:
        """
        Predict a batch of time series and return the predictions in the original order of the input time series.

        Parameters
        ----------
        context
            A tensor of shape (batch, context_length) containing the historical context for each time series in the batch.
            Note that the context may contain both target and covariate time series, which are left-padded if necessary.
        unrolled_quantiles_tensor
            A tensor containing the quantiles to use for long-horizon prediction unrolling, of shape (n_unrolled_quantiles,)
        prediction_length
            The total length to predict for each time series in the batch.
        max_output_patches
            The maximum number of output patches to generate in a single forward pass before the long-horizon heuristic kicks in.

        Returns
        -------
        predictions
            A list of tensors containing the predictions for each time series in the batch.
        """
        context = context.to(device=self.model.device, dtype=torch.float32)

        # Store the predicted results and the length to be predicted.
        predictions = []
        remaining = prediction_length

        # predict first set of patches up to max_output_patches
        prediction: torch.Tensor = self._predict_step(
            context=context,
            num_output_patches=get_num_output_patches(
                remaining_horizon=remaining,
                model_output_patch_size=self.model_output_patch_size,
                max_output_patches=max_output_patches,
            ),
        )

        # Record the prediction and update the remaining horizon
        predictions.append(prediction)
        # Calculate the remaining length that needs to be predicted.
        remaining -= prediction.shape[-1]

        # prepare inputs for long horizon prediction
        if remaining > 0:
            (
                context,
                unrolled_sample_weights,
            ) = self._prepare_inputs_for_long_horizon_unrolling(
                context=context,
                unrolled_quantiles=unrolled_quantiles_tensor,
            )

        # long horizon heuristic
        while remaining > 0:
            prediction, context = self._autoregressive_unroll_for_long_horizon(
                context=context,
                prediction=prediction,
                unrolled_quantiles=unrolled_quantiles_tensor,
                unrolled_sample_weights=unrolled_sample_weights,
                num_output_patches=get_num_output_patches(
                    remaining_horizon=remaining,
                    model_output_patch_size=self.model_output_patch_size,
                    max_output_patches=max_output_patches,
                ),
            )
            predictions.append(prediction)
            remaining -= prediction.shape[-1]

        batch_prediction = torch.cat(predictions, dim=-1)[..., :prediction_length].to(
            dtype=torch.float32, device="cpu"
        )

        return batch_prediction

    def _predict_step(
        self,
        context: torch.Tensor,
        num_output_patches: int,
    ) -> torch.Tensor:
        """
        Run a single forward prediction step (inference-only) on the underlying LiteSpecFormer model.

        Given the input context time series, this method predicts the next `num_output_patches`
        output patches and returns the model's quantile forecasts. Gradients are disabled via
        `torch.no_grad()` because this is used for inference/rollout inside the pipeline.

        Parameters
        ----------
        context : torch.Tensor
            Input context sequence fed to the model. Shape is typically (batch, context_length).
        num_output_patches : int
            Number of output patches (chunks) to forecast in this step.

        Returns
        -------
        torch.Tensor
            Quantile predictions produced by the model, moved to the same dtype/device as `context`.
            Expected shape: (batch, n_quantiles, horizon_length) where `horizon_length` corresponds
            to the total length covered by `num_output_patches`.
        """
        with torch.no_grad():
            # Forward pass through the model and extract quantile forecasts.
            prediction: torch.Tensor = self.model(
                context=context,
                num_output_patches=num_output_patches,
            ).quantile_preds.to(context)

        return prediction

    def predict_quantiles(  # type: ignore[override]
        self,
        inputs: (
            TensorOrArray
            | Sequence[TensorOrArray]
            | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray]]]
        ),
        prediction_length: int | None = None,
        quantile_levels: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **predict_kwargs,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Refer to ``LiteSpecFormerPipeline.predict`` for shared parameters.
        The code is adapted from the Chronos2Pipeline.predict_quantiles method in the Chronos library
        in https://github.com/amazon-science/chronos-forecasting
        with modifications to fit the LiteSpecFormer model and use case.

        Parameters
        ----------
        quantile_levels
            Quantile levels to compute, by default [0.1, 0.2, ..., 0.9]

        Returns
        -------
        quantiles
            A list of torch tensors containing quantile forecasts. Each element of the list has shape (n_variates, prediction_length, len(quantile_levels))
            and the number of elements are equal to the number of target time series (univariate or multivariate) in the `inputs`.
        meanPipelineRegistry
            A list of torch tensors containing containing mean (point) forecasts. Each element of the list has shape (n_variates, prediction_length)
            and the number of elements are equal to the number of target time series (univariate or multivariate) in the `inputs`.
        """
        training_quantile_levels = self.quantiles

        # Invoke predict to get quantile predictions
        # with shape (n_variates, len(training_quantile_levels), prediction_length)
        # for each time series in the batch
        predictions: list[torch.Tensor] = self.predict(
            inputs, prediction_length=prediction_length, **predict_kwargs
        )

        # Swap quantile and time axes for each prediction
        predictions = [rearrange(pred, "... q h -> ... h q") for pred in predictions]

        if set(quantile_levels).issubset(training_quantile_levels):
            # no need to perform intra/extrapolation
            quantile_indices = [
                training_quantile_levels.index(q) for q in quantile_levels
            ]
            quantiles = [pred[..., quantile_indices] for pred in predictions]
        else:
            # we interpolate quantiles if quantiles that LiteSpecFormer was trained on were not provided
            if min(quantile_levels) < min(training_quantile_levels) or max(
                quantile_levels
            ) > max(training_quantile_levels):
                logger.warning(
                    f"\tQuantiles to be predicted ({quantile_levels}) are not within the range of "
                    f"quantiles that LiteSpecFormer was trained on ({training_quantile_levels}). "
                    "Quantile predictions will be set to the minimum/maximum levels at which LiteSpecFormer "
                    "was trained on. This may significantly affect the quality of the predictions."
                )

            quantiles = [
                interpolate_quantiles(quantile_levels, training_quantile_levels, pred)
                for pred in predictions
            ]

        # NOTE: the median is returned as the mean here
        mean = [pred[..., training_quantile_levels.index(0.5)] for pred in predictions]

        return quantiles, mean

    @torch.no_grad()
    def embed(
        self,
        inputs: TensorOrArray | Sequence[TensorOrArray],
        batch_size: int = 256,
        context_length: int | None = None,
    ) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get encoder embeddings for the given time series.

        Parameters
        ----------
        inputs
            The input time series for which embeddings will be generated. The allowed formats of inputs are:
        batch_size
            The batch size used for generating embeddings. Note that the batch size here means the total number of time series which are input into the model.
            If your data has multiple variates, the effective number of time series tasks in a batch will be lower than this value, by default 256
        context_length
            The maximum context length used during for inference, by default set to the model's default context length

        Returns
        -------
        embeddings
            a list of `torch.Tensor` where each element has shape (n_variates, num_patches + 2, d_model) and the number of elements are equal to the number
            of target time series (univariate or multivariate) in the `inputs`. The extra +2 is due to embeddings of the [REG] token and a masked output patch token.
        loc_scale
            a list of tuples with the mean and standard deviation of each time series.
        """
        if context_length is None:
            context_length = self.model_context_length

        if context_length > self.model_context_length:
            warnings.warn(
                f"The specified context_length {context_length} is greater than the model's default context length {self.model_context_length}. "
                f"Resetting context_length to {self.model_context_length}."
            )
            context_length = self.model_context_length

        test_dataset = LiteSpecFormerDataset.convert_inputs(
            inputs=inputs,
            context_length=context_length,
            prediction_length=0,
            batch_size=batch_size,
            output_patch_size=self.model_output_patch_size,
            mode=DatasetMode.TEST,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=self.model.device.type == "cuda",
            shuffle=False,
            drop_last=False,
        )

        all_embeds: list[torch.Tensor] = []
        all_loc_scales: list[tuple[torch.Tensor, torch.Tensor]] = []

        for batch in test_loader:
            # assert batch["future_target"] is None
            batch_context = batch["context"].squeeze(0)

            encoder_outputs, _, (locs, scales), _ = self.model.encode(
                context=batch_context.to(device=self.model.device, dtype=torch.float32)
            )
            batch_embeds = encoder_outputs.last_hidden_state.cpu()
            batch_loc_scales = (locs.cpu(), scales.cpu())

            all_embeds.append(batch_embeds)
            all_loc_scales.append(batch_loc_scales)

        return all_embeds, all_loc_scales

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load the model, either from a local path, S3 prefix or from the HuggingFace Hub.
        Supports the same parameters as ``AutoConfig`` and ``AutoModel`` from ``transformers``.
        """

        # Check if the hub model_id or local path is a LoRA adapter
        if find_adapter_config_file(pretrained_model_name_or_path) is not None:
            if not is_peft_available():
                raise ImportError(
                    f"The model at {pretrained_model_name_or_path} is a `peft` adaptor, but `peft` is not available. "
                    f"Please install `peft` with `pip install peft` to use this model. "
                )
            from peft import AutoPeftModel

            model = AutoPeftModel.from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs
            )
            model = model.merge_and_unload()
            return cls(model=model)

        # Handle the case for the base model
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

        assert hasattr(
            config, "litespecformer_pipeline_class"
        ), "Not a LiteSpecFormer config file"
        pipeline_class_name = getattr(
            config, "litespecformer_pipeline_class", "LiteSpecFormerPipeline"
        )

        model = LiteSpecFormerModel.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        return cls(model=model)

    def save_pretrained(self, save_directory: str | Path, *args, **kwargs):
        """
        Save the underlying model to a local directory or to HuggingFace Hub.
        """
        self.model.save_pretrained(save_directory, *args, **kwargs)

    def calculate_metrics(
        self,
        contexts: Union[np.ndarray, torch.FloatTensor],
        targets: Union[np.ndarray, torch.FloatTensor],
        predictions: Union[np.ndarray, torch.FloatTensor],
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics given the contexts, targets and predictions. This is a placeholder method and should be implemented based on the specific metrics you want to compute.

        Parameters
        ----------
        contexts: Union[np.ndarray, torch.FloatTensor]
            The historical context time series data used for making predictions, with shape (batch_size, num_variables, context_length).
        targets: Union[np.ndarray, torch.FloatTensor]
            The true values for the test set, with shape (batch_size, num_variables, prediction_length).
        predictions: Union[np.ndarray, torch.FloatTensor]
            The predicted values of LiteSpecFormer, expected to have shape (batch_size, num_variables, num_quantiles, prediction_length).

        Returns
        -------
        Dict[str, float]
            A dictionary containing the computed evaluation metrics, including:
            - "MSE": Mean Squared Error between the median predictions and the targets.
            - "MAE": Mean Absolute Error between the median predictions and the targets.
            - "RMSE": Root Mean Squared Error between the median predictions and the targets.
            - "MAPE": Mean Absolute Percentage Error between the median predictions and the targets.
            - "MASE": Mean Absolute Scaled Error, which is the mean absolute error of the predictions scaled by the mean absolute error of a naive forecast (e.g., using the last observed value as the forecast).
            - "MSPE": Mean Scaled Pinball Error, which is a common metric for evaluating quantile forecasts, calculated as the pinball loss of the predicted quantiles normalized by the pinball loss of a naive forecast (e.g., using the last observed value as the forecast).
            - "RSE": Relative Squared Error, calculated as the sum of squared errors of the predictions relative to the sum of squared errors of a naive forecast.
        """
        # Convert inputs to torch tensors if they are numpy arrays
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float().to(self.model.device)
        if isinstance(predictions, np.ndarray):
            predictions = torch.from_numpy(predictions).float().to(self.model.device)

        # Check the shape of inputs data
        assert (
            targets.ndim == 3
        ), f"Expected targets to have 3 dimensions (batch_size, num_variables, prediction_length), but got {targets.shape}"
        # Compute metrics
        return calculate_metrics(
            contexts=contexts, predictions=predictions, targets=targets
        )

    def evaluate_spectrum_library(
        self,
        dataset_name_or_path: str,
        seq_length: int,
        prediction_length: int,
        batch_size: int = 256,
        split_ratio: float = 0.6,
        accelerator: Accelerator | None = None,
        output_dir: str | None = None,
    ) -> Dict[str, float]:
        """
        This method is used to test the zero-shot confidence prediction capability of the spectral prediction model in the current pipeline.

        The main testing methods and procedures are derived from: https://github.com/FlowVortex/Spectrum-Prediction-Library

        Parameters
        ----------
        dataset_name_or_path: str
            Path to the spectrum library dataset.
        seq_length: int
            Length of the input sequence.
        prediction_length: int
            Length of the prediction sequence.
        batch_size: int, optional
            Batch size for evaluation (default is 256).
        split_ratio: float, optional
            Ratio for splitting the dataset (default is 0.6).
        accelerator: Accelerator | None, optional
            Accelerator for distributed evaluation (default is None).
        output_dir: str | None, optional
            Directory to save evaluation results (default is None).

        Returns
        -------
        Dict[str, float]
            A dictionary containing the computed evaluation metrics.
        """
        time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dataset_name = dataset_name_or_path.split("/")[-1]
        setting = f"{time_now}_{dataset_name}_seq{seq_length}_pred{prediction_length}_batch{batch_size}_split{split_ratio}"

        if torch.cuda.is_available():
            self.model.to(torch.device("cuda"))

        if accelerator is None:
            # Create an Accelerator for distributed evaluation if one is not provided.
            # This allows the evaluation to be performed on multiple GPUs if available, which can speed up the process significantly for large datasets.
            accelerator = Accelerator()
        accelerator.print(
            f"[{time_now}] Starting evaluation on spectrum library with:",
            dataset_name_or_path,
        )
        accelerator.print(
            f"\tseq_length={seq_length}, prediction_length={prediction_length}, batch_size={batch_size}, split_ratio={split_ratio}"
        )

        accelerator.print("Loading model onto device:", accelerator.device)
        accelerator.print(
            Fore.RED + "Loading the Spectrum Library dataset" + Style.RESET_ALL,
            end=" -> ",
        )
        # load the dataset for evaluation
        dataset = SpectrumLibraryDataset(
            seq_length=seq_length,
            prediction_length=prediction_length,
            dataset_name_or_path=dataset_name_or_path,
            split_ratio=split_ratio,
            flag="test",
        )
        # create the dataloader for evaluation
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        accelerator.print(Fore.GREEN + "Done" + Style.RESET_ALL)

        # Prepare the model and dataloader for distributed evaluation with accelerator
        accelerator.print(
            "Preparing the model and dataloader for distributed evaluation..."
        )
        self.model, data_loader = accelerator.prepare(self.model, data_loader)

        # Create lists to store contexts, targets and predictions for metric computation after the evaluation loop
        context_list, target_list, prediction_list = [], [], []

        accelerator.print(Fore.BLUE + "Starting evaluation loop" + Style.RESET_ALL)
        # Create a tqdm progress bar for the dataloader
        data_loader = tqdm(data_loader, desc="Evaluating")
        with torch.no_grad():
            # iterate through the dataloader and evaluate the model
            for batch_x, batch_y in data_loader:
                # Get the shape of the batch data
                batch_size, seq_length, num_variates = batch_x.shape

                # [batch_size, num_variates, seq_length]
                context, target = batch_x.permute(0, 2, 1), batch_y.permute(0, 2, 1)
                # [batch_size * num_variates, seq_length] -> channel independent processing
                context = context.reshape(-1, context.shape[-1])
                target = target.reshape(-1, target.shape[-1])

                # Predict the future values using the model's predict method
                prediciton = self.predict(
                    inputs=context,
                    prediction_length=prediction_length,
                    batch_size=batch_size,
                )

                # Reshape the context and target tensor to [batch_size, num_variates, prediction_length]
                context = context.reshape(batch_size, num_variates, seq_length)
                target = target.reshape(batch_size, num_variates, prediction_length)

                # Concatenate the each prediction in the batch
                prediction = torch.concatenate(
                    [pred.unsqueeze(0) for pred in prediciton], dim=0
                )
                # Reshape it to [batch_size, num_variates, num_quantiles, prediction_length]
                prediction = prediction.reshape(
                    batch_size, num_variates, -1, prediction_length
                )

                # Record the experimental results of this evaluation.
                context_list.append(context.cpu())
                target_list.append(target.cpu())
                prediction_list.append(prediction.cpu())

        accelerator.print(Fore.BLUE + "Evaluation loop completed" + Style.RESET_ALL)

        # Compute evaluation metrics using the collected contexts, targets and predictions
        accelerator.print("Computing evaluation metrics...")
        contexts, targets, predictions = (
            torch.concatenate(context_list, dim=0).to(self.model.device),
            torch.concatenate(target_list, dim=0).to(self.model.device),
            torch.concatenate(prediction_list, dim=0).to(self.model.device),
        )
        computed_metrics = self.calculate_metrics(
            contexts=contexts, targets=targets, predictions=predictions
        )
        for key in computed_metrics.keys():
            computed_metrics[key] = float(computed_metrics[key])
            accelerator.print(f"\t{key}: {computed_metrics[key]}")

        mse, mae, rmse, mape, mase, mspe, rse = (
            computed_metrics.get("mse"),
            computed_metrics.get("mae"),
            computed_metrics.get("rmse"),
            computed_metrics.get("mape"),
            computed_metrics.get("mase"),
            computed_metrics.get("mspe"),
            computed_metrics.get("rse"),
        )

        # Create output directory if it doesn't exist and save the computed metrics
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = "./results/zero_shot_evaluation"
            os.makedirs(output_dir, exist_ok=True)

        save_path = path.join(output_dir, setting + ".pth")
        accelerator.print(f"Saving outputs results to {save_path}", end=" -> ")
        if accelerator.is_main_process:
            torch.save(
                obj={
                    "context": contexts,
                    "target": targets,
                    "prediction": predictions,
                    "mse": mse,
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                    "mase": mase,
                    "mspe": mspe,
                    "rse": rse,
                },
                f=save_path,
            )
        accelerator.print(Fore.GREEN + "Done" + Style.RESET_ALL)

        # Logging the results to csv file locally
        accelerator.print("Logging results to csv file", end=" -> ")
        csv_save_path = path.join(output_dir, "results.csv")
        logging_messages = {
            "time": time_now,
            "model": "LiteSpecFormer",
            "dataset": dataset_name_or_path,
            "seq_length": seq_length,
            "prediction_length": prediction_length,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "mase": mase,
            "mspe": mspe,
            "rse": rse,
        }
        logging_results(
            accelerator=accelerator,
            logging_path=csv_save_path,
            headers=list(logging_messages.keys()),
            messages=logging_messages,
        )
        accelerator.print(Fore.GREEN + "Done" + Style.RESET_ALL)

        # Clear the model and dataloader from accelerator to free up memory before computing metrics
        accelerator.print(
            "Clearing model and dataloader from accelerator to free up memory",
            end=" -> ",
        )
        accelerator.clear(self.model, data_loader)
        accelerator.print(Fore.GREEN + "Done" + Style.RESET_ALL)

        return computed_metrics
