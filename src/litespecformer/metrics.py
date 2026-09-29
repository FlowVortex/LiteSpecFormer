from typing import Dict, Sequence, Union

import numpy as np
import torch

# Quantile levels used by CRPS. The model also predicts 0.05 and 0.95;
# those two tail levels are left out of the score.
DEFAULT_QUANTILE_LEVELS = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)
_CRPS_EXCLUDED_QUANTILE_LEVELS = (0.05, 0.95)


def mean_squared_error(
    predictions: torch.FloatTensor, targets: torch.FloatTensor
) -> Union[torch.FloatTensor, float]:
    """Calculates the Mean Squared Error (MSE) between predictions and targets."""
    return torch.mean((targets - predictions) ** 2)


def mean_absolute_error(
    predictions: torch.FloatTensor, targets: torch.FloatTensor
) -> Union[torch.FloatTensor, float]:
    """Calculates the Mean Absolute Error (MAE) between predictions and targets."""
    return torch.mean(torch.abs(targets - predictions))


def root_mean_squared_error(
    predictions: torch.FloatTensor, targets: torch.FloatTensor
) -> Union[torch.FloatTensor, float]:
    """Calculates the Root Mean Squared Error (RMSE) between predictions and targets."""
    return torch.sqrt(mean_squared_error(predictions, targets))


def mean_absolute_percentage_error(
    predictions: torch.FloatTensor, targets: torch.FloatTensor
) -> Union[torch.FloatTensor, float]:
    """Calculates the Mean Absolute Percentage Error (MAPE) between predictions and targets."""

    return torch.mean(torch.abs((targets - predictions) / targets))


def mean_squared_percentage_error(
    predictions: torch.FloatTensor, targets: torch.FloatTensor
) -> Union[torch.FloatTensor, float]:
    """Calculates the Mean Squared Percentage Error (MSPE) between predictions and targets."""
    return torch.mean((targets - predictions) ** 2 / targets**2)


def root_squared_error(
    predictions: torch.FloatTensor, targets: torch.FloatTensor
) -> Union[torch.FloatTensor, float]:
    """Calculates the Root Squared Error (RSE) between predictions and targets."""
    return torch.sqrt(torch.sum((targets - predictions) ** 2)) / torch.sqrt(
        torch.sum((targets - targets.mean()) ** 2)
    )


def quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantile_levels: Union[float, Sequence[float], torch.Tensor],
) -> torch.Tensor:
    r"""
    Element-wise quantile (pinball) loss, scaled by 2.

    This is the integrand of the continuous ranked probability score:

    .. math::
        z_q(\hat{Y}, Y) = 2 \left|(\hat{Y} - Y)\,(\mathbf{1}\{Y \le \hat{Y}\} - q)\right|

    ``quantile_levels`` may be a scalar or a vector. A vector is broadcast over
    the quantile axis of ``predictions`` (the second-to-last dimension).
    CRPS integrates this term over quantile levels; see
    :func:`continuous_ranked_probability_score`.
    """
    levels = torch.as_tensor(
        quantile_levels, dtype=predictions.dtype, device=predictions.device
    )
    if levels.ndim == 0:
        return 2 * torch.abs(
            (predictions - targets)
            * ((targets <= predictions).to(dtype=predictions.dtype) - levels)
        )

    if levels.ndim != 1:
        raise ValueError(
            f"quantile_levels must be a scalar or a 1D vector, got shape {tuple(levels.shape)}"
        )
    if predictions.shape[-2] != levels.numel():
        raise ValueError(
            "The quantile axis of predictions must match quantile_levels. "
            f"Got predictions shape {tuple(predictions.shape)} and "
            f"{levels.numel()} quantile levels."
        )

    # [..., num_quantiles, prediction_length]
    targets = targets.unsqueeze(-2)
    levels = levels.view(*([1] * (predictions.ndim - 2)), -1, 1)
    return 2 * torch.abs(
        (predictions - targets)
        * ((targets <= predictions).to(dtype=predictions.dtype) - levels)
    )


def _crps_trapezoid_weights(quantile_levels: torch.Tensor) -> torch.Tensor:
    """
    Normalized trapezoidal weights for integrating the quantile loss into CRPS.

    For sorted levels ``q_i``, the unnormalized bin width is
    ``(q_{i+1} - q_{i-1}) / 2``, with one-sided differences at the endpoints.
    Dividing by the total width estimates the integral over ``(0, 1)`` from the
    levels the model actually predicts, including grids that are tighter in the tails.
    """
    num_quantiles = quantile_levels.shape[0]
    if num_quantiles < 2:
        return torch.ones(
            num_quantiles,
            dtype=quantile_levels.dtype,
            device=quantile_levels.device,
        )

    weights = torch.empty(
        num_quantiles,
        dtype=quantile_levels.dtype,
        device=quantile_levels.device,
    )
    weights[0] = 0.5 * (quantile_levels[1] - quantile_levels[0])
    if num_quantiles > 2:
        weights[1:-1] = 0.5 * (quantile_levels[2:] - quantile_levels[:-2])
    weights[-1] = 0.5 * (quantile_levels[-1] - quantile_levels[-2])
    return weights / weights.sum()


def _as_quantile_level_list(
    quantile_levels: Union[Sequence[float], torch.Tensor],
) -> list[float]:
    if isinstance(quantile_levels, torch.Tensor):
        return [float(level) for level in quantile_levels.detach().cpu().reshape(-1)]
    return [float(level) for level in quantile_levels]


def _is_excluded_crps_level(level: float) -> bool:
    return any(
        abs(level - excluded) <= 1e-6 for excluded in _CRPS_EXCLUDED_QUANTILE_LEVELS
    )


def _select_crps_quantiles(
    predictions: torch.Tensor,
    quantile_levels: Union[Sequence[float], torch.Tensor],
) -> tuple[torch.Tensor, list[float]]:
    """Drop the 0.05 and 0.95 forecasts before the CRPS integral."""
    levels = _as_quantile_level_list(quantile_levels)
    if len(levels) != predictions.shape[-2]:
        raise ValueError(
            "quantile_levels must match the quantile axis of predictions. "
            f"Got {len(levels)} levels and predictions shape {tuple(predictions.shape)}."
        )
    kept_indices = [
        index
        for index, level in enumerate(levels)
        if not _is_excluded_crps_level(level)
    ]
    if not kept_indices:
        raise ValueError(
            "CRPS needs at least one quantile level other than 0.05 and 0.95."
        )
    kept_levels = [levels[index] for index in kept_indices]
    if len(kept_indices) == len(levels):
        return predictions, kept_levels
    index = torch.tensor(kept_indices, device=predictions.device, dtype=torch.long)
    return predictions.index_select(dim=-2, index=index), kept_levels


def continuous_ranked_probability_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantile_levels: Union[Sequence[float], torch.Tensor] = DEFAULT_QUANTILE_LEVELS,
) -> torch.Tensor:
    r"""
    Mean continuous ranked probability score (CRPS) of a quantile forecast.

    CRPS is the integral of the quantile loss over quantile levels:

    .. math::
        \mathrm{CRPS}(F, y) = \int_0^1 z_q(F^{-1}(q), y)\,dq

    With a discrete quantile forecast this integral is approximated by a
    normalized trapezoidal rule on the provided levels. The result is averaged
    over every batch item, variable, and forecast step. A degenerate forecast
    that places every quantile on the same value scores CRPS equal to the
    absolute error, which is the closed-form CRPS of a point mass.

    Parameters
    ----------
    predictions
        Quantile forecasts with shape ``[..., num_quantiles, prediction_length]``.
    targets
        Observations with shape ``[..., prediction_length]``.
    quantile_levels
        Quantile levels corresponding to the quantile axis of ``predictions``.
        Levels must lie in ``(0, 1)`` and be strictly increasing after sorting.
    """
    if not isinstance(quantile_levels, torch.Tensor):
        quantile_levels = list(quantile_levels)
    levels = torch.as_tensor(
        quantile_levels, dtype=predictions.dtype, device=predictions.device
    ).reshape(-1)

    if predictions.ndim < 2:
        raise ValueError(
            "predictions must have shape [..., num_quantiles, prediction_length], "
            f"got {tuple(predictions.shape)}"
        )
    if levels.numel() != predictions.shape[-2]:
        raise ValueError(
            "quantile_levels must match the quantile axis of predictions. "
            f"Got {levels.numel()} levels and predictions shape {tuple(predictions.shape)}."
        )
    expected_target_shape = predictions.shape[:-2] + predictions.shape[-1:]
    if targets.shape != expected_target_shape:
        raise ValueError(
            "targets must have shape [..., prediction_length] matching predictions "
            f"without the quantile axis. Got targets {tuple(targets.shape)}, "
            f"expected {tuple(expected_target_shape)}."
        )

    order = torch.argsort(levels)
    levels = levels[order]
    predictions = predictions.index_select(dim=-2, index=order)
    if levels.numel() > 1 and torch.any(levels[1:] <= levels[:-1]):
        raise ValueError("quantile_levels must be strictly increasing and unique.")
    if torch.any(levels <= 0) or torch.any(levels >= 1):
        raise ValueError("quantile_levels must lie strictly inside (0, 1).")

    pointwise_loss = quantile_loss(predictions, targets, levels)
    weights = _crps_trapezoid_weights(levels).view(
        *([1] * (pointwise_loss.ndim - 2)), -1, 1
    )
    # Sum over quantile levels (the integral), then average over all other axes.
    return torch.sum(pointwise_loss * weights, dim=-2).mean()


def mean_absolute_scaled_error(
    contexts: Union[torch.Tensor, list, np.ndarray],
    targets: Union[torch.Tensor, list, np.ndarray],
    predictions: Union[torch.Tensor, list, np.ndarray],
    sp: int = 1,
    eps: float = 1e-10,
) -> Union[float, torch.Tensor]:
    """
    Calculates the Mean Absolute Scaled Error (MASE) between predictions and targets.

    Parameters
    ----------
    contexts
        The historical time series data used for training, with shape [num_samples, context_length].
    targets
        The true future values for the test set, with shape [num_samples, prediction_length].
    predictions
        The predicted future values for the test set, with shape [num_samples, prediction_length].
    sp
        The seasonal period of the time series (default=1 for non-seasonal data).
    eps
        A small value to prevent division by zero when the training set baseline error is zero.

    Returns
    -------
    The MASE value.
    """

    # Unified conversion to PyTorch tensors
    def to_tensor(x):
        return (
            torch.as_tensor(x, dtype=torch.float32)
            if not isinstance(x, torch.Tensor)
            else x
        )

    contexts = to_tensor(contexts)
    targets = to_tensor(targets)
    predictions = to_tensor(predictions)

    # Input shape validation
    if contexts.ndim != 2:
        raise ValueError(
            f"contexts must be a 2D tensor [num_samples, context_length], current shape: {contexts.shape}"
        )
    if targets.ndim != 2 or predictions.ndim != 2:
        raise ValueError(
            "targets and predictions must be 2D tensors [num_samples, prediction_length]"
        )
    if (
        contexts.shape[0] != targets.shape[0]
        or targets.shape[0] != predictions.shape[0]
    ):
        raise ValueError(
            "contexts, targets, predictions must have the same num_samples"
        )
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape")

    num_samples, context_length = contexts.shape
    if context_length <= sp:
        raise ValueError(
            f"Training sequence length context_length ({context_length}) must be greater than seasonal period sp ({sp})"
        )

    # Calculate the training set baseline error (MAE_naive) for each sample.
    if sp == 1:
        # Non-seasonal: First-order difference [num_samples, context_length-1]
        train_diff = (
            contexts[:, 1:] - contexts[:, :-1]
        )  # Use the previous value to predict the next value.
    else:
        # Seasonality: Seasonal Difference [num_samples, context_length-sp]
        train_diff = contexts[:, sp:] - contexts[:, :-sp]
    mae_naive = torch.mean(torch.abs(train_diff), dim=1)  # [num_samples]

    # Calculate the prediction error (MAE_pred) for each sample.
    pred_errors = torch.abs(targets - predictions)  # [num_samples, prediction_length]
    mae_pred = torch.mean(pred_errors, dim=1)  # [num_samples]

    # Calculate the MASE for each sample (avoid division by zero).
    mae_naive_clamped = torch.clamp(mae_naive, min=eps)
    mase_per_sample = mae_pred / mae_naive_clamped  # [num_samples]

    return torch.mean(mase_per_sample).item()


def calculate_metrics(
    contexts: torch.FloatTensor,
    predictions: torch.FloatTensor,
    targets: torch.FloatTensor,
    medium_index: int = 5,
    quantile_levels: Union[Sequence[float], torch.Tensor] = DEFAULT_QUANTILE_LEVELS,
) -> Dict[str, torch.FloatTensor]:
    """
    Calculates a set of common regression metrics between predictions and targets.
    Includes Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE),
    Mean Absolute Percentage Error (MAPE), Mean Squared Percentage Error (MSPE), Root Squared Error (RSE),
    and Continuous Ranked Probability Score (CRPS).

    Parameters
    ----------
    contexts
        The historical time series data used for training, with shape [batch_size, num_variables, context_length].
    predictions
        The predicted values of LiteSpecFormer, expected to have shape [batch_size, num_variables, num_quantiles, prediction_length].
    targets
        The true values for the test set, expected to have shape [batch_size, num_variables, prediction_length].
    medium_index
        The index of the median quantile (0.5) in the predictions, used for point-forecast metrics (default is 5 for 11 quantiles).
    quantile_levels
        Quantile levels aligned with the quantile axis of ``predictions``.
        CRPS uses these levels after dropping 0.05 and 0.95.

    Returns
    -------
    A dictionary containing the calculated metrics:
    - "mse": Mean Squared Error
    - "mae": Mean Absolute Error
    - "rmse": Root Mean Squared Error
    - "mape": Mean Absolute Percentage Error
    - "mase": Mean Absolute Scaled Error
    - "mspe": Mean Squared Percentage Error
    - "rse": Root Squared Error
    - "crps": Continuous Ranked Probability Score
    """

    # Input validation
    batch_size, num_variables, context_length = contexts.shape
    if predictions.ndim != 4:
        raise ValueError(
            "predictions must have shape [batch_size, num_variables, num_quantiles, prediction_length], "
            f"got {tuple(predictions.shape)}"
        )
    if predictions.shape[0] != batch_size:
        raise ValueError(
            f"Batch size of predictions must match contexts. Got predictions batch size: {predictions.shape[0]}, contexts batch size: {batch_size}"
        )
    if predictions.shape[1] != num_variables:
        raise ValueError(
            f"Number of variables in predictions must match contexts. Got predictions num variables: {predictions.shape[1]}, contexts num variables: {num_variables}"
        )
    prediction_length = predictions.shape[-1]
    if not 0 <= medium_index < predictions.shape[2]:
        raise ValueError(
            f"medium_index must address the quantile axis of predictions, got {medium_index} for {predictions.shape[2]} quantiles."
        )

    # CRPS integrates 0.1 ... 0.9. Point metrics below still use the median forecast.
    crps_predictions, crps_levels = _select_crps_quantiles(predictions, quantile_levels)
    crps = continuous_ranked_probability_score(
        predictions=crps_predictions,
        targets=targets,
        quantile_levels=crps_levels,
    )

    # Get the median quantile predictions for metric calculation
    predictions = predictions[:, :, medium_index, :]

    return {
        "mse": mean_squared_error(predictions, targets),
        "mae": mean_absolute_error(predictions, targets),
        "rmse": root_mean_squared_error(predictions, targets),
        "mape": mean_absolute_percentage_error(predictions, targets),
        "mase": mean_absolute_scaled_error(
            contexts=torch.reshape(
                contexts,
                shape=(batch_size * num_variables, context_length),
            ),
            targets=torch.reshape(
                targets,
                shape=(batch_size * num_variables, prediction_length),
            ),
            predictions=torch.reshape(
                predictions,
                shape=(batch_size * num_variables, prediction_length),
            ),
        ),
        "mspe": mean_squared_percentage_error(predictions, targets),
        "rse": root_squared_error(predictions, targets),
        "crps": crps,
    }
