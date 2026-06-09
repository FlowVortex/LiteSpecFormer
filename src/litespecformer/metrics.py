from typing import Dict, Union

import numpy as np
import torch


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


def quantile_loss(predictions: np.ndarray, targets: np.ndarray, q: float) -> float:
    r"""
    Quantile loss.

    .. math::
        quantile\_loss = 2 * sum(|(Y - \hat{Y}) * (Y <= \hat{Y}) - q|)
    """
    return 2 * np.sum(np.abs((predictions - targets) * ((targets <= predictions) - q)))


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
) -> Dict[str, torch.FloatTensor]:
    """
    Calculates a set of common regression metrics between predictions and targets.
    Includes Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE),
    Mean Absolute Percentage Error (MAPE), Mean Squared Percentage Error (MSPE), and Root Squared Error (RSE).

    Parameters
    ----------
    contexts
        The historical time series data used for training, with shape [batch_size, num_variables, context_length].
    predictions
        The predicted values of LiteSpecFormer, expected to have shape [batch_size, num_variables, num_quantiles, prediction_length].
    targets
        The true values for the test set, expected to have shape [batch_size, num_variables, prediction_length].
    medium_index
        The index of the median quantile (0.5) in the predictions, used for loss calculation and evaluation (default is 5 for 11 quantiles).

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
    """

    # Input validation
    batch_size, num_variables, context_length = contexts.shape
    if predictions.shape[0] != batch_size:
        raise ValueError(
            f"Batch size of predictions must match contexts. Got predictions batch size: {predictions.shape[0]}, contexts batch size: {batch_size}"
        )
    if predictions.shape[1] != num_variables:
        raise ValueError(
            f"Number of variables in predictions must match contexts. Got predictions num variables: {predictions.shape[1]}, contexts num variables: {num_variables}"
        )
    prediction_length = predictions.shape[-1]

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
    }
