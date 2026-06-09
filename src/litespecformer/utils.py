from typing import (
    Union,
    Any,
    Dict,
    Tuple,
    List,
    Iterator,
    Mapping,
    Sequence,
    TypeAlias,
)
import csv
import os
from os import path
import math

from accelerate import Accelerator

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F

from einops import repeat

from litespecformer.config import DatasetMode

TensorOrArray: TypeAlias = torch.Tensor | np.ndarray


def _check_quantiles(
    num_quantiles: list[float], median_quantile_index: int = None
) -> None:
    """
    Validate quantile configuration used by quantile forecasting components.

    This function performs two consistency checks:
    1) If `median_quantile_index` is provided, it must be a valid index into
       `num_quantiles`.
    2) The quantile range must be valid, i.e., the first and last elements of
       `num_quantiles` satisfy: 0 <= lower_bound < upper_bound <= 1.

    Parameters
    ----------
    num_quantiles : list[float]
        Ordered list of quantile levels (e.g., [0.1, 0.5, 0.9]).
        The implementation assumes the first element is the lower bound and
        the last element is the upper bound of the quantile interval.
    median_quantile_index : int, optional
        Index indicating which entry in `num_quantiles` should be treated as
        the "median-like" prediction (often corresponding to quantile 0.5).
        If provided, it must satisfy:
            0 <= median_quantile_index < len(num_quantiles)

    Raises
    ------
    AssertionError
        If `median_quantile_index` is provided but falls outside the valid
        index range of `num_quantiles`.
    ValueError
        If the boundary quantiles are invalid, i.e., the interval does not
        satisfy 0 <= first < last <= 1.

    Notes
    -----
    - This function currently validates only the boundary constraints of
      `num_quantiles` and index validity.
    - It does not check whether:
      * the list is strictly sorted internally,
      * quantile values are unique,
      * 0.5 exists when a median index is required.
      Those checks may be added separately if stricter validation is needed.
    """
    if median_quantile_index is not None:
        # If num_quantiles is provided, median_quantile should be a valid index
        assert (
            0 <= median_quantile_index < len(num_quantiles)
        ), "Median quantile index should be within the range of num_quantiles"

    # Check that num_quantiles is a list of quantiles between 0 and 1, with the first element less than the last element
    lower_bound = num_quantiles[0]
    upper_bound = num_quantiles[-1]

    if not (0 <= lower_bound < upper_bound <= 1):
        raise ValueError(
            "num_quantiles should be a list of quantiles between 0 and 1, with the first element less than the last element."
        )


def plot_confidence_prediction(
    context: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    num_quantiles: list[float] = None,
    median_quantile_index: int = None,
    figsize: tuple = (12, 3),
    dpi: int = 128,
    grid: bool = True,
    context_color: str = "royalblue",
    prediction_color: str = "darkorange",
    confidence_color: str = "darkorange",
) -> plt.Figure:
    """
    Plots the context, target, and confidence prediction in a single figure for `LiteSpecFormerPipeline`.

    Parameters
    ----------
    context
        A 1D numpy array containing the context (historical) data with the shape of (context_length,).
    target
        A 1D numpy array containing the target (future) data with the shape of (prediction_length,).
    prediction
        A 2D numpy array containing the quantile predictions with the shape of (num_quantiles, prediction_length).
    num_quantiles
        A list of quantiles corresponding to the predictions in `prediction`.
        The first element should be the lower quantile (e.g., 0.05 for 5% quantile) and
        the last element should be the upper quantile (e.g., 0.95 for 95% quantile).
        If not provided, it is assumed that the first row of `prediction` corresponds
        to the lower bound and the last row corresponds to the upper bound.
    median_quantile_index
        An integer specifying the index of the median quantile in `prediction`.
        If not provided, it defaults to the middle quantile (e.g., 0.5 quantile)
        if `num_quantiles` is provided, or the middle row of `prediction` if `num_quantiles` is not provided.
    figsize
        A tuple specifying the size of the figure (width, height) in inches. Default is (12, 3).
    dpi
        An integer specifying the resolution of the figure in dots per inch. Default is 128.
    grid
        A boolean indicating whether to display a grid on the plot. Default is True.
    context_color
        A string specifying the color for the context line. Default is "royalblue".
    prediction_color
        A string specifying the color for the prediction line. Default is "darkorange".
    confidence_color
        A string specifying the color for the confidence interval. Default is "darkorange".

    Returns
    -------
    plt.Figure
        A matplotlib Figure object containing the plot of context, target, and confidence prediction.
    """

    # Check the shapes of the inputs and ensure they are compatible
    assert context.ndim == 1, "Context should be a 1D array"
    assert target.ndim == 1, "Target should be a 1D array"
    assert (
        prediction.ndim == 2
    ), "Prediction should be a 2D array (num_quantiles, num_output_patches)"
    assert (
        prediction.shape[1] == target.shape[0]
    ), "Prediction's output patches should match the length of target"

    # Create an array of time steps for plotting
    time_steps = np.arange(len(context) + len(target))

    # Create the figure and axis for plotting
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if grid:
        ax.grid(alpha=0.18, color="gray", linestyle="--")

    # Ploting the context and target time series data
    ax.plot(
        time_steps,
        np.concatenate([context, target]),
        label="Context",
        color=context_color,
    )

    # Ploting the median prediction (50% quantile)
    if num_quantiles is not None and median_quantile_index is not None:
        # If both num_quantiles and median_quantile_index are provided, perform checks
        _check_quantiles(
            num_quantiles=num_quantiles, median_quantile_index=median_quantile_index
        )
        # Calculate the confidence interval
        max_confidence_interval = np.max(upper_bound - lower_bound)

    if num_quantiles is not None and median_quantile_index is None:
        # If num_quantiles is provided but median_quantile_index is not specified,
        # a check is performed and the median is used by default.
        _check_quantiles(
            num_quantiles=num_quantiles, median_quantile_index=median_quantile_index
        )
        # Default to the middle quantile if not specified
        median_quantile_index = len(num_quantiles) // 2
        # Calculate the confidence interval
        max_confidence_interval = np.max(upper_bound - lower_bound)

    if num_quantiles is None and median_quantile_index is not None:
        # If num_quantiles is not provided but median_quantile_index is specified, then a check is performed.
        assert (
            median_quantile_index <= prediction.shape[0]
        ), "Median quantile index should be less than or equal to the number of quantiles in prediction"
        # Unable to calculate confidence interval
        max_confidence_interval = None

    elif num_quantiles is None and median_quantile_index is None:
        # Default to the middle quantile
        median_quantile_index = prediction.shape[0] // 2
        # Unable to calculate confidence interval
        max_confidence_interval = None

    median_prediction = prediction[median_quantile_index, :]  # 50% quantile
    ax.plot(
        time_steps[len(context) :],
        median_prediction,
        label="Median Prediction",
        color=prediction_color,
        linestyle="--",
    )

    # Ploting the 95% confidence interval
    lower_bound = prediction[0, :]  # 5% quantile
    upper_bound = prediction[-1, :]  # 95% quantile
    ax.fill_between(
        time_steps[len(context) :],
        lower_bound,
        upper_bound,
        alpha=0.2,
        color=confidence_color,
        label=(
            str(np.round(max_confidence_interval * 100, 2)) + "% Confidence Interval"
            if max_confidence_interval is not None
            else "Confidence Interval"
        ),
    )

    ax.set_xlabel("Time Steps", fontsize=11)
    ax.set_ylabel("Values", fontsize=11)
    ax.legend(loc="upper left", fontsize="9")

    return fig


class Patch(nn.Module):
    """
    Convert a time series into sliding temporal patches.

    This module first ensures that the time dimension is divisible by `patch_size`
    by left-padding with NaNs when needed, then extracts patches using
    `torch.Tensor.unfold` along the last dimension.

    Typical use case:
    - Input shape: (..., time_length)
    - Output shape: (..., num_patches, patch_size)
    where:
        num_patches = floor((padded_length - patch_size) / patch_stride) + 1
    """

    def __init__(self, patch_size: int, patch_stride: int) -> None:
        """
        Parameters
        ----------
        patch_size : int
            Number of time steps contained in each patch.
        patch_stride : int
            Step size between consecutive patches along the time dimension.
            - `patch_stride < patch_size`: overlapping patches
            - `patch_stride = patch_size`: non-overlapping patches
            - `patch_stride > patch_size`: gapped patches
        """
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply left-padding (if required) and extract sliding patches.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor whose last dimension is interpreted as time.
            Expected shape: (..., length).

        Returns
        -------
        torch.Tensor
            Tensor of extracted patches with shape
            (..., num_patches, patch_size).

        Notes
        -----
        - Padding is added on the LEFT side of the time axis using NaN values.
          This preserves the most recent observations at the right side.
        - If `length` is already divisible by `patch_size`, no padding is added.
        - Patching is performed via `unfold(dimension=-1, size=patch_size, step=patch_stride)`.
        """
        # Original length of the time axis (last dimension).
        length = x.shape[-1]

        # If length is not divisible by patch_size, pad on the left so that
        # the patched representation aligns to full patch boundaries.
        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )

            # NaN padding is used so downstream modules can distinguish
            # padded tokens from real observations if masking is applied.
            padding = torch.full(
                size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device
            )
            x = torch.concat((padding, x), dim=-1)

        # Extract sliding windows (patches) along time axis.
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x


class InstanceNorm(nn.Module):
    """
    Reversible instance-wise normalization for time series.

    This module normalizes each sample independently along the last dimension
    (time axis), using NaN-aware statistics:
      - location (mean): `loc`
      - scale (standard deviation-like): `scale`

    It supports an optional `arcsinh` transform after normalization, which is
    useful for heavy-tailed or high-dynamic-range signals.

    The transformation is invertible through `inverse(...)` as long as the same
    `(loc, scale)` pair is provided.
    """

    def __init__(self, eps: float = 1e-5, use_arcsinh: bool = False) -> None:
        """
        Parameters
        ----------
        eps : float, optional
            Small positive fallback value used when estimated scale is zero.
            Prevents division-by-zero during normalization.
        use_arcsinh : bool, optional
            If True, applies `arcsinh` to normalized values in `forward`,
            and applies the inverse `sinh` in `inverse`.
        """
        super().__init__()
        self.eps = eps
        self.use_arcsinh = use_arcsinh

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Normalize input time series and return normalization statistics.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor where the last dimension is treated as time.
            Shape can be arbitrary, e.g. (batch, channels, time) or (batch, time).
        loc_scale : tuple[torch.Tensor, torch.Tensor] | None, optional
            Precomputed `(loc, scale)` used for normalization.
            - If None: compute NaN-aware statistics from `x`.
            - If provided: reuse external statistics (e.g., for consistent scaling
              across related tensors).

        Returns
        -------
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
            normalized_x : torch.Tensor
                Normalized tensor (and optionally arcsinh-transformed), same dtype as input.
            (loc, scale) : tuple[torch.Tensor, torch.Tensor]
                Statistics used for normalization, to be reused by `inverse(...)`.

        Notes
        -----
        - Internal computation is promoted to float32 for numerical stability.
        - NaN handling:
          * Mean uses `nanmean`; if fully NaN along time, fallback loc=0.
          * Scale uses NaN-aware RMS around loc; if fully NaN, fallback scale=1.
        - Zero scale values are replaced by `eps`.
        """
        orig_dtype = x.dtype
        x = x.to(torch.float32)

        if loc_scale is None:
            # NaN-aware mean over the time dimension.
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)

            # NaN-aware standard deviation-like scale over the time dimension.
            scale = torch.nan_to_num(
                (x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0
            )

            # Avoid division by zero for constant/degenerate sequences.
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            # Reuse externally provided normalization statistics.
            loc, scale = loc_scale

        # Standardize: zero-center and scale to unit variance-like range.
        scaled_x = (x - loc) / scale

        # Optional monotonic transform to compress large magnitudes.
        if self.use_arcsinh:
            scaled_x = torch.arcsinh(scaled_x)

        return scaled_x.to(orig_dtype), (loc, scale)

    def inverse(
        self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Invert normalization and recover values in the original scale.

        Parameters
        ----------
        x : torch.Tensor
            Normalized tensor produced by `forward(...)` (possibly after arcsinh).
        loc_scale : tuple[torch.Tensor, torch.Tensor]
            The exact `(loc, scale)` returned by `forward(...)`.

        Returns
        -------
        torch.Tensor
            Tensor mapped back to the original value space, same dtype as input `x`.

        Notes
        -----
        Inversion order:
        1) If `use_arcsinh=True`, apply `sinh` to undo `arcsinh`.
        2) De-standardize with `x * scale + loc`.
        """
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        loc, scale = loc_scale

        # Undo optional arcsinh transform.
        if self.use_arcsinh:
            x = torch.sinh(x)

        # Restore original location and scale.
        x = x * scale + loc

        return x.to(orig_dtype)


def left_pad_and_stack_1D(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    This function finds the longest tensor in a list and pads the other tensors to make them the same length.
    The padding method involves padding the shortest tensors to ensure all tensors have the same length.
    The padding method uses full-value padding (NaN).

    The significance of this function lies in considering that the sequence values of different samples have different lengths.

    To achieve comprehensive prediction, it is necessary to pad sequences of different lengths to make them all the same length.

    Parameters
    ----------
    tensors (List[torch.Tensor]): A list of 1D time series data.

    Returns
    -------
    torch.Tensor: A tensor with the same shape as the input, but with the shortest tensors padded to the same length.
    """

    # Get the maximum length in the list of 1D time series data
    max_len = max(len(c) for c in tensors)
    padded = []

    for c in tensors:
        # Ensure that the data is one-dimensional tensor data.
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 1

        # Fill all input data to the desired length.
        padding = torch.full(
            size=(max_len - len(c),), fill_value=torch.nan, device=c.device
        )
        padded.append(torch.concat((padding, c), dim=-1))

    # Return the filled result
    return torch.stack(padded)


def left_pad_and_cat_2D(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Left pads tensors in the list to the length of the longest tensor along the second axis, then concats
    these equal length tensors along the first axis.
    """
    # Get the maximum length of tensors in a batch of data.
    max_len = max(tensor.shape[-1] for tensor in tensors)
    padded = []

    for tensor in tensors:
        n_variates, length = tensor.shape
        if length < max_len:
            # Filling short time series
            padding = torch.full(
                (n_variates, max_len - length),
                fill_value=torch.nan,
                device=tensor.device,
            )
            tensor = torch.cat([padding, tensor], dim=-1)
        padded.append(tensor)

    return torch.cat(padded, dim=0)


def validate_and_prepare_single_dict_task(
    task: Mapping[str, TensorOrArray | Mapping[str, TensorOrArray]],
    idx: int,
    prediction_length: int,
    min_past: int = 32,
    mode: str = DatasetMode.TRAIN,
) -> Union[torch.Tensor, None]:
    """Validates and prepares a single dictionary task for LiteSpecFormerModel.

    The code is adapted from the Chronos2Pipeline.validate_and_prepare_single_dict_task method in the Chronos library
    in https://github.com/amazon-science/chronos-forecasting
    with modifications to fit the LiteSpecFormer model and use case.

    Parameters
    ----------
    task
        A dictionary representing a time series task with "target" key.
        The target must be a 1-d or 2-d `torch.Tensor` or `np.ndarray`
        with shape (history_length,) or (n_variates, history_length).
    idx
        Index of this task in the list of tasks, used for error messages
    prediction_length
        Number of future time steps to predict, used to validate future covariates

    Returns
    ------
    task_target: Concatenated tensor of target and past covariates of shape (history_length,)
    """
    # validate keys
    keys = set(task.keys())
    if "target" not in keys:
        raise ValueError(
            f"Element at index {idx} does not contain the required key 'target'"
        )

    # validate target
    task_target = task["target"]

    if isinstance(task_target, np.ndarray):
        task_target = torch.from_numpy(task_target)
    assert isinstance(task_target, torch.Tensor)
    if task_target.ndim > 2:
        raise ValueError(
            "When the input is a list of dicts, the `target` should either be 1-d with shape (history_length,) "
            f" or 2-d with shape (n_variates, history_length). Found element at index {idx} with shape {tuple(task_target.shape)}."
        )
    history_length = task_target.shape[-1]

    if mode != DatasetMode.TEST:
        if history_length < min_past + prediction_length:
            # filter out time series that are too short to provide the required context and prediction length
            return None
    else:
        if history_length < min_past:
            # filter out time series that are too short to provide the required prediction length
            return None

    return task_target


def convert_df_input_to_list_of_dicts_input(
    df: "pd.DataFrame",
    future_df: "pd.DataFrame | None",
    target_columns: list[str],
    prediction_length: int,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> tuple[
    list[dict[str, np.ndarray | dict[str, np.ndarray]]],
    np.ndarray,
    dict[str, "pd.DatetimeIndex"],
]:
    """
    Convert from dataframe input format to a list of dictionaries input format.

    Parameters
    ----------
    df
        Input dataframe containing time series data with columns:
        - id_column: Identifier for each time series
        - timestamp_column: Timestamps for each observation
        - target_columns: One or more target variables to forecast
        - Additional columns are treated as covariates
    future_df
        Optional dataframe containing future covariate values with columns:
        - id_column: Identifier for each time series
        - timestamp_column: Future timestamps
        - Subset of covariate columns from df
    target_columns
        Names of target columns to forecast
    prediction_length
        Number of future time steps to predict
    id_column
        Name of column containing time series identifiers
    timestamp_column
        Name of column containing timestamps

    Returns
    -------
    A tuple containing:
    - List of dictionaries in the format expected by `LiteSpecFormerPipeline.predict`
    - Original order of time series IDs
    - Dictionary mapping series IDs to future time index
    """

    import pandas as pd

    (
        df,
        future_df,
        freq,
        series_lengths,
        future_series_lengths,
        original_order,
    ) = validate_df_inputs(
        df,
        future_df=future_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
        target_columns=target_columns,
        prediction_length=prediction_length,
    )

    # Convert to list of dicts format
    inputs: list[dict[str, np.ndarray | dict[str, np.ndarray]]] = []
    prediction_timestamps: dict[str, pd.DatetimeIndex] = {}
    start_idx: int = 0
    future_start_idx: int = 0

    for i, length in enumerate(series_lengths):
        series_data = df.iloc[start_idx : start_idx + length]
        # Extract target(s)
        target_data = (
            series_data[target_columns].to_numpy().T
        )  # Shape: (n_targets, history_length)
        task: dict[str, np.ndarray | dict[str, np.ndarray]] = {"target": target_data}

        # Generate future timestamps
        series_id = series_data.iloc[0][id_column]
        last_timestamp = series_data[timestamp_column].iloc[-1]
        future_ts = pd.date_range(
            start=last_timestamp, periods=prediction_length + 1, freq=freq
        )[1:]
        prediction_timestamps[series_id] = future_ts

        # Handle covariates if present
        covariate_cols = [
            col
            for col in series_data.columns
            if col not in [id_column, timestamp_column] + target_columns
        ]

        if covariate_cols:
            past_covariates = {
                col: series_data[col].to_numpy() for col in covariate_cols
            }
            task["past_covariates"] = past_covariates

            # Handle future covariates
            if future_df is not None:
                assert future_series_lengths is not None
                future_length = future_series_lengths[i]
                future_data = future_df.iloc[
                    future_start_idx : future_start_idx + future_length
                ]
                assert future_data[timestamp_column].iloc[0] == future_ts[0], (
                    f"the first timestamp in future_df must be the first forecast timestamp, found mismatch "
                    f"({future_data[timestamp_column].iloc[0]} != {future_ts[0]}) in series {series_id}"
                )

                if len(future_data) > 0:
                    future_covariates = {
                        col: future_data[col].to_numpy()
                        for col in covariate_cols
                        if col in future_data.columns
                    }
                    if future_covariates:
                        task["future_covariates"] = future_covariates
                future_start_idx += future_length

        inputs.append(task)
        start_idx += length

    assert len(inputs) == len(series_lengths)

    return inputs, original_order, prediction_timestamps


def validate_df_inputs(
    df: "pd.DataFrame",
    future_df: "pd.DataFrame | None",
    target_columns: list[str],
    prediction_length: int,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> tuple[
    "pd.DataFrame",
    "pd.DataFrame | None",
    "pd.Timedelta",
    list[int],
    list[int] | None,
    np.ndarray,
]:
    """
    Validates and prepares dataframe inputs passed to `LiteSpecFormerPipeline.predict_df`.

    Parameters
    ----------
    df
        Input dataframe containing time series data with columns:
        - id_column: Identifier for each time series
        - timestamp_column: Timestamps for each observation
        - target_columns: One or more target variables to forecast
        - Additional columns are treated as covariates
    future_df
        Optional dataframe containing future covariate values with columns:
        - id_column: Identifier for each time series
        - timestamp_column: Future timestamps
        - Subset of covariate columns from df
    target_columns
        Names of target columns to forecast
    prediction_length
        Number of future time steps to predict
    id_column
        Name of column containing time series identifiers
    timestamp_column
        Name of column containing timestamps

    Returns
    -------
    A tuple containing:
    - Validated and sorted input dataframe
    - Validated and sorted future dataframe (if provided)
    - Inferred frequency of the time series
    - List of series lengths from input dataframe
    - List of series lengths from future dataframe (if provided)
    - Original order of time series IDs

    Raises
    ------
    ValueError
        If validation fails for:
        - Missing required columns
        - Invalid data types
        - Inconsistent frequencies
        - Insufficient data points
        - Mismatched series between df and future_df
        - Invalid future_df lengths
    """

    import pandas as pd

    required_cols = [id_column, timestamp_column] + target_columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"df does not contain all expected columns. Missing columns: {missing_cols}"
        )

    if future_df is not None:
        future_required_cols = [id_column, timestamp_column]
        missing_future_cols = [
            col for col in future_required_cols if col not in future_df.columns
        ]
        targets_in_future = [col for col in future_df.columns if col in target_columns]
        extra_future_cols = [col for col in future_df.columns if col not in df.columns]
        if missing_future_cols:
            raise ValueError(
                f"future_df does not contain all expected columns. Missing columns: {missing_future_cols}"
            )
        if targets_in_future:
            raise ValueError(
                f"future_df cannot contain target columns. Target columns found in future_df: {targets_in_future}"
            )
        if extra_future_cols:
            raise ValueError(
                f"future_df cannot contain columns not present in df. Extra columns: {extra_future_cols}"
            )

    df, future_df = _validate_df_types_and_cast(
        df,
        future_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
        target_columns=target_columns,
    )

    # Get the original order of time series IDs
    original_order = df[id_column].unique()

    # Sort and prepare df
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df = df.sort_values([id_column, timestamp_column])

    # Get series lengths
    series_lengths = df[id_column].value_counts(sort=False).to_list()

    def validate_freq(timestamps: pd.Series, series_id: str):
        freq = pd.infer_freq(timestamps)
        if not freq:
            raise ValueError(f"Could not infer frequency for series {series_id}")
        return freq

    # Validate each series
    all_freqs = []
    start_idx = 0
    for length in series_lengths:
        if length < 3:
            series_id = df.iloc[start_idx][id_column]
            raise ValueError(
                f"Every time series must have at least 3 data points, found {length=} for series {series_id}"
            )

        series_data = df.iloc[start_idx : start_idx + length]
        timestamps = series_data[timestamp_column]
        series_id = series_data.iloc[0][id_column]
        all_freqs.append(validate_freq(timestamps, series_id))
        start_idx += length

    if len(set(all_freqs)) > 1:
        raise ValueError("All time series must have the same frequency")

    inferred_freq = all_freqs[0]

    # Sort future_df if provided and validate its series lengths
    future_series_lengths = None
    if future_df is not None:
        future_df[timestamp_column] = pd.to_datetime(future_df[timestamp_column])
        future_df = future_df.sort_values([id_column, timestamp_column])

        # Validate that future_df contains all series from df
        context_ids = set(df[id_column].unique())
        future_ids = set(future_df[id_column].unique())
        if context_ids != future_ids:
            raise ValueError("future_df must contain the same time series IDs as df")

        future_series_lengths = future_df[id_column].value_counts(sort=False).to_list()

        # Validate future series lengths match prediction_length
        future_start_idx = 0
        for future_length in future_series_lengths:
            future_series_data = future_df.iloc[
                future_start_idx : future_start_idx + future_length
            ]
            future_timestamps = future_series_data[timestamp_column]
            future_series_id = future_series_data.iloc[0][id_column]
            if future_length != prediction_length:
                raise ValueError(
                    f"Future covariates all time series must have length {prediction_length}, got {future_length} for series {future_series_id}"
                )
            if future_length < 3 or inferred_freq != validate_freq(
                future_timestamps, future_series_id
            ):
                raise ValueError(
                    f"Future covariates must have the same frequency as context, found series {future_series_id} with a different frequency"
                )
            future_start_idx += future_length

        assert len(series_lengths) == len(future_series_lengths)

    return (
        df,
        future_df,
        inferred_freq,
        series_lengths,
        future_series_lengths,
        original_order,
    )


def _validate_df_types_and_cast(
    df: "pd.DataFrame",
    future_df: "pd.DataFrame | None",
    target_columns: list[str],
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> tuple["pd.DataFrame", "pd.DataFrame | None"]:
    import pandas as pd

    astype_dict = {}
    future_astype_dict = {}
    for col in df.columns.drop([id_column, timestamp_column]):
        col_dtype = df[col].dtype
        if col in target_columns and not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"All target columns must be numeric but got {col=} with dtype={col_dtype}"
            )

        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or isinstance(col_dtype, pd.CategoricalDtype)
        ):
            astype_dict[col] = "category"
        elif pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(
            df[col]
        ):
            astype_dict[col] = "float32"
        else:
            raise ValueError(
                f"All columns must contain numeric, object, category, string, or bool dtype but got {col=} with dtype={col_dtype}"
            )

        if future_df is not None and col in future_df.columns:
            if future_df[col].dtype != col_dtype:
                raise ValueError(
                    f"Column {col} in future_df has dtype {future_df[col].dtype} but column in df has dtype {col_dtype}"
                )
            future_astype_dict[col] = astype_dict[col]

    df = df.astype(astype_dict, copy=True)
    if future_df is not None:
        future_df = future_df.astype(future_astype_dict, copy=True)

    return df, future_df


def convert_list_of_tensors_input_to_list_of_dicts_input(
    list_of_tensors: Sequence[TensorOrArray],
) -> list[dict[str, torch.Tensor]]:
    """Convert a list of tensors input format to a list of dictionaries input format.


    Parameters
    ----------
    list_of_tensors
        A sequence of tensors or numpy arrays, where each element represents a time series.
        Each element should be either 1-d with shape (history_length,) or 2-d with shape
        (n_variates, history_length).

    Returns
    -------
    A list of dictionaries, where each dictionary represents a time series and contains:
    - `target`: a 1-d or 2-d torch.Tensor of shape (history_length,) or (n_variates, history_length).
    """

    output: list[dict[str, torch.Tensor]] = []
    for idx, tensor in enumerate(list_of_tensors):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if tensor.ndim > 2:
            raise ValueError(
                "When the input is a list of torch tensors or numpy arrays, the elements should either be 1-d with shape (history_length,) "
                f" or 2-d with shape (n_variates, history_length). Found element at index {idx} with shape {tuple(tensor.shape)}."
            )
        length = tensor.shape[-1]
        tensor = tensor.view(-1, length)

        output.append({"target": tensor})

    return output


def convert_tensor_input_to_list_of_dicts_input(
    tensor: TensorOrArray,
) -> list[dict[str, torch.Tensor]]:
    """
    Convert a tensor input format to a list of dictionaries input format.

    Parameters
    ----------
    tensor
        A tensor or numpy array representing multiple time series.
        Should be 3-d with shape (n_series, n_variates, history_length).

    Returns
    -------
    A list of dictionaries, where each dictionary represents a time series and contains:
    - `target`: a 2-d torch.Tensor of shape (n_variates, history_length).
    """

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if tensor.ndim != 2:
        raise ValueError(
            "When the input is a torch tensor or numpy array, it should be 2-d with shape (n_series, history_length). "
            f" Found shape: {tuple(tensor.shape)}."
        )

    output: list[dict[str, torch.Tensor]] = []
    n_series = tensor.shape[0]
    for i in range(n_series):
        output.append({"target": tensor[i]})

    return output


def interpolate_quantiles(
    query_quantile_levels: torch.Tensor | list[float],
    original_quantile_levels: torch.Tensor | list[float],
    original_values: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolates quantile values at specified query levels using linear interpolation using original
    quantile levels and their corresponding values. This behaves similar to `torch.quantile` in terms of
    the linear interpolation but also supports non-equidistant original quantile levels.

    Parameters
    ----------
    query_quantile_levels : torch.Tensor | list[float]
        The quantile levels at which to interpolate values, all levels must be between 0 and 1
    original_quantile_levels : torch.Tensor | list[float]
        The quantile levels corresponding to the original values, all levels must be between 0 and 1.
        Can be a 1D tensor or list matching the last dimension of `original_values`, or a tensor with the
        same shape as `original_values`
    original_values : torch.Tensor
        The values corresponding to the original quantile levels, can have any number of leading dimensions

    Returns
    -------
    torch.Tensor
        The interpolated quantiles at the query quantile levels. All leading dimensions have the same size
        as `original_values` and the last dimension has size `len(query_quantile_levels)`.
    """
    assert torch.is_floating_point(
        original_values
    ), "`original_values` must be a floating point tensor"
    orig_dtype = original_values.dtype
    if isinstance(query_quantile_levels, list):
        query_quantile_levels = torch.tensor(query_quantile_levels, dtype=torch.float32)
    if isinstance(original_quantile_levels, list):
        original_quantile_levels = torch.tensor(
            original_quantile_levels, dtype=torch.float32
        )

    assert (
        query_quantile_levels.ndim == 1
    ), "`query_quantile_levels` must be 1-dimensional"
    if original_quantile_levels.ndim > 1:
        assert (
            original_quantile_levels.shape == original_values.shape
        ), "If `original_quantile_levels` is not 1D, its shape must match `original_values`"
    else:
        assert (
            len(original_quantile_levels) == original_values.shape[-1]
        ), "If `original_quantile_levels` is 1D, its length must match the last dim of `original_values`"
    assert (
        query_quantile_levels.min() >= 0.0 and query_quantile_levels.max() <= 1.0
    ), "`query_quantile_levels` must be between 0 and 1"
    assert (
        original_quantile_levels.min() >= 0.0 and original_quantile_levels.max() <= 1.0
    ), "`original_quantile_levels` must be between 0 and 1"
    original_quantile_levels = torch.clamp(original_quantile_levels, min=0.0, max=1.0)

    device = original_values.device
    query_quantile_levels = query_quantile_levels.to(device)
    original_quantile_levels = original_quantile_levels.to(device)
    original_values = original_values.to(torch.float32)

    orig_values_shape = original_values.shape
    num_original_quantiles = original_quantile_levels.shape[-1]
    original_values = original_values.reshape(-1, num_original_quantiles)
    batch_size = original_values.shape[0]

    # If original_quantile_levels is 1D, expand it to match the batch dimension
    if original_quantile_levels.ndim == 1:
        original_quantile_levels = original_quantile_levels.expand(batch_size, -1)
    else:
        original_quantile_levels = original_quantile_levels.reshape(
            -1, num_original_quantiles
        )

    # Sort original quantile levels and the corresponding values
    sorted_levels, sorted_indices = torch.sort(original_quantile_levels, dim=-1)
    sorted_values = torch.gather(original_values, dim=-1, index=sorted_indices)

    # Add extreme quantiles (0., 1.) to handle extrapolation and queries at 0 or 1
    zeros_padding = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    ones_padding = torch.ones((batch_size, 1), dtype=torch.float32, device=device)

    # Only pad when extreme quantiles are not available in original_quantile_levels
    sorted_levels_with_padding = []
    sorted_values_with_padding = []
    if original_quantile_levels.min() > 0.0:
        sorted_levels_with_padding.append(zeros_padding)
        sorted_values_with_padding.append(sorted_values[:, :1])
    sorted_levels_with_padding.append(sorted_levels)
    sorted_values_with_padding.append(sorted_values)
    if original_quantile_levels.max() < 1.0:
        sorted_levels_with_padding.append(ones_padding)
        sorted_values_with_padding.append(sorted_values[:, -1:])

    sorted_levels = torch.cat(sorted_levels_with_padding, dim=-1).contiguous()
    sorted_values = torch.cat(sorted_values_with_padding, dim=-1)

    # Shape goes from (num_queries,) to (batch_size, num_queries).
    query_levels_expanded = repeat(
        query_quantile_levels, "q -> b q", b=batch_size
    ).contiguous()

    # Find (sorted) index of smallest original quantile level strictly larger than the query quantile level
    upper_indices = torch.searchsorted(sorted_levels, query_levels_expanded, right=True)
    upper_indices = torch.clamp(upper_indices, max=sorted_levels.shape[-1] - 1)
    lower_indices = upper_indices - 1

    # Gather the lower and upper levels and values for each item in the batch
    lower_levels = torch.gather(sorted_levels, dim=1, index=lower_indices)
    upper_levels = torch.gather(sorted_levels, dim=1, index=upper_indices)
    lower_values = torch.gather(sorted_values, dim=1, index=lower_indices)
    upper_values = torch.gather(sorted_values, dim=1, index=upper_indices)

    # Perform linear interpolation
    level_diff = upper_levels - lower_levels
    weight = torch.nan_to_num(
        (query_levels_expanded - lower_levels) / level_diff, nan=0.0
    )
    interpolated_values = lower_values + weight * (upper_values - lower_values)

    final_shape = (*orig_values_shape[:-1], len(query_quantile_levels))
    return interpolated_values.reshape(final_shape).to(orig_dtype)


def get_num_output_patches(
    remaining_horizon: int, model_output_patch_size: int, max_output_patches: int
) -> int:
    """
    Compute how many output patches should be generated for the current autoregressive step.

    Given the remaining forecast horizon, this function determines the minimum number
    of model output patches needed to cover it, then caps the value by
    `max_output_patches` to respect model/runtime constraints.

    Parameters
    ----------
    remaining_horizon : int
        Number of future time steps that still need to be predicted.
    model_output_patch_size : int
        Number of time steps produced by one output patch.
    max_output_patches : int
        Upper bound on how many patches are allowed in a single prediction call.

    Returns
    -------
    int
        Number of output patches to request from the model for this step.

    Notes
    -----
    - `ceil(remaining_horizon / model_output_patch_size)` ensures enough patches
      are requested to cover the remaining horizon.
    - `min(..., max_output_patches)` enforces a safety/efficiency cap.
    """
    num_output_patches = math.ceil(remaining_horizon / model_output_patch_size)
    num_output_patches = min(num_output_patches, max_output_patches)

    return num_output_patches


def interpolate_quantiles(
    query_quantile_levels: torch.Tensor | list[float],
    original_quantile_levels: torch.Tensor | list[float],
    original_values: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolates quantile values at specified query levels using linear interpolation using original
    quantile levels and their corresponding values. This behaves similar to `torch.quantile` in terms of
    the linear interpolation but also supports non-equidistant original quantile levels.

    Parameters
    ----------
    query_quantile_levels : torch.Tensor | list[float]
        The quantile levels at which to interpolate values, all levels must be between 0 and 1
    original_quantile_levels : torch.Tensor | list[float]
        The quantile levels corresponding to the original values, all levels must be between 0 and 1.
        Can be a 1D tensor or list matching the last dimension of `original_values`, or a tensor with the
        same shape as `original_values`
    original_values : torch.Tensor
        The values corresponding to the original quantile levels, can have any number of leading dimensions

    Returns
    -------
    torch.Tensor
        The interpolated quantiles at the query quantile levels. All leading dimensions have the same size
        as `original_values` and the last dimension has size `len(query_quantile_levels)`.
    """
    assert torch.is_floating_point(
        original_values
    ), "`original_values` must be a floating point tensor"
    orig_dtype = original_values.dtype
    if isinstance(query_quantile_levels, list):
        query_quantile_levels = torch.tensor(query_quantile_levels, dtype=torch.float32)
    if isinstance(original_quantile_levels, list):
        original_quantile_levels = torch.tensor(
            original_quantile_levels, dtype=torch.float32
        )

    assert (
        query_quantile_levels.ndim == 1
    ), "`query_quantile_levels` must be 1-dimensional"
    if original_quantile_levels.ndim > 1:
        assert (
            original_quantile_levels.shape == original_values.shape
        ), "If `original_quantile_levels` is not 1D, its shape must match `original_values`"
    else:
        assert (
            len(original_quantile_levels) == original_values.shape[-1]
        ), "If `original_quantile_levels` is 1D, its length must match the last dim of `original_values`"
    assert (
        query_quantile_levels.min() >= 0.0 and query_quantile_levels.max() <= 1.0
    ), "`query_quantile_levels` must be between 0 and 1"
    assert (
        original_quantile_levels.min() >= 0.0 and original_quantile_levels.max() <= 1.0
    ), "`original_quantile_levels` must be between 0 and 1"
    original_quantile_levels = torch.clamp(original_quantile_levels, min=0.0, max=1.0)

    device = original_values.device
    query_quantile_levels = query_quantile_levels.to(device)
    original_quantile_levels = original_quantile_levels.to(device)
    original_values = original_values.to(torch.float32)

    orig_values_shape = original_values.shape
    num_original_quantiles = original_quantile_levels.shape[-1]
    original_values = original_values.reshape(-1, num_original_quantiles)
    batch_size = original_values.shape[0]

    # If original_quantile_levels is 1D, expand it to match the batch dimension
    if original_quantile_levels.ndim == 1:
        original_quantile_levels = original_quantile_levels.expand(batch_size, -1)
    else:
        original_quantile_levels = original_quantile_levels.reshape(
            -1, num_original_quantiles
        )

    # Sort original quantile levels and the corresponding values
    sorted_levels, sorted_indices = torch.sort(original_quantile_levels, dim=-1)
    sorted_values = torch.gather(original_values, dim=-1, index=sorted_indices)

    # Add extreme quantiles (0., 1.) to handle extrapolation and queries at 0 or 1
    zeros_padding = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    ones_padding = torch.ones((batch_size, 1), dtype=torch.float32, device=device)

    # Only pad when extreme quantiles are not available in original_quantile_levels
    sorted_levels_with_padding = []
    sorted_values_with_padding = []
    if original_quantile_levels.min() > 0.0:
        sorted_levels_with_padding.append(zeros_padding)
        sorted_values_with_padding.append(sorted_values[:, :1])
    sorted_levels_with_padding.append(sorted_levels)
    sorted_values_with_padding.append(sorted_values)
    if original_quantile_levels.max() < 1.0:
        sorted_levels_with_padding.append(ones_padding)
        sorted_values_with_padding.append(sorted_values[:, -1:])

    sorted_levels = torch.cat(sorted_levels_with_padding, dim=-1).contiguous()
    sorted_values = torch.cat(sorted_values_with_padding, dim=-1)

    # Shape goes from (num_queries,) to (batch_size, num_queries).
    query_levels_expanded = repeat(
        query_quantile_levels, "q -> b q", b=batch_size
    ).contiguous()

    # Find (sorted) index of smallest original quantile level strictly larger than the query quantile level
    upper_indices = torch.searchsorted(sorted_levels, query_levels_expanded, right=True)
    upper_indices = torch.clamp(upper_indices, max=sorted_levels.shape[-1] - 1)
    lower_indices = upper_indices - 1

    # Gather the lower and upper levels and values for each item in the batch
    lower_levels = torch.gather(sorted_levels, dim=1, index=lower_indices)
    upper_levels = torch.gather(sorted_levels, dim=1, index=upper_indices)
    lower_values = torch.gather(sorted_values, dim=1, index=lower_indices)
    upper_values = torch.gather(sorted_values, dim=1, index=upper_indices)

    # Perform linear interpolation
    level_diff = upper_levels - lower_levels
    weight = torch.nan_to_num(
        (query_levels_expanded - lower_levels) / level_diff, nan=0.0
    )
    interpolated_values = lower_values + weight * (upper_values - lower_values)

    final_shape = (*orig_values_shape[:-1], len(query_quantile_levels))
    return interpolated_values.reshape(final_shape).to(orig_dtype)


def weighted_quantile(
    query_quantile_levels: torch.Tensor | list[float],
    sample_weights: torch.Tensor | list[float],
    samples: torch.Tensor,
):
    """
    Computes quantiles from a distribution specified by `samples` and their corresponding probability mass
    `sample_weights`. `samples` are first sorted along the last axis and an empirical cumulative distribution
    function (CDF) is constructed. Specific `query_quantile_levels` are then interpolated using this CDF.

    Parameters
    ----------
    query_quantile_levels : torch.Tensor | list[float]
        The quantile levels to interpolate from the empirical CDF, must be between 0 and 1
    sample_weights : torch.Tensor | list[float]
        The weights corresponding to each sample, must be non-negative. The sample_weights correspond to the
        last axis of `samples` and all leading batch dimensions share the same sample weights
    samples : torch.Tensor
        The sample values used to construct the empirical CDF along the last axis. The last dim must
        match the length of `sample_weights`, can have any number of leading dimensions

    Returns
    -------
    torch.Tensor
        The interpolated quantiles at the query quantile levels. All leading dimensions have the same size
        as `samples` and the last dimension has size `len(query_quantile_levels)`.
    """
    # FIXME: this interpolation works reasonably well in practice but may not be the best way to extrapolate
    assert torch.is_floating_point(
        samples
    ), "`original_values` must be a floating point tensor"
    orig_dtype = samples.dtype
    if isinstance(query_quantile_levels, list):
        query_quantile_levels = torch.tensor(query_quantile_levels, dtype=torch.float32)
    if isinstance(sample_weights, list):
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    assert (
        query_quantile_levels.ndim == 1 and sample_weights.ndim == 1
    ), "`query_quantile_levels` and `sample_weights` must be 1-dimensional"
    assert (
        len(sample_weights) == samples.shape[-1]
    ), "the last dim of `samples` must be equal to the length of `sample_weights`"
    assert (
        query_quantile_levels.min() >= 0.0 and query_quantile_levels.max() <= 1.0
    ), "`query_quantile_levels` must be between 0 and 1"
    assert sample_weights.min() > 0.0, "`sample_weights` must be > 0"

    device = samples.device
    query_quantile_levels = query_quantile_levels.to(device)
    sample_weights = sample_weights.to(device)
    samples = samples.to(torch.float32)

    orig_samples_shape = samples.shape
    num_samples = len(sample_weights)
    samples = samples.reshape(-1, num_samples)
    batch_size = samples.shape[0]

    # Normalize and expand weights
    sample_weights = sample_weights / sample_weights.sum(dim=-1, keepdim=True)
    sample_weights = sample_weights.expand(batch_size, -1).contiguous()

    # Sort samples and the corresponding weights
    sorted_samples, sort_indices = torch.sort(samples, dim=-1)
    sorted_weights = torch.gather(sample_weights, dim=-1, index=sort_indices)

    # Compute cumulative weights
    cumul_weights = torch.cumsum(sorted_weights, dim=-1)
    cumul_weights = torch.clamp(cumul_weights, min=0.0, max=1.0)

    # Get interpolated quantiles
    interpolated_quantiles = interpolate_quantiles(
        query_quantile_levels=query_quantile_levels,
        original_quantile_levels=cumul_weights,
        original_values=sorted_samples,
    )

    # Reshape to original shape
    final_shape = (*orig_samples_shape[:-1], len(query_quantile_levels))
    return interpolated_quantiles.reshape(final_shape).to(dtype=orig_dtype)


def calculate_batch_autocorrelation_function(
    x: torch.Tensor, max_lag: int = None
) -> torch.Tensor:
    """
    Calculate the sample Autocorrelation Function (ACF) that is exactly aligned with statsmodels.tsa.stattools.acf.
    Supports batch processing of inputs with shape [num_samples, seq_length].

    Parameters
    ----------
    x
        Input time series with shape [num_samples, seq_length]
    max_lag
        Maximum lag value, defaults to seq_length - 1

    Returns
    -------
    acf
        Normalized autocorrelation results with shape [num_samples, max_lag + 1]
        The results are exactly consistent with statsmodels.acf(nlags=max_lag, fft=False)
    """
    num_samples, seq_length = x.shape

    # 1. Preprocessing: Set max_lag & remove mean
    if max_lag is None:
        max_lag = seq_length - 1
    max_lag = min(max_lag, seq_length - 1)  # Prevent exceeding sequence length

    mean_val = x.mean(dim=-1, keepdim=True)
    x_centered = x - mean_val  # [num_samples, seq_length]

    # 2. Calculate denominator: Lag 0 autocorrelation (i.e., sum of variances) for all samples
    denominator = (x_centered**2).sum(dim=-1, keepdim=True)  # [num_samples, 1]
    denominator = denominator.clamp_min(1e-8)  # Prevent division by zero

    # 3. Calculate numerator: Compute autocovariance for each lag individually
    acf_list = []
    for lag in range(max_lag + 1):
        # For each lag k, calculate the sum of (x_t - μ)(x_{t+k} - μ)
        if lag == 0:
            # Lag 0: Sum of squares of all elements (equal to denominator)
            numerator = denominator
        else:
            # Slice alignment: Element-wise multiplication of x[:, :-lag] and x[:, lag:] followed by summation
            numerator = (x_centered[:, :-lag] * x_centered[:, lag:]).sum(
                dim=-1, keepdim=True
            )

        # Normalize to get ACF
        acf_lag = numerator / denominator
        acf_list.append(acf_lag)

    # 4. Concatenate results for all lags
    acf = torch.cat(acf_list, dim=-1)  # [num_samples, max_lag + 1]

    return acf


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Core function to count model parameters

    Parameters
    ----------
    model
        The model for which to count parameters

    Returns
    -------
    total_params
        Total number of parameters in the model (including trainable and non-trainable)
    trainable_params
        Number of trainable parameters in the model
    """
    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Total parameters (including non-trainable)
    total_params = sum(p.numel() for p in model.parameters())

    # Format output (convert to millions/thousands for better readability)
    def format_num(num):
        if num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return str(num)

    print(f"Total Parameters: {format_num(total_params)} ({total_params} units)")
    print(
        f"Trainable Parameters: {format_num(trainable_params)} ({trainable_params} units)"
    )
    return total_params, trainable_params


def cyclic_sample_generator(
    data_list: List[Any], sample_size: int
) -> Iterator[Tuple[Any, ...]]:
    """
    Looping Sample Generator: Continuously reads elements from a list in a specified loop.

    Parameters
    ----------
    data_list
        The original list of data from which to sample. This can be any list of elements, such as a list of file paths, a list of data samples, etc.
    sample_size
        The number of samples to read in each iteration. This determines how many elements are returned in each cycle.

    Yields
    -------
    A tuple containing the sampled elements from the list.
    The generator will yield a new tuple of samples in each iteration,
    and it will loop back to the beginning of the list when it reaches the end, ensuring continuous sampling.
    """
    # Verify parameter validity
    if not isinstance(data_list, list):
        raise TypeError("data_list must be a list type")
    if not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if len(data_list) == 0:
        raise ValueError("data_list cannot be an empty list")

    # Get the length of the data list for later use in indexing
    list_len = len(data_list)
    # Initialize the starting index for sampling
    start_idx = 0

    # Start an infinite loop to continuously yield samples
    while True:
        # Sample a batch of elements from the list based on the current starting index and sample size
        sample = []
        # Read the specified number of elements
        for i in range(sample_size):
            # Calculate the index of the current element (using modulo for circular indexing)
            current_idx = (start_idx + i) % list_len
            sample.append(data_list[current_idx])
        # Update the starting index for the next iteration
        start_idx = (start_idx + sample_size) % list_len
        # Return the current sample
        # (converted to tuples to prevent external modification).
        yield tuple(sample)


def logging_results(
    accelerator: Accelerator,
    logging_path: str,
    headers: List[str],
    messages: Dict[str, Union[str, float]],
) -> None:
    """
    The training results are recorded to a CSV file,
    supporting multi-GPU training scenarios.

    Parameters
    ----------
    accelerator
        An Accelerator object from the Accelerate library, used for multi-GPU synchronization and main process checking
    logging_path
        The file path (including filename) where the CSV file will be saved
    headers
        A list of column headers for the CSV file
    messages
        A dictionary of results to be recorded, where keys should correspond to the headers
    """
    # File write operations are performed only in the main process to avoid multi-process conflicts.
    if accelerator.is_main_process:
        # Check if the key of the message matches the headers.
        message_keys = set(messages.keys())
        header_set = set(headers)
        if not message_keys.issubset(header_set):
            missing_keys = message_keys - header_set
            raise ValueError(
                f"The message contains a key whose header does not exist.: {missing_keys}"
            )

        # Ensure the directory exists
        os.makedirs(path.dirname(logging_path), exist_ok=True)

        # Check if a file exists
        file_exists = path.exists(logging_path)

        # Write to CSV file
        with open(logging_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)

            # If the file does not exist, write the header first
            if not file_exists:
                writer.writeheader()

            # Write the data in the order of the table headers (ensure the order is consistent).
            # Filter and organize the data in header order
            row_data = {header: messages.get(header, "") for header in headers}
            writer.writerow(row_data)

        # Synchronize all processes to ensure that other processes can continue only after the file has been written.
        accelerator.wait_for_everyone()
