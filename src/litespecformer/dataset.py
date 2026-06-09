from typing import (
    Union,
    Tuple,
    List,
    Iterator,
    Mapping,
    Sequence,
    TypeAlias,
    cast,
)

from os import path
import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, IterableDataset

from tqdm import tqdm
from datasets import Dataset as HFDataset

from litespecformer.config import DatasetMode
from litespecformer.utils import (
    left_pad_and_cat_2D,
    validate_and_prepare_single_dict_task,
    convert_list_of_tensors_input_to_list_of_dicts_input,
    convert_tensor_input_to_list_of_dicts_input,
)

TensorOrArray: TypeAlias = torch.Tensor | np.ndarray


class LiteSpecFormerDataset(IterableDataset):
    """
    The training and inference dataset for LiteSpecFormer models for spectrum prediction.

    Parameters
    ----------
    inputs
        Time series data. Must be a list of dictionaries where each dictionary may have the following keys.
        - `target` (required): a 1-d or 2-d `torch.Tensor` or `np.ndarray` of shape (history_length,) or (n_variates, history_length).
        Forecasts will be generated for items in `target`.

        - `past_covariates` (optional): a dict of past-only covariates or past values of known future covariates. The keys of the dict
        must be names of the covariates and values must be 1-d `torch.Tensor` or `np.ndarray` with length equal to the `history_length`
        of `target`.

        - `future_covariates` (optional): a dict of future values of known future covariates. The keys of the dict must be names of the
        covariates and values must be 1-d `torch.Tensor` or `np.ndarray` with length equal to the `prediction_length`. All keys in
        `future_covariates` must be a subset of the keys in `past_covariates`.
        Note: when the mode is set to TRAIN, the values inside `future_covariates` are not technically used for training the model;
        however, this key is used to infer which covariates are known into the future. Therefore, if your task contains known future covariates,
        make sure that this key exists in `inputs`. The values of individual future covariates may be set to `None` or an empty array.

    context_length
        The maximum context length used for training or inference TODO: 注意这里是最大的上下文长度
    prediction_length
        The prediction horizon
    batch_size
        The batch size for training the model. Note that the batch size here means the number of time series, including target(s) and
        covariates, that are input into the model. If your data has multiple target and/or covariates, the effective number of time series
        tasks in a batch will be lower than this value.
    output_patch_size
        The output patch size of the model. This is used to compute the number of patches needed to cover `prediction_length`
    min_past
        The minimum number of time steps the context must have during training. All time series shorter than `min_past + prediction_length`
        are filtered out, by default 1
    mode
        `DatasetMode` governing whether to generate training, validation or test samples, by default "train"
    """

    def __init__(
        self,
        inputs: Sequence[
            Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]
        ],
        context_length: int,
        prediction_length: int,
        batch_size: int,
        output_patch_size: int,
        min_past: int = 1,
        mode: str | DatasetMode = DatasetMode.TRAIN,
    ) -> None:
        super().__init__()
        assert mode in {
            DatasetMode.TRAIN,
            DatasetMode.VALIDATION,
            DatasetMode.TEST,
        }, f"Invalid mode: {mode}"

        # List of data
        self.tasks = LiteSpecFormerDataset._prepare_tasks(
            inputs, prediction_length, min_past, mode
        )
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.num_output_patches = math.ceil(prediction_length / output_patch_size)
        self.min_past = min_past
        self.mode = mode

    @staticmethod
    def _prepare_tasks(
        inputs: Sequence[
            Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]
        ],
        prediction_length: int,
        min_past: int,
        mode: str | DatasetMode,
    ):
        tasks = []

        for idx, raw_task in enumerate(inputs):
            raw_task = cast(
                dict[str, TensorOrArray | Mapping[str, TensorOrArray]], raw_task
            )

            # convert to a format compatible with model's forward
            task = validate_and_prepare_single_dict_task(
                raw_task,
                idx=idx,
                prediction_length=prediction_length,
                min_past=min_past,
                mode=mode,
            )

            if task is None:
                # filter out time series that are too short to provide the required context and prediction length
                continue

            if (
                mode != DatasetMode.TEST
                and task[0].shape[-1] < min_past + prediction_length
            ):
                # filter tasks based on min_past + prediction_length
                continue
            tasks.append(task)

        # Check if the dataset is empty
        if len(tasks) == 0:
            raise ValueError(
                "The dataset is empty after filtering based on the length of the time series (length >= min_past + prediction_length). "
                "Please provide longer time series or reduce `min_past` or `prediction_length`. "
            )

        return tasks

    def _construct_slice(
        self, task_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Load the univariate time series data
        task_past_tensor = self.tasks[task_idx]

        # Backup the input look-back and the predicted values.
        task_past_tensor = task_past_tensor.clone().unsqueeze(
            0
        )  # add a batch dimension for easier slicing, shape: (1, history_length)

        # Get the length of this sample
        full_length = task_past_tensor.shape[-1]

        if self.mode == DatasetMode.TRAIN:
            # slice a random subsequence from the full series
            slice_idx = np.random.randint(
                self.min_past, full_length - self.prediction_length + 1
            )

        elif self.mode == DatasetMode.VALIDATION:
            # slice the last window for validation
            slice_idx = full_length - self.prediction_length

        else:
            # slice the full series for prediction
            slice_idx = full_length

        if slice_idx >= self.context_length:
            # slice series, if it is longer than context_length
            task_context = task_past_tensor[
                :, slice_idx - self.context_length : slice_idx
            ]
        else:
            task_context = task_past_tensor[:, :slice_idx]

        # In the TEST mode, we have no target available and the task_future_covariates can be directly used
        # In the TRAIN and VALIDATION modes, the target and task_future_covariates need to be constructed from
        # the task_context_tensor by slicing the appropriate indices which we do below
        if self.mode in [DatasetMode.TRAIN, DatasetMode.VALIDATION]:
            # the first task_n_targets elements in task_context_tensor are the targets
            task_future_target = task_past_tensor[
                :, slice_idx : slice_idx + self.prediction_length
            ].clone()

        else:
            task_future_target = None

        return task_context, task_future_target

    def _build_batch(
        self, task_indices: list[int]
    ) -> dict[str, torch.Tensor | int | list[tuple[int, int]] | None]:
        """Build a batch from given task indices."""

        # Create the list for the input context time series
        batch_context_tensor_list = []
        # Create the list for the input future time series
        # (targets for TRAIN and VALIDATION, future covariates for TEST)
        batch_future_target_tensor_list = []

        # Select the data from the indices
        for task_idx in task_indices:
            # Create a slice of the task
            task_context, task_future_target = self._construct_slice(task_idx)

            # Append the sliced task to the batch list
            batch_context_tensor_list.append(task_context)
            batch_future_target_tensor_list.append(task_future_target)

        return {
            "context": left_pad_and_cat_2D(batch_context_tensor_list),
            "future_target": (
                None
                if self.mode == DatasetMode.TEST
                else torch.cat(
                    cast(list[torch.Tensor], batch_future_target_tensor_list), dim=0
                )
            ),
            "num_output_patches": self.num_output_patches,
        }

    def _generate_train_batches(
        self,
    ) -> Iterator[dict[str, torch.Tensor | int | list[tuple[int, int]] | None]]:
        """"""
        while True:
            current_batch_size = 0
            task_indices = []

            while current_batch_size < self.batch_size:
                # Randomly select indices from these for model training
                task_idx = np.random.randint(len(self.tasks))
                task_indices.append(task_idx)
                current_batch_size += self.tasks[task_idx][0].shape[0]

            yield self._build_batch(task_indices)

    def _generate_sequential_batches(
        self,
    ) -> Iterator[dict[str, torch.Tensor | int | list[tuple[int, int]] | None]]:
        task_idx = 0
        while task_idx < len(self.tasks):
            current_batch_size = 0
            task_indices = []

            while task_idx < len(self.tasks) and current_batch_size < self.batch_size:
                task_indices.append(task_idx)
                # current_batch_size += self.tasks[task_idx][0].shape[0]
                task_idx += 1

            yield self._build_batch(task_indices)

    def __iter__(
        self,
    ) -> Iterator[dict[str, torch.Tensor | int | list[tuple[int, int]] | None]]:
        """
        Generate batches of data for the Chronos-2 model. In training mode, this iterator is infinite.

        Yields
        ------
        dict
            A dictionary containing:
            - context: torch.Tensor of shape (batch_size, context_length) containing input sequences
            - future_target: torch.Tensor of shape (batch_size, prediction_length) containing future target sequences, None in TEST mode
            - target_idx_ranges: (only in TEST mode) list of tuples indicating the start & end indices of targets in context
        """
        if self.mode == DatasetMode.TRAIN:
            for batch in self._generate_train_batches():
                batch.pop("target_idx_ranges")
                yield batch
        elif self.mode == DatasetMode.VALIDATION:
            for batch in self._generate_sequential_batches():
                batch.pop("target_idx_ranges")
                yield batch
        else:
            # yield from self._generate_sequential_batches()
            for batch in self._generate_sequential_batches():
                batch.pop("future_target")
                yield batch

    @classmethod
    def convert_inputs(
        cls,
        inputs: (
            TensorOrArray
            | Sequence[TensorOrArray]
            | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]]
        ),
        context_length: int,
        prediction_length: int,
        batch_size: int,
        output_patch_size: int,
        min_past: int = 64,
        mode: str | DatasetMode = DatasetMode.TRAIN,
    ) -> "LiteSpecFormerDataset":
        """Convert from different input formats to a LiteSpecFormerDataset."""

        if isinstance(inputs, (torch.Tensor, np.ndarray)):
            inputs = convert_tensor_input_to_list_of_dicts_input(inputs)
        elif isinstance(inputs, HFDataset):
            pass
        else:
            raise ValueError("Unexpected inputs format")

        inputs = cast(list[dict[str, TensorOrArray | dict[str, TensorOrArray]]], inputs)

        return cls(
            inputs,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=output_patch_size,
            min_past=min_past,
            mode=mode,
        )


class LiteSpecFormerTestingDataset(IterableDataset):
    """
    An IterableDataset used to evaluate training-time model performance on
    fixed-length rolling windows.
    This dataset iterates over one or multiple context window lengths and
    yields samples of:
    - context: (num_variables, max_context_length) with left NaN padding
    - future_target: (num_variables, prediction_length)
    Notes
    -----
    - Input `tasks` is expected to be a HuggingFace `datasets.Dataset`-like object
      where each item has a "target" field (a 1D torch Tensor / array-like).
    - Each series is standardized independently: (x - mean) / std.
    - The dataset is "training-phase testing": it produces many overlapping
      windows from the same underlying sequences for quick evaluation.
    """

    def __init__(
        self,
        tasks: Dataset,
        prediction_length: int,
        test_context_lengths: int = 256,
    ) -> None:
        """
        Parameters
        ----------
        tasks
            A dataset containing multiple target series. Each element must provide
            `item["target"]` as a 1D sequence.
        prediction_length
            Number of future time steps to predict for each sample.
        test_context_lengths
            One or multiple context window sizes (in time steps) to evaluate.
            The implementation expects this to be iterable (e.g., list[int]).
        """
        super().__init__()
        # Load and standardize all target series, then stack into a tensor
        # with shape: (num_variables, seq_length).
        self.data = self.load_data(tasks)
        # Number of variables (channels) and total sequence length.
        self.num_variables, self.seq_length = self.data.shape
        self.prediction_length = prediction_length
        self.test_context_lengths = test_context_lengths
        # The largest context length among all evaluation settings.
        self.max_context_length = max(test_context_lengths)
        # Total number of yielded samples per full iteration.
        # For each context_length, we slide a window over time such that:
        # - context uses data up to (t + context_length)
        # - future_target uses the next prediction_length steps
        self.num_samples = sum(
            [
                self.seq_length - 2 * self.prediction_length - context_length + 1
                for context_length in self.test_context_lengths
            ]
        )

    def left_padding(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Left-pad a context tensor to `max_context_length` using NaNs.
        This ensures that all contexts returned by the dataset have the same
        temporal length, which can simplify batching/evaluation even when
        multiple context lengths are tested.
        Parameters
        ----------
        tensor
            Context tensor with shape (num_variables, context_length).
        Returns
        -------
        torch.Tensor
            Padded (or truncated) tensor with shape (num_variables, max_context_length).
            If `context_length` >= `max_context_length`, the most recent
            `max_context_length` steps are kept.
        """
        context_length = tensor.shape[-1]
        # If already long enough, keep only the most recent max_context_length steps.
        if context_length >= self.max_context_length:
            return tensor[:, -self.max_context_length :]
        else:
            # Pad on the left with NaNs so the right side aligns as the "latest" context.
            padding = torch.full(
                (self.num_variables, self.max_context_length - context_length),
                fill_value=torch.nan,
                device=tensor.device,
            )
            return torch.cat([padding, tensor], dim=-1)

    def load_data(self, dataset) -> torch.FloatTensor:
        """
        Load and standardize target series from the given dataset.
        The dataset is expected to contain multiple samples, each with a "target"
        field. Each target series is standardized independently (per series).
        Parameters
        ----------
        dataset
            A dataset-like object supporting `len(dataset)` and `dataset[i]`.
            Each `dataset[i]` must be a mapping that contains key "target".
        Returns
        -------
        torch.FloatTensor
            Stacked standardized series with shape (num_variables, seq_length).
        """
        data_list = []
        # Iterate over each sample (variable) and collect standardized targets.
        for i in range(len(dataset)):
            series = dataset[i]["target"]
            data_list.append(((series - series.mean()) / series.std()).unsqueeze(0))
        # Concatenate into (num_variables, seq_length).
        return torch.concatenate(data_list, dim=0).float()

    def __len__(self) -> int:
        """
        Return the number of samples produced in one full iteration.
        """
        return self.num_samples

    def __iter__(self) -> Iterator:
        """
        Yield evaluation samples for each configured context length.
        For each `context_length`, the iterator slides a window over time and yields:
        - "context": standardized past observations, left-padded to max_context_length
        - "future_target": the corresponding future segment of length prediction_length
        Yields
        ------
        dict
            {
              "context": Tensor[num_variables, max_context_length],
              "future_target": Tensor[num_variables, prediction_length],
            }
        """
        for context_length in self.test_context_lengths:
            # Split into:
            # - data_x: everything except the last prediction_length steps
            # - data_y: aligned future targets starting at `context_length`
            data_x, data_y = (
                self.data[:, : -self.prediction_length],
                self.data[:, context_length:],
            )
            # Number of sliding windows for this context length.
            num_samples = data_x.shape[-1] - context_length - self.prediction_length + 1
            for idx in range(0, num_samples):
                yield {
                    "context": self.left_padding(data_x[:, idx : idx + context_length]),
                    "future_target": data_y[:, idx : idx + self.prediction_length],
                }


class SpectrumLibraryDataset(Dataset):
    """
    PyTorch ``Dataset`` for multivariate spectrum (time series) forecasting.

    Loads data from a ``.csv``, ``.npy``, or on-disk Hugging Face dataset, optionally
    applies per-feature standardization, and exposes sliding-window samples for training
    or testing. Each item returns a context window of ``seq_length`` steps and a future
    target of ``prediction_length`` steps.

    The timeline is split once by ``split_ratio``: the training segment uses the leading
    portion of the series; the test segment uses the trailing portion with enough history
    retained to form valid context windows.

    Parameters
    ----------
    seq_length
        Number of past time steps in each input window.
    prediction_length
        Number of future time steps to predict per sample.
    dataset_name_or_path
        Path to a ``.csv`` / ``.npy`` file or a Hugging Face dataset directory. See
        :meth:`load_data` for supported formats.
    split_ratio
        Fraction of the timeline used for training (must be in ``(0, 1)``). When
        ``flag="test"``, samples are drawn from the complementary tail segment.
    standard_scale
        If ``True``, fit :class:`~sklearn.preprocessing.StandardScaler` on the full loaded
        series before splitting.
    flag
        ``"train"`` or ``"test"``. Selects which temporal segment this dataset exposes.

    Attributes
    ----------
    data_x
        Input sequences used to build context windows, shape ``(T, num_variables)``.
    data_y
        Target sequences aligned with ``data_x`` for sliding-window forecasting, same shape.
    num_variables
        Number of variates (channels) in the loaded series.
    """

    def __init__(
        self,
        seq_length: int,
        prediction_length: int,
        dataset_name_or_path: str,
        split_ratio: float = 0.6,
        standard_scale: bool = True,
        flag: str = "train",
    ) -> None:
        super().__init__()

        self.seq_length, self.prediction_length = seq_length, prediction_length

        assert flag in ["train", "test"]
        self.flag = flag

        self.standard_scale = standard_scale
        self.scaler = StandardScaler()

        self.dataset_name_or_path = dataset_name_or_path

        assert 0 < split_ratio < 1, "split_ratio must be between 0 and 1"
        self.split_ratio = split_ratio

        self.data, self.data_x, self.data_y = self.preprocessing(
            self.dataset_name_or_path
        )

        self.data_x, self.data_y = (
            torch.from_numpy(self.data_x).float(),
            torch.from_numpy(self.data_y).float(),
        )

        self._num_variables = self.data_x.shape[1]

    def load_data(self, dataset_name_or_path: str) -> np.ndarray:
        """
        Load the data from the specified path.
        The data can be in the form of a .npy file, a .csv file, or a Hugging Face Dataset.

        Parameters
        ----------
        dataset_name_or_path
            The path to the data file or directory.
            If the path is a file, it can be either a .npy file or a .csv file.
            If the path is a directory, it is assumed to be a Hugging Face Dataset.

        Returns
        -------
        np.ndarray
            A numpy array containing the loaded data. The shape of the array should be (seq_length, num_variables).
        """
        if not path.exists(dataset_name_or_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_name_or_path}")

        if dataset_name_or_path.endswith(".npy"):
            # The shape of data should be (seq_length, num_variables)
            data = np.load(dataset_name_or_path)

        elif dataset_name_or_path.endswith(".csv"):
            # Remove the first column which is the index column
            data = pd.read_csv(dataset_name_or_path).iloc[:, 1:].values

        elif path.isdir(dataset_name_or_path):
            # If the data path is a directory,
            # we assume it is a Hugging Face Dataset and try to load it using the datasets library
            hf_dataset = HFDataset.load_from_disk(
                dataset_path=dataset_name_or_path, keep_in_memory=True
            )
            hf_dataset.set_format("numpy")

            # Get the data from the Hugging Face Dataset, we assume that the target series is stored under the key "target"
            data = np.stack(
                [hf_dataset[i]["target"] for i in range(len(hf_dataset))], axis=0
            ).transpose(
                1, 0
            )  # shape: (seq_length, num_variables)

        else:
            raise ValueError(
                f"Unsupported data format for file: {dataset_name_or_path}"
            )

        return data

    def preprocessing(
        self, dataset_name_or_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load, scale, split, and align input/target arrays for sliding-window sampling.

        After optional standardization, the series is partitioned by ``split_ratio`` and
        ``flag``:

        - ``flag="train"``: use ``data[:split_index + prediction_length]``
        - ``flag="test"``: use ``data[split_index - seq_length:]`` so the first test
          window still has a full context

        ``data_x`` drops the last ``prediction_length`` steps; ``data_y`` is shifted by
        ``seq_length`` so that ``__getitem__`` can pair context and future targets.

        Parameters
        ----------
        dataset_name_or_path
            Path passed to :meth:`load_data`.

        Returns
        -------
        data
            The segment selected for this split, shape ``(T, num_variables)``.
        data_x
            Context source array, shape ``(T - prediction_length, num_variables)``.
        data_y
            Target source array aligned for forecasting, same shape as ``data_x``.
        """

        # Load the data from the dataset_name_or_path
        data = self.load_data(dataset_name_or_path=dataset_name_or_path)

        # Scale the data if required
        if self.standard_scale:
            self.scaler.fit(data)
            data = self.scaler.transform(data)

        # Get the length of the data
        length = len(data)

        # Split the data into train and test sets
        split_index = int(length * self.split_ratio)

        # get the data for the given flag
        if self.flag == "train":
            data = data[: split_index + self.prediction_length]
        else:
            data = data[split_index - self.seq_length :]

        # split the data into input and target
        data_x = data[0 : -self.prediction_length]
        data_y = data[self.seq_length :]

        return data, data_x, data_y

    @property
    def num_variables(self) -> int:
        """Get the number of variables in the dataset."""
        return self._num_variables

    def __len__(self) -> int:
        """
        Number of valid sliding windows in the current split.

        Returns
        -------
        int
            ``len(data_x) - seq_length - prediction_length + 1``.
        """
        return len(self.data_x) - self.seq_length - self.prediction_length + 1

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return one (context, target) pair at the given window index.

        Parameters
        ----------
        index
            Starting index of the sliding window along the time axis.

        Returns
        -------
        context
            Input window with shape ``(seq_length, num_variables)``.
        target
            Future window with shape ``(prediction_length, num_variables)``.
        """
