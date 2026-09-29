import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import _support  # noqa: F401  # puts src/ on sys.path

from litespecformer.config import DatasetMode
from litespecformer.dataset import (
    LiteSpecFormerDataset,
    LiteSpecFormerTestingDataset,
    SpectrumLibraryDataset,
)


def _write_series(directory: Path, length: int = 80, variables: int = 3) -> Path:
    series = np.arange(length * variables, dtype=np.float32).reshape(length, variables)
    path = directory / "series.npy"
    np.save(path, series)
    return path


class SpectrumLibraryDatasetTest(unittest.TestCase):
    def test_sliding_window_pairs_context_with_the_following_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_series(Path(directory))
            dataset = SpectrumLibraryDataset(
                seq_length=16,
                prediction_length=4,
                dataset_name_or_path=str(path),
                split_ratio=0.6,
                standard_scale=False,
                flag="train",
            )
            context, target = dataset[0]
            self.assertEqual(tuple(context.shape), (16, 3))
            self.assertEqual(tuple(target.shape), (4, 3))
            self.assertTrue(torch.equal(dataset.data_x[16:20], dataset.data_y[0:4]))
            self.assertEqual(
                len(dataset),
                len(dataset.data_x)
                - dataset.seq_length
                - dataset.prediction_length
                + 1,
            )

            last_context, last_target = dataset[len(dataset) - 1]
            self.assertEqual(tuple(last_context.shape), (16, 3))
            self.assertEqual(tuple(last_target.shape), (4, 3))

            batch_context, batch_target = next(iter(DataLoader(dataset, batch_size=2)))
            self.assertEqual(tuple(batch_context.shape), (2, 16, 3))
            self.assertEqual(tuple(batch_target.shape), (2, 4, 3))

    def test_csv_drops_the_index_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "series.csv"
            path.write_text(
                "index,a,b\n0,1,2\n1,3,4\n2,5,6\n3,7,8\n4,9,10\n5,11,12\n",
                encoding="utf-8",
            )
            dataset = SpectrumLibraryDataset(
                seq_length=2,
                prediction_length=1,
                dataset_name_or_path=str(path),
                split_ratio=0.5,
                standard_scale=False,
                flag="test",
            )
            self.assertEqual(dataset.num_variables, 2)

    def test_missing_file_and_unknown_flag_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            SpectrumLibraryDataset(8, 2, "/tmp/missing-litespecformer-series.npy")
        with tempfile.TemporaryDirectory() as directory:
            path = _write_series(Path(directory), length=40)
            with self.assertRaises(AssertionError):
                SpectrumLibraryDataset(8, 2, str(path), flag="validation")


class TaskDatasetTest(unittest.TestCase):
    def test_training_batch_contains_context_and_future_target(self):
        series = torch.arange(48, dtype=torch.float32).view(1, 48)
        dataset = LiteSpecFormerDataset(
            [{"target": series}],
            context_length=16,
            prediction_length=8,
            batch_size=1,
            output_patch_size=8,
            min_past=8,
            mode=DatasetMode.TRAIN,
        )
        self.assertEqual(len(dataset.tasks), 1)
        self.assertEqual(dataset.tasks[0].shape[-1], 48)
        self.assertEqual(dataset.num_output_patches, 1)

    def test_empty_dataset_after_length_filtering_is_rejected(self):
        with self.assertRaises(ValueError):
            LiteSpecFormerDataset(
                [{"target": torch.arange(4, dtype=torch.float32)}],
                context_length=8,
                prediction_length=4,
                batch_size=1,
                output_patch_size=4,
                min_past=8,
                mode=DatasetMode.TRAIN,
            )

    def test_testing_dataset_pads_context_to_the_longest_window(self):
        class _Tasks:
            def __len__(self):
                return 2

            def __getitem__(self, index):
                start = float(index + 1)
                return {"target": torch.arange(40, dtype=torch.float32) + start}

        dataset = LiteSpecFormerTestingDataset(
            _Tasks(), prediction_length=4, test_context_lengths=[16]
        )
        self.assertEqual(len(dataset), 40 - 2 * 4 - 16 + 1)
        sample = next(iter(dataset))
        self.assertEqual(tuple(sample["context"].shape), (2, 16))
        self.assertEqual(tuple(sample["future_target"].shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
