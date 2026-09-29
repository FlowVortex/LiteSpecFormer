import csv
import tempfile
import unittest
from pathlib import Path

import torch

import _support  # noqa: F401  # puts src/ on sys.path

from litespecformer.utils import (
    InstanceNorm,
    Patch,
    _check_quantiles,
    calculate_batch_autocorrelation_function,
    convert_tensor_input_to_list_of_dicts_input,
    count_parameters,
    cyclic_sample_generator,
    get_num_output_patches,
    interpolate_quantiles,
    left_pad_and_cat_2D,
    left_pad_and_stack_1D,
    logging_results,
    plot_confidence_prediction,
    validate_and_prepare_single_dict_task,
    weighted_quantile,
)
from litespecformer.config import DatasetMode


class QuantileUtilityTest(unittest.TestCase):
    def test_quantile_bounds_must_sit_inside_the_unit_interval(self):
        _check_quantiles([0.1, 0.5, 0.9], median_quantile_index=1)
        with self.assertRaises(ValueError):
            _check_quantiles([0.9, 0.1])
        with self.assertRaises(AssertionError):
            _check_quantiles([0.1, 0.9], median_quantile_index=3)

    def test_interpolation_hits_known_levels_and_the_midpoint(self):
        values = torch.tensor([[1.0, 5.0, 9.0]])
        queried = interpolate_quantiles([0.1, 0.3, 0.5], [0.1, 0.5, 0.9], values)
        self.assertTrue(torch.allclose(queried, torch.tensor([[1.0, 3.0, 5.0]])))

    def test_weighted_quantile_of_equal_weights_matches_the_sample_order(self):
        samples = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        quantiles = weighted_quantile([0.0, 1.0], [1.0, 1.0, 1.0, 1.0], samples)
        self.assertEqual(tuple(quantiles.shape), (1, 2))
        self.assertAlmostEqual(float(quantiles[0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(quantiles[0, -1]), 4.0, places=4)


class SeriesUtilityTest(unittest.TestCase):
    def test_patch_left_pads_to_a_full_patch(self):
        patch = Patch(patch_size=4, patch_stride=4)
        series = torch.arange(10, dtype=torch.float32)
        patched = patch(series)
        self.assertEqual(tuple(patched.shape), (3, 4))
        self.assertTrue(torch.isnan(patched[0, 0]))
        self.assertTrue(torch.equal(patched[-1], torch.tensor([6.0, 7.0, 8.0, 9.0])))

    def test_instance_norm_round_trip(self):
        series = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
        for use_arcsinh in (False, True):
            normalizer = InstanceNorm(use_arcsinh=use_arcsinh)
            normalized, loc_scale = normalizer(series)
            restored = normalizer.inverse(normalized, loc_scale)
            self.assertTrue(torch.allclose(restored, series, atol=1e-5))

    def test_left_padding_aligns_the_latest_observation(self):
        stacked = left_pad_and_stack_1D(
            [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])]
        )
        self.assertTrue(torch.isnan(stacked[0, 0]))
        self.assertEqual(float(stacked[0, -1]), 2.0)
        self.assertEqual(float(stacked[1, -1]), 5.0)

        concatenated = left_pad_and_cat_2D([torch.ones(1, 2), torch.ones(2, 4)])
        self.assertEqual(tuple(concatenated.shape), (3, 4))
        self.assertTrue(torch.isnan(concatenated[0, 0]))

    def test_short_training_series_are_filtered_out(self):
        task = {"target": torch.arange(4, dtype=torch.float32)}
        prepared = validate_and_prepare_single_dict_task(
            task, idx=0, prediction_length=4, min_past=8, mode=DatasetMode.TRAIN
        )
        self.assertIsNone(prepared)
        kept = validate_and_prepare_single_dict_task(
            {"target": torch.arange(20, dtype=torch.float32)},
            idx=0,
            prediction_length=4,
            min_past=8,
            mode=DatasetMode.TEST,
        )
        self.assertEqual(tuple(kept.shape), (20,))

    def test_tensor_input_becomes_one_task_per_series(self):
        tasks = convert_tensor_input_to_list_of_dicts_input(torch.arange(6).view(2, 3))
        self.assertEqual(len(tasks), 2)
        self.assertTrue(torch.equal(tasks[1]["target"], torch.tensor([3, 4, 5])))

    def test_output_patch_count_covers_the_horizon_up_to_the_cap(self):
        self.assertEqual(get_num_output_patches(20, 8, 4), 3)
        self.assertEqual(get_num_output_patches(100, 8, 2), 2)

    def test_autocorrelation_at_lag_zero_is_one(self):
        series = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        acf = calculate_batch_autocorrelation_function(series, max_lag=2)
        self.assertEqual(tuple(acf.shape), (1, 3))
        self.assertAlmostEqual(float(acf[0, 0]), 1.0, places=5)

    def test_parameter_count_includes_only_trainable_weights(self):
        layer = torch.nn.Linear(4, 2, bias=False)
        total, trainable = count_parameters(layer)
        self.assertEqual(total, 8)
        self.assertEqual(trainable, 8)

    def test_cyclic_sampler_wraps_around_the_list(self):
        generator = cyclic_sample_generator([0, 1, 2], sample_size=2)
        self.assertEqual(next(generator), (0, 1))
        self.assertEqual(next(generator), (2, 0))
        self.assertEqual(next(generator), (1, 2))
        with self.assertRaises(ValueError):
            next(cyclic_sample_generator([], sample_size=1))

    def test_logging_appends_a_csv_row(self):
        class _Accelerator:
            is_main_process = True

            def wait_for_everyone(self):
                return None

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            messages = {"mse": 0.5, "dataset": "Madrid"}
            logging_results(_Accelerator(), str(path), list(messages), messages)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["dataset"], "Madrid")
            self.assertEqual(float(rows[0]["mse"]), 0.5)

    def test_confidence_plot_returns_a_figure(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        figure = plot_confidence_prediction(
            context=torch.linspace(0, 1, 8).numpy(),
            target=torch.linspace(1, 2, 4).numpy(),
            prediction=torch.stack(
                [torch.full((4,), 0.5), torch.full((4,), 1.5), torch.full((4,), 2.5)]
            ).numpy(),
        )
        self.assertEqual(len(figure.axes), 1)
        figure.clf()


if __name__ == "__main__":
    unittest.main()
