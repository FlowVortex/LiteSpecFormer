import unittest

import torch

import _support  # noqa: F401  # puts src/ on sys.path

from litespecformer.metrics import (
    DEFAULT_QUANTILE_LEVELS,
    calculate_metrics,
    continuous_ranked_probability_score,
    mean_absolute_error,
    mean_absolute_scaled_error,
    mean_squared_error,
    quantile_loss,
    root_mean_squared_error,
)


class PointMetricTest(unittest.TestCase):
    def test_squared_and_absolute_errors(self):
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 0.0, 6.0])
        self.assertAlmostEqual(
            float(mean_squared_error(predictions, targets)), 13.0 / 3.0, places=5
        )
        self.assertAlmostEqual(
            float(mean_absolute_error(predictions, targets)), 5.0 / 3.0, places=5
        )
        self.assertAlmostEqual(
            float(root_mean_squared_error(predictions, targets)) ** 2,
            float(mean_squared_error(predictions, targets)),
            places=5,
        )

    def test_mase_of_a_repeated_naive_forecast_is_one(self):
        contexts = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        targets = torch.tensor([[4.0, 5.0]])
        predictions = torch.tensor([[3.0, 4.0]])
        mase = mean_absolute_scaled_error(contexts, targets, predictions)
        self.assertAlmostEqual(mase, 1.0, places=5)


class CrpsTest(unittest.TestCase):
    def test_default_levels_omit_the_tail_quantiles(self):
        self.assertNotIn(0.05, DEFAULT_QUANTILE_LEVELS)
        self.assertNotIn(0.95, DEFAULT_QUANTILE_LEVELS)
        self.assertEqual(DEFAULT_QUANTILE_LEVELS[0], 0.1)
        self.assertEqual(DEFAULT_QUANTILE_LEVELS[-1], 0.9)

    def test_pinball_term_matches_the_two_sided_definition(self):
        predictions = torch.tensor([2.0, 0.0])
        targets = torch.tensor([0.0, 2.0])
        loss = quantile_loss(predictions, targets, 0.2)
        # Over-prediction contributes (1 - q) * |error|; under-prediction contributes q * |error|.
        expected = 2 * torch.tensor([(1 - 0.2) * 2.0, 0.2 * 2.0])
        self.assertTrue(torch.allclose(loss, expected))

    def test_point_mass_forecast_scores_the_absolute_error(self):
        levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        predictions = torch.full((2, 1, len(levels), 3), 4.0)
        targets = torch.full((2, 1, 3), 1.0)
        score = continuous_ranked_probability_score(predictions, targets, levels)
        self.assertAlmostEqual(float(score), 3.0, places=4)

    def test_perfect_quantile_forecast_scores_zero(self):
        levels = DEFAULT_QUANTILE_LEVELS
        targets = torch.randn(2, 3, 5)
        predictions = targets.unsqueeze(-2).expand(-1, -1, len(levels), -1).contiguous()
        score = continuous_ranked_probability_score(predictions, targets, levels)
        self.assertAlmostEqual(float(score), 0.0, places=5)

    def test_tail_quantiles_do_not_change_the_score(self):
        full_levels = [0.05, 0.1, 0.5, 0.9, 0.95]
        targets = torch.ones(1, 1, 4)
        predictions = torch.full((1, 1, len(full_levels), 4), 5.0)
        predictions[:, :, 1:-1, :] = 1.0
        contexts = torch.randn(1, 1, 16)
        metrics = calculate_metrics(
            contexts, predictions, targets, medium_index=2, quantile_levels=full_levels
        )
        self.assertAlmostEqual(float(metrics["crps"]), 0.0, places=5)
        self.assertAlmostEqual(float(metrics["mae"]), 0.0, places=5)

    def test_calculate_metrics_reports_every_score(self):
        levels = [0.1, 0.5, 0.9]
        contexts = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
        targets = torch.ones(1, 1, 2)
        predictions = torch.zeros(1, 1, len(levels), 2)
        predictions[:, :, 1, :] = 1.0
        metrics = calculate_metrics(
            contexts, predictions, targets, medium_index=1, quantile_levels=levels
        )
        self.assertEqual(
            set(metrics),
            {"mse", "mae", "rmse", "mape", "mase", "mspe", "rse", "crps"},
        )
        self.assertAlmostEqual(float(metrics["mae"]), 0.0, places=5)
        self.assertGreater(float(metrics["crps"]), 0.0)

    def test_mismatched_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            calculate_metrics(
                contexts=torch.zeros(2, 1, 8),
                predictions=torch.zeros(1, 1, 3, 2),
                targets=torch.zeros(2, 1, 2),
            )


if __name__ == "__main__":
    unittest.main()
