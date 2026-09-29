import unittest

import torch

from _support import tiny_model

from litespecformer.pipeline import ForecastType, LiteSpecFormerPipeline


class LiteSpecFormerPipelineTest(unittest.TestCase):
    def setUp(self):
        self.pipeline = LiteSpecFormerPipeline(model=tiny_model())

    def test_forecast_metadata_comes_from_the_model(self):
        self.assertEqual(self.pipeline.forecast_type, ForecastType.QUANTILES)
        self.assertEqual(self.pipeline.model_context_length, 32)
        self.assertEqual(self.pipeline.quantiles, [0.1, 0.5, 0.9])
        self.assertEqual(self.pipeline.model_output_patch_size, 8)

    def test_probability_masses_sum_to_one(self):
        levels = torch.tensor([0.1, 0.5, 0.9])
        masses = LiteSpecFormerPipeline._get_prob_mass_per_quantile_level(levels)
        self.assertEqual(tuple(masses.shape), (3,))
        self.assertAlmostEqual(float(masses.sum()), 1.0, places=5)
        self.assertTrue(torch.all(masses > 0))

    def test_calculate_metrics_uses_the_median_and_reports_crps(self):
        contexts = torch.arange(16, dtype=torch.float32).view(1, 1, 16)
        targets = torch.ones(1, 1, 4)
        predictions = torch.zeros(1, 1, 3, 4)
        predictions[:, :, 1, :] = 1.0
        metrics = self.pipeline.calculate_metrics(contexts, targets, predictions)
        self.assertAlmostEqual(float(metrics["mae"]), 0.0, places=5)
        self.assertAlmostEqual(float(metrics["mse"]), 0.0, places=5)
        self.assertIn("crps", metrics)
        self.assertGreater(float(metrics["crps"]), 0.0)

    def test_numpy_inputs_are_accepted(self):
        contexts = torch.randn(1, 1, 16).numpy()
        targets = torch.randn(1, 1, 4).numpy()
        predictions = torch.randn(1, 1, 3, 4).numpy()
        metrics = self.pipeline.calculate_metrics(contexts, targets, predictions)
        self.assertTrue(
            all(
                torch.is_tensor(value) or isinstance(value, float)
                for value in metrics.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
