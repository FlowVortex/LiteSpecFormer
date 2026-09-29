import unittest

import torch

from _support import tiny_config, tiny_model

from litespecformer.model import LiteSpecFormerModel


class LiteSpecFormerModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = tiny_model()
        self.quantiles = [0.1, 0.5, 0.9]

    def test_construction_records_the_forecasting_quantiles(self):
        self.assertEqual(self.model.num_quantiles, 3)
        self.assertEqual(self.model.medium_index, 1)
        self.assertTrue(
            torch.equal(
                self.model.quantiles.cpu(),
                torch.tensor(self.quantiles),
            )
        )

    def test_patch_sizes_must_match(self):
        config = tiny_config(input_patch_size=8, output_patch_size=4)
        with self.assertRaises(AssertionError):
            LiteSpecFormerModel(config)

    def test_inference_returns_one_patch_of_quantiles(self):
        context = torch.randn(2, 32)
        output = self.model(context)
        self.assertIsNone(output.loss)
        self.assertEqual(tuple(output.quantile_preds.shape), (2, 3, 8))
        self.assertTrue(torch.isfinite(output.quantile_preds).all())

    def test_training_forward_returns_a_finite_loss(self):
        context = torch.randn(2, 32)
        future = torch.randn(2, 8)
        output = self.model(context, future_target=future)
        self.assertEqual(tuple(output.quantile_preds.shape), (2, 3, 8))
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.loss.ndim, 0)


if __name__ == "__main__":
    unittest.main()
