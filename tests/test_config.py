import unittest

from _support import tiny_config

from litespecformer.config import (
    DatasetMode,
    LiteSpecFormerConfig,
    LiteSpecFormerForecastingConfig,
    LiteSpecFormerTrainingConfig,
)


class DatasetModeTest(unittest.TestCase):
    def test_split_names(self):
        self.assertEqual(DatasetMode.TRAIN.value, "train")
        self.assertEqual(DatasetMode.VALIDATION.value, "validation")
        self.assertEqual(DatasetMode.TEST.value, "test")


class ForecastingConfigTest(unittest.TestCase):
    def test_default_quantiles_place_the_median_at_index_five(self):
        config = LiteSpecFormerForecastingConfig()
        self.assertEqual(config.quantiles[0], 0.05)
        self.assertEqual(config.quantiles[-1], 0.95)
        self.assertEqual(config.quantiles[config.medium_index], 0.5)
        self.assertEqual(config.medium_index, 5)

    def test_custom_quantiles_without_one_half_use_the_middle_index(self):
        config = LiteSpecFormerForecastingConfig(quantiles=[0.2, 0.4, 0.8])
        self.assertEqual(config.medium_index, 1)
        self.assertEqual(config.context_length, 768)
        self.assertEqual(config.time_encoding_scale, config.context_length)


class ModelConfigTest(unittest.TestCase):
    def test_defaults_and_attribute_map(self):
        config = LiteSpecFormerConfig()
        self.assertEqual(config.model_type, "time_series_transformer")
        self.assertEqual(config.hidden_size, config.d_model)
        self.assertEqual(config.num_attention_heads, config.num_heads)
        self.assertEqual(config._attn_implementation, "sdpa")
        self.assertIsInstance(
            config.forecasting_config, LiteSpecFormerForecastingConfig
        )

    def test_forecasting_options_are_forwarded(self):
        config = tiny_config(context_length=48)
        self.assertEqual(config.forecasting_config.context_length, 48)
        self.assertEqual(config.d_model, 32)
        self.assertEqual(config.num_layers, 1)

    def test_unsupported_attention_gate_is_rejected(self):
        with self.assertRaises(AssertionError):
            LiteSpecFormerConfig(attn_output_gate="softmax")

    def test_unsupported_attention_implementation_is_rejected(self):
        with self.assertRaises(AssertionError):
            LiteSpecFormerConfig(attn_implementation="flash")


class TrainingConfigTest(unittest.TestCase):
    def test_optimizer_defaults(self):
        config = LiteSpecFormerTrainingConfig()
        self.assertEqual(config.optimizer, "AdamW")
        self.assertEqual(config.anneal_strategy, "cos")
        self.assertGreater(config.num_training_steps, config.num_warmup_steps)
        self.assertEqual(config.eval_context_lengths[0], 128)


if __name__ == "__main__":
    unittest.main()
