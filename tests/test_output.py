import unittest

import torch

import _support  # noqa: F401  # puts src/ on sys.path

from litespecformer.output import (
    AttentionOutput,
    LiteSpecFormerEncoderBlockOutput,
    LiteSpecFormerEncoderOutput,
    LiteSpecFormerOutput,
)


class OutputContainerTest(unittest.TestCase):
    def test_model_output_keeps_named_fields(self):
        context = torch.zeros(2, 8)
        quantiles = torch.zeros(2, 3, 8)
        output = LiteSpecFormerOutput(
            loss=torch.tensor(0.25),
            context=context,
            quantile_preds=quantiles,
            enc_attn_weights=None,
        )
        self.assertEqual(float(output.loss), 0.25)
        self.assertIs(output.context, context)
        self.assertEqual(tuple(output.quantile_preds.shape), (2, 3, 8))

    def test_attention_outputs_default_to_empty(self):
        attention = AttentionOutput()
        block = LiteSpecFormerEncoderBlockOutput()
        encoder = LiteSpecFormerEncoderOutput()
        self.assertIsNone(attention.hidden_states)
        self.assertIsNone(block.attn_weights)
        self.assertIsNone(encoder.last_hidden_state)
        self.assertIsNone(encoder.all_attn_weights)


if __name__ == "__main__":
    unittest.main()
