import unittest

import torch

from _support import tiny_config

from litespecformer.layers import (
    ConvNet,
    DepthWiseConv,
    FeedForward,
    RMSNorm,
    ResidualBlock,
    RoPE,
    SEAttention,
    Transpose,
)


class NormalizationAndPositionTest(unittest.TestCase):
    def test_rmsnorm_keeps_the_input_shape(self):
        layer = RMSNorm(hidden_size=4, eps=1e-6)
        output = layer(torch.ones(2, 3, 4))
        self.assertEqual(tuple(output.shape), (2, 3, 4))
        expected = 1.0 / torch.sqrt(torch.tensor(1.0 + 1e-6))
        self.assertTrue(torch.allclose(output, expected.expand_as(output), atol=1e-5))

    def test_rope_embeddings_match_the_head_dimension(self):
        rope = RoPE(dim=8, base=10000)
        hidden = torch.zeros(2, 4, 8)
        positions = torch.arange(4).expand(2, -1)
        cos, sin = rope(hidden, positions)
        self.assertEqual(tuple(cos.shape), (2, 4, 8))
        self.assertEqual(tuple(sin.shape), (2, 4, 8))
        rotated = RoPE.rotate_half(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.assertTrue(torch.equal(rotated, torch.tensor([-3.0, -4.0, 1.0, 2.0])))


class ConvolutionAndResidualTest(unittest.TestCase):
    def test_transpose_swaps_the_requested_axes(self):
        layer = Transpose(1, 2)
        output = layer(torch.zeros(2, 3, 5))
        self.assertEqual(tuple(output.shape), (2, 5, 3))

    def test_depthwise_convolution_preserves_time(self):
        layer = DepthWiseConv(in_channels=4, out_channels=6, kernel_size=3, padding=1)
        output = layer(torch.randn(2, 4, 10))
        self.assertEqual(tuple(output.shape), (2, 6, 10))

    def test_squeeze_excitation_reweights_channels(self):
        layer = SEAttention(n_dims=1, n_channels=8, reduction=4)
        output = layer(torch.randn(2, 8, 12))
        self.assertEqual(tuple(output.shape), (2, 8, 12))

    def test_residual_block_projects_to_the_output_width(self):
        block = ResidualBlock(
            in_dim=6, h_dim=8, out_dim=4, act_fn_name="gelu", use_layer_norm=True
        )
        output = block(torch.randn(2, 5, 6))
        self.assertEqual(tuple(output.shape), (2, 5, 4))

    def test_feed_forward_is_residual_around_the_model_width(self):
        config = tiny_config()
        layer = FeedForward(config)
        hidden = torch.randn(2, 5, config.d_model)
        output = layer(hidden)
        self.assertEqual(tuple(output.shape), tuple(hidden.shape))

        conv = ConvNet(config)
        projected = conv(hidden)
        self.assertEqual(tuple(projected.shape), tuple(hidden.shape))


if __name__ == "__main__":
    unittest.main()
