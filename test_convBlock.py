import pytest
import torch
from .conv_block import ConvBlock


class Test_convBlock:
    def test_no_parameters(self):
        with pytest.raises(TypeError, match="missing 2 required positional arguments"):
            conv_block = ConvBlock()

    def test_no_layer(self):
        with pytest.raises(ValueError, match="at least 1 layer"):
            conv_block = ConvBlock(in_channels=1, out_channels=32, conv_block='')

    def test_lowercase1(self):
        with pytest.raises(ValueError, match="at least 1 layer"):
            conv_block = ConvBlock(in_channels=1, out_channels=32, conv_block='convMax')

    def test_lowercase2(self):
        with pytest.raises(ValueError, match="expects a single convolution layer"):
            conv_block = ConvBlock(in_channels=1, out_channels=32, conv_block='Maxconv')

    def test_two_conv_layers(self):
        with pytest.raises(ValueError, match="expects a single convolution layer"):
            conv_block = ConvBlock(in_channels=1, out_channels=32, conv_block='ConvMaxConv')

    def test_pooling_with_no_kernel(self):
        with pytest.raises(ValueError, match="pool_kernel can not be None for pool layers"):
            conv_block = ConvBlock(in_channels=1, out_channels=32, conv_block='ConvMax')

    def test_pooling_with_no_stride(self):
        conv_block = ConvBlock(in_channels=1, out_channels=32, conv_block='ConvMax', pool_kernel=3)
        assert 2 == len(conv_block.layers)

    def test_up_with_no_factor(self):
        with pytest.raises(ValueError, match="upsample can not be None for Up sample layers"):
            conv_block = ConvBlock(in_channels=1, out_channels=32, conv_block='ConvUp')

    def test_bn_init_wrong_string(self):
        with pytest.raises(ValueError, match="Invalid bn_init"):
            conv_block = ConvBlock(in_channels=1, out_channels=32, conv_block='ConvBn', bn_init='twos')

    def test_wrong_dropout(self):
        with pytest.raises(ValueError, match="dropout probability has to be between 0 and 1"):
            conv_block = ConvBlock(in_channels=1, out_channels=32, conv_block='ConvDrop', dropout=2)

    def test_bad_layer(self):
        with pytest.raises(ValueError, match="Unrecognised layer Bla"):
            conv_block = ConvBlock(in_channels=1, out_channels=32, conv_block='ConvBla')

    def test_all_layers(self):
        conv_block = ConvBlock(in_channels=1, out_channels=32, conv_block='ConvBnMaxAvgDropUpActiv', pool_kernel=2, pool_stride=2, padding='SAME',
                               causal=True, upsample=2, dropout=0.2)
        assert 7 == len(conv_block.layers)

    def test_forward(self):
        torch.manual_seed(1234)
        conv_block = ConvBlock(in_channels=1, out_channels=32, conv_block='MaxConvBnDrop', pool_kernel=2, pool_stride=2, padding='SAME', causal=True)
        input = torch.rand(64, 1, 28, 28) * 20 - 10
        # x = conv_block(input); torch.save(x,'y_test_forward.pt')
        y = torch.load('y_test_forward.pt')
        assert torch.allclose(conv_block(input), y, atol=1e-5), "Wrong convolution result"


if __name__ == '__main__':
    pytest.main()
