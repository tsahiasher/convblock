import pytest
from .convolutions import ConvLayer
import torch
import numpy as np


class Test_conv:

    def test_no_parameters(self):
        with pytest.raises(TypeError, match="missing 2 required positional arguments"):
            conv_layer = ConvLayer()

    def test_padding_wrong_string(self):
        with pytest.raises(ValueError, match="Invalid padding string"):
            conv_layer = ConvLayer(in_channels=1, out_channels=32, padding='SameSame')

    def test_padding_bad_stride(self):
        with pytest.raises(ValueError, match="padding='same' is not supported for strided convolutions"):
            conv_layer = ConvLayer(in_channels=1, out_channels=32, padding='Same', stride=2)

    def test_weight_init1(self):
        conv_layer = ConvLayer(in_channels=1, out_channels=1, conv_init='zeros')
        assert torch.count_nonzero(conv_layer.conv_layer.weight) == 0
        assert torch.count_nonzero(conv_layer.conv_layer.bias) == 0

    def test_weight_init2(self):
        conv_layer = ConvLayer(in_channels=256, out_channels=512, conv_init='kaiming')
        assert np.mean(conv_layer.conv_layer.weight.detach().numpy()) == pytest.approx(0, abs=1e-4)
        shape = conv_layer.conv_layer.weight.shape
        n = shape[1] * shape[2] * shape[3]
        assert np.std(conv_layer.conv_layer.weight.detach().numpy()) == pytest.approx(np.sqrt(2 / n), abs=1e-4)
        assert torch.count_nonzero(conv_layer.conv_layer.bias) == 0

    def test_weight_init3(self):
        conv_layer = ConvLayer(in_channels=256, out_channels=512, conv_init='xavier')
        assert np.mean(conv_layer.conv_layer.weight.detach().numpy()) == pytest.approx(0, abs=1e-4)
        assert np.std(conv_layer.conv_layer.weight.detach().numpy()) == pytest.approx(0, abs=1e-1)
        assert torch.count_nonzero(conv_layer.conv_layer.bias) == 0

    def test_forward1(self):
        input = torch.rand(64, 1, 28, 28) * 20 - 10
        conv_layer = ConvLayer(in_channels=1, out_channels=32, padding='valid')
        y = conv_layer(input)
        assert torch.Size([64, 32, 26, 26]) == y.shape
        assert conv_layer.padding == (0, 0)

    def test_forward2(self):
        input = torch.rand(64, 1, 28, 28) * 20 - 10
        conv_layer = ConvLayer(in_channels=1, out_channels=32, padding='same', separable=True)
        y = conv_layer(input)
        assert torch.Size([64, 32, 28, 28]) == y.shape
        assert conv_layer.padding == (1, 1)

    def test_forward3(self):
        input = torch.rand(64, 1, 28, 28) * 20 - 10
        conv_layer = ConvLayer(in_channels=1, out_channels=32, padding='same', kernel_size=5)
        y = conv_layer(input)
        assert torch.Size([64, 32, 28, 28]) == y.shape
        assert conv_layer.padding == (2, 2)

    def test_forward4(self):
        input = torch.rand(64, 1, 28, 28) * 20 - 10
        conv_layer = ConvLayer(in_channels=1, out_channels=32, causal=True)
        y = conv_layer(input)
        assert torch.Size([64, 32, 28, 28]) == y.shape
        assert conv_layer.padding == (2, 2)

    def test_forward5(self):
        input = torch.rand(64, 1, 28, 28) * 20 - 10
        conv_layer = ConvLayer(in_channels=1, out_channels=32, padding='same', separable=True)
        y = conv_layer(input)
        assert torch.Size([64, 32, 28, 28]) == y.shape
        assert conv_layer.padding == (1, 1)

    def test_forward6(self):
        input = torch.rand(64, 1, 28, 28) * 20 - 10
        conv_layer = ConvLayer(in_channels=1, out_channels=32, causal=True, dilation=2)
        y = conv_layer(input)
        assert torch.Size([64, 32, 28, 28]) == y.shape
        assert conv_layer.padding == (4, 4)

    def test_forward7(self):
        input = torch.rand(64, 1, 28, 28) * 20 - 10
        conv_layer = ConvLayer(in_channels=1, out_channels=32, padding='same', dilation=2)
        y = conv_layer(input)
        assert torch.Size([64, 32, 28, 28]) == y.shape
        assert conv_layer.padding == (2, 2)

    def test_forward8(self):
        input = torch.rand(64, 1, 28, 28) * 20 - 10
        conv_layer = ConvLayer(in_channels=1, out_channels=32, padding='valid', kernel_size=5, stride=2)
        y = conv_layer(input)
        assert torch.Size([64, 32, 12, 12]) == y.shape
        assert conv_layer.padding == (0, 0)


if __name__ == '__main__':
    pytest.main()
