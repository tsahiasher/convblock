import torch.nn as nn


class UpSampleLayer(nn.Module):

    def __init__(self, scale_factor=None):
        super(UpSampleLayer, self).__init__()

        if scale_factor is None:
            raise ValueError("upsample can not be None for Up sample layers")
        self.upsample = nn.Upsample(scale_factor=scale_factor)

    def forward(self, input):
        output = self.upsample(input)
        return output
