import torch.nn as nn


class BatchNormLayer(nn.Module):
    """
    Batch Norm layer abstraction. Input is assumed to be in canonical form: [Batch_size, channels, frames, features] for the 2d case.
    """

    def __init__(self, bn_channels, bn_init='default'):
        super(BatchNormLayer, self).__init__()

        self.batchnorm = nn.BatchNorm2d(bn_channels)
        if bn_init == 'ones':
            nn.init.constant_(self.batchnorm.weight, 1)
            nn.init.constant_(self.batchnorm.bias, 0)
        elif bn_init == 'zeros':
            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            nn.init.constant_(self.batchnorm.weight, 0)
            nn.init.constant_(self.batchnorm.bias, 0)
        elif bn_init != 'default':
            raise ValueError(f"Invalid bn_init {bn_init}, should be one of 'ones', 'zeros'")

    def forward(self, input):
        output = self.batchnorm(input)
        return output
