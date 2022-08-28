import torch.nn as nn


class PoolLayer(nn.Module):
    """
    Pooling layer abstraction. Input is assumed to be in canonical form: [Batch_size, channels, frames, features] for the 2d case.
    """

    def __init__(self, pool_kernel=2, pool_stride=2, pool_type='Max', tensor_type='2d'):
        super(PoolLayer, self).__init__()

        if pool_kernel is None:
            raise ValueError("pool_kernel can not be None for pool layers")

        if tensor_type == '2d':
            if pool_type == 'Avg':
                self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)

            elif pool_type == 'Max':
                self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, input):
        output = self.pool(input)
        return output
