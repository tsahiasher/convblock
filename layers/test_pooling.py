import pytest
import torch
from .pooling import PoolLayer


class Test_pooling:
    def test_forward1(self):
        input = torch.zeros((1,5,5))
        input[0,1:4, 1:4] = 18
        pool_layer = PoolLayer(pool_kernel=3, pool_stride=1, pool_type='Max')
        y = pool_layer(input)
        assert torch.equal(y, torch.full((1,3,3),18.0))

    def test_forward2(self):
        input = torch.zeros((1, 5, 5))
        input[0, 1:4, 1:4] = 18
        pool_layer = PoolLayer(pool_kernel=3, pool_stride=1, pool_type='Avg')
        y = pool_layer(input)
        assert torch.equal(y, torch.Tensor([[[8.0,12.0,8.0],[12.0,18.0,12.0],[8.0,12.0,8.0]]]))
