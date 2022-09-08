import torch
import torch.nn as nn
import warnings
from layers.convolutions import ConvLayer
from layers.pooling import PoolLayer
from layers.activations import ActivationLayer
from layers.upsamples import UpSampleLayer
from layers.dropouts import DropOutLayer
from layers.batchnorms import BatchNormLayer
from layers.fullyconnected import FullyConnected
import re


class ConvBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=3,
                 padding='SAME',
                 stride=1,
                 dilation=1,
                 tensor_type='2d',
                 conv_block='ConvBnActiv',
                 causal=False,
                 batch_norm=True,
                 groups=1,
                 activation='relu',
                 separable=False,
                 pool_stride=None,
                 pool_kernel=None,
                 upsample=None,
                 dropout=0.0,
                 conv_init='kaiming',
                 bn_init='default',
                 in_features=None,
                 out_features=None,
                 fc_init='default'):

        """This class allows chaining of common layers, operations, etc. making it easier and quicker to prototype new network architectures.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input image
        out_channels : int
            Number of channels produced by the convolution
        kernel_size : int|tuple
            Size of the convolving kernel. Default: 3
        padding : string
            Padding method ['Same', 'Valid']. Default: 'SAME'
        stride : int|tuple
            Stride of the convolution. Default: 1
        dilation : int|tuple
            Spacing between kernel elements. Default: 1
        tensor_type : string
            dim of kernel. Default '2d'
        conv_block : string
            String that defines which operations, and their order, a given ConvBlock object implements.
            An arbitrary number of operations is allowed, but it must include at least one convolution. This string
            defines the operations as a concatenated string list (separated by) caps. E.g. ‘ConvBnActiv’ defines a
            convolution => batch norm => activation ConvBlock. Default 'ConvBnActiv'
        causal : bool
            Defines causal convolution. Default: False
        batch_norm : bool
            Flag for batch normalization. Default: True
        groups : int
            Channels to output channels. Default: 1
        activation : string
            Activation function. Default : 'relu'
        separable : bool
            Defines seperable convolution. Default : False
        pool_stride : int
            Stride of the pool. Default: None
        pool_kernel : int
            Kernel size of the pool. Default: None
        upsample : float
            Upsample factor. Default : None
        dropout : float
            Probability of a channel to be zeroed. Default : 0.0
        conv_init : string
            Init method. Default : 'kaiming'
        bn_init : string
            Batch normalization init method. Default : 'default'
        """

        super(ConvBlock, self).__init__()

        if len(conv_block) == 0 or conv_block[0].islower():
            raise ValueError("conv_block should have at least 1 layer, each layer should start with an upper case letter")

        op_list = re.findall('[A-Z][a-z]*', conv_block)

        if op_list.count('Conv') != 1:
            raise ValueError("ConvBlock expects a single convolution layer")

        if not batch_norm and 'Bn' in op_list:
            op_list.remove('Bn')
            warnings.warn("batch_norm is False but Batch normalization layer was defined")

        dim = int(tensor_type[0])

        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size,) * dim

        if type(dilation) is not tuple:
            dilation = (dilation,) * dim

        if type(stride) is not tuple:
            stride = (stride,) * dim

        self.layers = []
        for ii, layer_name in enumerate(op_list):

            if layer_name == 'Conv':
                self.layers.append(ConvLayer(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, padding=padding, stride=stride,
                                             conv_init=conv_init, dilation=dilation, tensor_type=tensor_type,
                                             causal=causal, groups=groups, separable=separable))
                # assumes the stride and kernel are the same size

            elif layer_name in ['Avg', 'Max']:
                self.layers.append(PoolLayer(pool_kernel=pool_kernel, pool_stride=pool_stride, pool_type=layer_name, tensor_type=tensor_type))

            elif layer_name == 'Bn':
                if ii > op_list.index('Conv'):
                    bn_channels = out_channels
                else:
                    bn_channels = in_channels
                self.layers.append(BatchNormLayer(bn_channels, bn_init))

            elif layer_name == 'Activ':
                self.layers.append(ActivationLayer(activation))

            elif layer_name == 'Drop':
                self.layers.append(DropOutLayer(p=dropout))


            elif layer_name == 'Up':
                self.layers.append(UpSampleLayer(scale_factor=upsample))

            elif layer_name == 'Fc':
                self.layers.append(FullyConnected(in_features=in_features*out_channels, out_features=out_features, fc_init=fc_init))

            else:
                raise ValueError("Unrecognised layer {}".format(layer_name))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.layers(x)
        return out
