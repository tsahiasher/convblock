import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Conv layer abstraction. Input is assumed to be in canonical form: [Batch_size, channels, frames, features] for the 2d case.
    Causal convolution implemented according to:
    from https://gist.github.com/paultsw/7a9d6e3ce7b70e9e2c61bc9287addefc
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding='SAME', stride=1,
                 conv_init='kaiming', dilation=1, tensor_type='2d', causal=False, groups=1, separable=False):
        super(ConvLayer, self).__init__()
        dim = int(tensor_type[0])
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size,) * dim

        if type(dilation) is not tuple:
            dilation = (dilation,) * dim

        if type(stride) is not tuple:
            stride = (stride,) * dim

        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding.lower() not in valid_padding_strings:
                raise ValueError("Invalid padding string {!r}, should be one of {}".format(padding, valid_padding_strings))
            if padding.lower() == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        self.dilation = dilation
        self.causal = causal
        self.kernel_size = kernel_size
        self.conv_init = conv_init
        self.padding = ()
        # 2D convolution
        if tensor_type == '2d':
            # Padding
            if not causal:
                if padding.lower() == 'same':
                    for ii in range(len(kernel_size)):
                        self.padding += (((dilation[ii] * (kernel_size[ii] - 1)) // 2),)
                elif padding.lower() == 'valid':
                    self.padding = (0,) * len(kernel_size)
            else:
                for ii in range(len(kernel_size)):
                    self.padding += ((dilation[ii] * (kernel_size[ii] - 1)),)

            # Full 2d conv
            if separable is False:
                self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                            padding=self.padding, dilation=dilation, stride=stride, groups=groups)
                self.weight_init(self.conv_layer)
            # Separable conv
            else:
                conv_layer = []
                conv_layer.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                            padding=self.padding, dilation=dilation, stride=stride, groups=in_channels))
                self.weight_init(conv_layer[-1])
                conv_layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                            padding=0, dilation=1, stride=1))
                self.weight_init(conv_layer[-1])

                self.conv_layer = nn.Sequential(*conv_layer)

    def weight_init(self, conv_layer):
        if self.conv_init == 'kaiming':
            nn.init.kaiming_normal_(conv_layer.weight)
            nn.init.constant_(conv_layer.bias, 0.)
        elif self.conv_init == 'xavier':
            nn.init.xavier_normal_(conv_layer.weight)
            nn.init.constant_(conv_layer.bias, 0.)
        elif self.conv_init == 'zeros':
            nn.init.constant_(conv_layer.weight, 0.)
            nn.init.constant_(conv_layer.bias, 0.)
        else:
            raise ValueError(f"Invalid conv_init {self.conv_init}, should be one of 'kaiming', 'xavier', 'zeros'")

    def forward(self, input):
        output = self.conv_layer(input)
        if self.causal:
            output = output[..., 0:-self.padding[0], 0:-self.padding[1]]

        return output
