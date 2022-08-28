import torch.nn as nn


class ActivationLayer(nn.Module):
    """
    Activation layer abstraction. Input is assumed to be in canonical form: [Batch_size, channels, frames, features] for the 2d case.
    """

    def __init__(self, act_type="relu"):
        super(ActivationLayer, self).__init__()

        if act_type == "relu":
            self.active = nn.ReLU()
        elif act_type == "relu6":
            self.active = nn.ReLU6()
        elif act_type == "lrelu":
            self.active = nn.LeakyReLU()
        elif act_type == "tanh":
            self.active = nn.Tanh()
        elif act_type == "sigmoid":
            self.active = nn.Sigmoid()
        elif act_type == "elu":
            self.active = nn.ELU()

    def forward(self, input):
        output = self.active(input)
        return output
