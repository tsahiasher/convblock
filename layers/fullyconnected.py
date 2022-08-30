import torch.nn as nn


class FullyConnected(nn.Module):
    """
    Fully connected layer abstraction. Input is assumed to be in canonical form: [Batch_size, channels, frames, features] for the 2d case.
    """

    def __init__(self, in_features, out_features, fc_init, bias = True):
        super(FullyConnected, self).__init__()
        if in_features is None or out_features is None:
            raise ValueError("in_features or our_features can not be None for Fully connected layers")
        self.fc_init = fc_init
        self.fc = nn.Linear(in_features = in_features, out_features = out_features, bias=bias)
        self.weight_init()

    def weight_init(self):
        if self.fc_init == 'kaiming':
            nn.init.kaiming_normal_(self.fc.weight)
            nn.init.constant_(self.fc.bias, 0.)
        elif self.fc_init == 'xavier':
            nn.init.xavier_normal_(self.fc.weight)
            nn.init.constant_(self.fc.bias, 0.)
        elif self.fc_init == 'zeros':
            nn.init.constant_(self.fc.weight, 0.)
            nn.init.constant_(self.fc.bias, 0.)

    def forward(self, input):
        return self.fc(input.view(input.size(0), -1))

