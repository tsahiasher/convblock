import torch.nn as nn


class DropOutLayer(nn.Module):
    """
    dropout layer abstraction. Input is assumed to be in canonical form: [Batch_size, channels, frames, features] for the 2d case.
    """

    def __init__(self, p=0.0):
        super(DropOutLayer, self).__init__()

        if p < 0.0 or p > 1.0:
            raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
        self.dropout = nn.Dropout2d(p=p)

    def forward(self, input):
        output = self.dropout(input)
        return output
