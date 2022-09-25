import torch
from torch import nn


class EstimationNetwork(nn.Module):
    """Defines a estimation network."""
    def __init__(self, dim_embed, num_mixtures):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_embed, 10),
                                 nn.Tanh(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(10, num_mixtures),
                                 nn.Softmax(dim=1))

    def forward(self, input):
        return self.net(input)