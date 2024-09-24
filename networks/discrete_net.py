import torch
import torch.nn as nn


class DiscreteObsNet(nn.Module):
    """A simple and flexible observation network for a Discrete observation spaces."""

    def __init__(self, hidden_sizes: list[int], o_dim: int):
        super().__init__()
        self.o_dim = o_dim

        layers = []
        in_size = 1
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, self.o_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = x["obs"]
        x = torch.as_tensor(x, dtype=torch.float32)

        if x.dim() == 0:
            # we get a single scalar input
            x = x.view(1, 1)
        elif x.dim() == 1:
            # we get a batch as input
            x = x.view(-1, 1)

        return self.fc(x)
