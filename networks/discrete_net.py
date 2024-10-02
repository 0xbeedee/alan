import torch
import torch.nn as nn
import gymnasium as gym


class DiscreteObsNet(nn.Module):
    """A simple and flexible observation network for Discrete observation spaces with device support."""

    def __init__(
        self,
        # for API compatibility
        observation_space: gym.Space,
        hidden_sizes: list[int],
        o_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.o_dim = o_dim
        self.device = device

        layers = []
        in_size = 1
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, self.o_dim))
        self.fc = nn.Sequential(*layers).to(device)

    def forward(self, x):
        x = x["obs"]
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x.dim() == 0:
            # we get a single scalar input
            x = x.view(1, 1)
        elif x.dim() == 1:
            # we get a batch as input
            x = x.view(-1, 1)
        return self.fc(x)

    def to(self, device):
        self.device = device
        return super().to(device)
