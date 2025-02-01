from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from .utils import reparameterise


class DiscreteVAE(nn.Module):
    """A simple variational autencoder for environments with Discrete observation spaces."""

    def __init__(
        self,
        observation_space: gym.Space,
        hidden_sizes: list[int],
        latent_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.encoder = DiscreteEncoder(
            observation_space=observation_space,
            hidden_sizes=hidden_sizes,
            latent_dim=latent_dim,
            device=device,
        )
        self.decoder = DiscreteDecoder(
            latent_dim=latent_dim,
            hidden_sizes=hidden_sizes,
            output_dim=observation_space["obs"].n,
            device=device,
        )

    def decode(self, z: torch.Tensor, is_dream: bool = False):
        """Decodes the latent vector into an observation compatible with the ones provided by Discrete environments wrapped with DictObservation."""
        recon_logits = self.decoder(z)
        probs = F.softmax(recon_logits, dim=-1)
        obs = torch.argmax(probs, dim=-1).item()
        return {"obs": obs}

    def forward(self, inputs: Dict[str, np.ndarray]):
        _, z, dist = self.encoder(inputs)
        recon = self.decoder(z)
        return recon, z, dist


class DiscreteEncoder(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        hidden_sizes: list[int],
        latent_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

        layers = []
        in_size = 1  # input is scalar due to Discrete observation space
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        self.encoder = nn.Sequential(*layers).to(device)

        self.latent_out = nn.Linear(in_size, self.latent_dim).to(device)
        self.fc_mu = nn.Linear(in_size, self.latent_dim).to(device)
        self.fc_logsigma = nn.Linear(in_size, self.latent_dim).to(device)

    def forward(self, inputs: Dict[str, np.ndarray]):
        x = inputs["obs"]
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() == 1:
            x = x.view(-1, 1)

        h = self.encoder(x)
        latent_obs = self.latent_out(h)
        mu = self.fc_mu(h)
        logsigma = self.fc_logsigma(h)
        z, dist = reparameterise(mu, logsigma)
        return latent_obs, z, dist


class DiscreteDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_sizes: list[int],
        output_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device

        layers = []
        in_size = latent_dim
        for hidden_size in reversed(hidden_sizes):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_dim))
        self.decoder = nn.Sequential(*layers).to(device)

    def forward(self, z: torch.Tensor):
        recon = self.decoder(z)
        return recon
