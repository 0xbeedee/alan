from typing import Tuple, Self

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """VAE Encoder using Linear layers."""

    def __init__(
        self,
        *,
        input_size: int,
        latent_size: int,
        device: torch.device = torch.device("cpu")
    ):
        super(Encoder, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_size, 128).to(self.device)
        self.fc2 = nn.Linear(128, 64).to(self.device)
        self.fc_mu = nn.Linear(64, latent_size).to(self.device)
        self.fc_logsigma = nn.Linear(64, latent_size).to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)
        return mu, logsigma

    def to(self, device: torch.device) -> Self:
        self.device = device
        self.fc1.to(device)
        self.fc2.to(device)
        self.fc_mu.to(device)
        self.fc_logsigma.to(device)
        return super().to(device)


class Decoder(nn.Module):
    """VAE Decoder using Linear layers."""

    def __init__(
        self,
        *,
        input_size: int,
        latent_size: int,
        device: torch.device = torch.device("cpu")
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(latent_size, 64).to(self.device)
        self.fc2 = nn.Linear(64, 128).to(self.device)
        self.fc3 = nn.Linear(128, input_size).to(self.device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Ensure input is on the correct device
        z = z.to(self.device)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        reconstruction = torch.sigmoid(self.fc3(z))
        return reconstruction

    def to(self, device: torch.device) -> Self:
        self.device = device
        self.fc1.to(device)
        self.fc2.to(device)
        self.fc3.to(device)
        return super().to(device)


class VAE(nn.Module):
    """Variational Autoencoder (VAE) Network."""

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        device: torch.device = torch.device("cpu"),
    ):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = Encoder(
            input_size=input_size, latent_size=latent_size, device=self.device
        )
        self.decoder = Decoder(
            input_size=input_size, latent_size=latent_size, device=self.device
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        mu, logsigma = self.encoder(x)
        sigma = torch.exp(logsigma)
        # Sample epsilon on the correct device
        eps = torch.randn_like(sigma, device=self.device)
        z = mu + sigma * eps  # Reparameterization trick
        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
