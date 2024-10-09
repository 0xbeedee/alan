import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """VAE Encoder using Linear layers."""

    def __init__(self, input_size: int, latent_size: int):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, latent_size)
        self.fc_logsigma = nn.Linear(64, latent_size)

    def forward(self, x):
        # x: [batch_size, input_size]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)
        return mu, logsigma


class Decoder(nn.Module):
    """VAE Decoder using Linear layers."""

    def __init__(self, input_size: int, latent_size: int):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, input_size)

    def forward(self, z):
        # z: [batch_size, latent_size]
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        reconstruction = torch.sigmoid(self.fc3(z))
        return reconstruction


class VAE(nn.Module):
    def __init__(self, input_size: int, latent_size: int):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, latent_size)
        self.decoder = Decoder(input_size, latent_size)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = torch.exp(logsigma)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps  # reparameterisation trick
        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
