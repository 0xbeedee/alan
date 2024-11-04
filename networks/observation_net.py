from tianshou.data import Batch

import torch
from torch import nn


class ObsNet(nn.Module):
    """A wrapper for the VAE Encoder that provides a clean interface for Tianshou.

    The point of it is to minimise changes to the existing code, while also adding to the conceptual clarity of it.
    """

    def __init__(
        self,
        *,
        vae_encoder: nn.Module,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.encoder = vae_encoder.to(self.device)
        self.o_dim = self.encoder.latent_dim

    @torch.no_grad()
    def forward(self, inputs: Batch) -> torch.Tensor:
        z, *_ = self.encoder(inputs)
        return z
