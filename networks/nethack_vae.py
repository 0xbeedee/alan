from typing import Tuple, Dict
import gymnasium as gym

import torch
from torch import nn
from collections import namedtuple

from .nethack_encoders_decoders import *


class NetHackEncoder(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        latent_dim: int,
        hidden_dim: int = 512,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.h_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device

        self.spatial_keys = [
            "glyphs",
            "chars",
            "colors",
            "specials",
            "tty_chars",
            "tty_colors",
        ]
        self.inv_keys = ["inv_glyphs", "inv_letters", "inv_oclasses", "inv_strs"]

        spatial_data = dict(
            [(key, self._observation_data(key)) for key in self.spatial_keys]
        )
        self.spatial_encoder = SpatialEncoder(
            h_dim=self.h_dim,
            input_shapes=spatial_data,
            device=device,
        )

        inv_data = dict([(key, self._observation_data(key)) for key in self.inv_keys])
        self.inventory_encoder = InventoryEncoder(
            h_dim=self.h_dim,
            inv_shapes=inv_data,
            device=device,
        )

        self.message_encoder = MessageEncoder(
            h_dim=self.h_dim,
            message_shape=self._observation_data("message"),
            device=device,
        )

        self.blstats_encoder = BlstatsEncoder(
            h_dim=self.h_dim,
            blstats_size=self._observation_data("blstats").shape[0],
            device=device,
        )

        self.screen_descriptions_encoder = ScreenDescriptionsEncoder(
            h_dim=self.h_dim,
            input_shape=self._observation_data("screen_descriptions"),
            embedding_dim=32,
            device=device,
        )

        self.tty_cursor_encoder = TtyCursorEncoder(
            h_dim=self.h_dim,
            device=device,
        )

        # combine all features and output mu and logsigma
        self.o_dim = self.h_dim * len(observation_space.keys())
        self.fc_mu = nn.Linear(self.o_dim, self.latent_dim).to(device)
        self.fc_logsigma = nn.Linear(self.o_dim, self.latent_dim).to(device)

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial_inputs = {key: inputs[key] for key in self.spatial_keys}
        inventory_inputs = {key: inputs[key] for key in self.inv_keys}

        spatial_features = self.spatial_encoder(spatial_inputs)
        inventory_features = self.inventory_encoder(inventory_inputs)
        message_features = self.message_encoder(inputs["message"])
        blstats_features = self.blstats_encoder(inputs["blstats"])
        screen_description_features = self.screen_descriptions_encoder(
            inputs["screen_descriptions"]
        )
        tty_cursor_features = self.tty_cursor_encoder(inputs["tty_cursor"])

        combined = torch.cat(
            [
                spatial_features,
                inventory_features,
                message_features,
                blstats_features,
                screen_description_features,
                tty_cursor_features,
            ],
            dim=1,
        )  # (B, o_dim)

        mu = self.fc_mu(combined)  # (B, latent_dim)
        logsigma = self.fc_logsigma(combined)  # (B, latent_dim)
        # it's convenient to have the encoder also return z
        z = self._reparameterise(mu, logsigma)  # (B, latent_dim)
        return z, mu, logsigma

    def _observation_data(self, key: str) -> Tuple[int, Tuple[int, ...]]:
        """Extracts the observation data corresponding to a key."""
        # we take the mean() because .high returns arrays of equal entries, and we only need one (this way is cleaner than indexing)
        data_tuple = namedtuple("obs_data", ["num_classes", "shape"])
        data = data_tuple(
            num_classes=int(self.observation_space[key].high.mean() + 1),
            shape=self.observation_space[key].shape,
        )
        return data

    def _reparameterise(self, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        """Uses the reparameterisatin trick to obtain an observation in latent space, given the means and logsigmas."""
        sigma = torch.exp(logsigma)
        eps = torch.randn_like(sigma, device=self.device)
        return mu + sigma * eps


class NetHackDecoder(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        latent_dim: int,
        hidden_dim: int = 512,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.h_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device

        # fully connected layer to expand latent vector
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.h_dim),
            nn.ReLU(),
        ).to(device)

        self.spatial_keys = [
            "glyphs",
            "chars",
            "colors",
            "specials",
            "tty_chars",
            "tty_colors",
        ]
        self.inv_keys = ["inv_glyphs", "inv_letters", "inv_oclasses", "inv_strs"]

        spatial_data = dict(
            [(key, self._observation_data(key)) for key in self.spatial_keys]
        )
        self.spatial_decoder = SpatialDecoder(
            h_dim=self.h_dim,
            input_shapes=spatial_data,
            device=device,
        )

        inv_data = dict([(key, self._observation_data(key)) for key in self.inv_keys])
        self.inventory_decoder = InventoryDecoder(
            h_dim=self.h_dim,
            inv_shapes=inv_data,
            device=device,
        )

        self.message_decoder = MessageDecoder(
            h_dim=self.h_dim,
            message_shape=self._observation_data("message"),
            device=device,
        )

        self.blstats_decoder = BlstatsDecoder(
            h_dim=self.h_dim,
            blstats_size=self._observation_data("blstats").shape[0],
            device=device,
        )

        self.screen_descriptions_decoder = ScreenDescriptionsDecoder(
            h_dim=self.h_dim,
            input_shape=self._observation_data("screen_descriptions"),
            embedding_dim=32,
            device=device,
        )

        self.tty_cursor_decoder = TtyCursorDecoder(
            h_dim=self.h_dim,
            device=device,
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.fc(z)  # (B, h_dim)

        spatial_outputs = self.spatial_decoder(x)
        inventory_outputs = self.inventory_decoder(x)

        reconstructions = {}
        reconstructions.update(spatial_outputs)
        reconstructions.update(inventory_outputs)
        reconstructions["message"] = self.message_decoder(x)
        reconstructions["blstats"] = self.blstats_decoder(x)

        return reconstructions

    def _observation_data(self, key: str) -> Tuple[int, Tuple[int, ...]]:
        """Extracts the observation data corresponding to a key."""
        # we take the mean() because .high returns arrays of equal entries, and we only need one (this way is cleaner than indexing)
        data_tuple = namedtuple("obs_data", ["num_classes", "shape"])
        data = data_tuple(
            num_classes=int(self.observation_space[key].high.mean() + 1),
            shape=self.observation_space[key].shape,
        )
        return data


class NetHackVAE(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        latent_dim: int,
        hidden_dim: int = 512,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device

        self.encoder = NetHackEncoder(
            observation_space,
            latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.decoder = NetHackDecoder(
            observation_space,
            latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        )

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        z, mu, logsigma = self.encoder(inputs)
        reconstructions = self.decoder(z)
        return reconstructions, mu, logsigma
