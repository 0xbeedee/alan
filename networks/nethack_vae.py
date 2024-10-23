from typing import Tuple, Dict
import gymnasium as gym

import torch
from torch import nn
from collections import namedtuple, OrderedDict

from .nethack_encoders_decoders import *
from .utils import Crop


class NetHackEncoder(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        latent_dim: int,
        hidden_dim: int = 512,
        crop_dim: int = 9,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        # latent dimension of mu and logsigma (and, thus, z)
        self.latent_dim = latent_dim
        # each encoder outputs h_dim-dimensional tensors
        self.h_dim = hidden_dim
        self.crop_dim = crop_dim
        self.device = device

        self.spatial_keys = [
            "glyphs",
            # "chars",
            # "colors",
            # "specials",
            # "tty_chars",
            # "tty_colors",
        ]
        # self.inv_keys = ["inv_glyphs", "inv_letters", "inv_oclasses", "inv_strs"]

        spatial_data = dict(
            [(key, self._observation_data(key)) for key in self.spatial_keys]
        )
        self.spatial_encoder = SpatialEncoder(
            h_dim=self.h_dim,
            input_shapes=spatial_data,
            device=device,
        )

        self.ego_encoder = EgocentricEncoder(
            h_dim=self.h_dim,
            input_shape=(
                self._observation_data("glyphs").num_classes,
                (self.crop_dim, self.crop_dim),
            ),
            device=device,
        )
        self.crop = Crop(
            *observation_space["glyphs"].shape, self.crop_dim, self.crop_dim
        )

        # inv_data = dict([(key, self._observation_data(key)) for key in self.inv_keys])
        # self.inventory_encoder = InventoryEncoder(
        #     h_dim=self.h_dim,
        #     inv_shapes=inv_data,
        #     device=device,
        # )

        # self.message_encoder = MessageEncoder(
        #     h_dim=self.h_dim,
        #     message_shape=self._observation_data("message"),
        #     device=device,
        # )

        self.blstats_encoder = BlstatsEncoder(
            h_dim=self.h_dim,
            blstats_size=self._observation_data("blstats").shape[0],
            device=device,
        )

        # self.screen_descriptions_encoder = ScreenDescriptionsEncoder(
        #     h_dim=self.h_dim,
        #     input_shape=self._observation_data("screen_descriptions"),
        #     device=device,
        # )

        # self.tty_cursor_encoder = TTYCursorEncoder(
        #     h_dim=self.h_dim,
        #     device=device,
        # )

        # just as in the original obs_net, we only consider the glyphs, the blstats and the egocentric view of the agent
        self.o_dim = self.h_dim * 3
        # combine all features and output mu and logsigma
        self.fc_mu = nn.Linear(self.o_dim, self.latent_dim).to(device)
        self.fc_logsigma = nn.Linear(self.o_dim, self.latent_dim).to(device)

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial_inputs = {key: inputs[key] for key in self.spatial_keys}
        cropped_inputs = self.crop(
            torch.as_tensor(inputs["glyphs"], device=self.device),
            torch.as_tensor(inputs["blstats"][:, :2], device=self.device),
        )
        # inventory_inputs = {key: inputs[key] for key in self.inv_keys}

        spatial_features = self.spatial_encoder(spatial_inputs)
        egocentric_features = self.ego_encoder(cropped_inputs)
        # inventory_features = self.inventory_encoder(inventory_inputs)
        # message_features = self.message_encoder(inputs["message"])
        blstats_features = self.blstats_encoder(inputs["blstats"])
        # screen_description_features = self.screen_descriptions_encoder(
        #     inputs["screen_descriptions"]
        # )
        # tty_cursor_features = self.tty_cursor_encoder(inputs["tty_cursor"])

        combined = torch.cat(
            [
                spatial_features,
                egocentric_features,
                # inventory_features,
                # message_features,
                blstats_features,
                # screen_description_features,
                # tty_cursor_features,
            ],
            dim=1,
        )  # (B, o_dim)

        mu = self.fc_mu(combined)  # (B, latent_dim)
        logsigma = self.fc_logsigma(combined)  # (B, latent_dim)
        # it's convenient to have the encoder return z as well
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
        encoder_out_dim: int,
        hidden_dim: int = 512,
        crop_dim: int = 9,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.latent_dim = latent_dim
        self.eo_dim = encoder_out_dim
        self.h_dim = hidden_dim
        self.crop_dim = crop_dim
        self.device = device

        # fully connected layer to expand latent vector
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.eo_dim),
            nn.ReLU(),
        ).to(device)

        self.spatial_keys = [
            "glyphs",
            # "chars",
            # "colors",
            # "specials",
            # "tty_chars",
            # "tty_colors",
        ]
        # self.inv_keys = ["inv_glyphs", "inv_letters", "inv_oclasses", "inv_strs"]

        spatial_data = dict(
            [(key, self._observation_data(key)) for key in self.spatial_keys]
        )
        self.spatial_decoder = SpatialDecoder(
            h_dim=self.h_dim,
            output_shapes=spatial_data,
            device=device,
        )

        self.ego_decoder = EgocentricDecoder(
            h_dim=self.h_dim,
            output_shape=(
                self._observation_data("glyphs").num_classes,
                (self.crop_dim, self.crop_dim),
            ),
            device=device,
        )

        # inv_data = dict([(key, self._observation_data(key)) for key in self.inv_keys])
        # self.inventory_decoder = InventoryDecoder(
        #     h_dim=self.h_dim,
        #     inv_shapes=inv_data,
        #     device=device,
        # )

        # self.message_decoder = MessageDecoder(
        #     h_dim=self.h_dim,
        #     message_shape=self._observation_data("message"),
        #     device=device,
        # )

        self.blstats_decoder = BlstatsDecoder(
            h_dim=self.h_dim,
            blstats_size=self._observation_data("blstats").shape[0],
            device=device,
        )

        # self.screen_descriptions_decoder = ScreenDescriptionsDecoder(
        #     h_dim=self.h_dim,
        #     output_shape=self._observation_data("screen_descriptions"),
        #     device=device,
        # )

        # self.tty_cursor_decoder = TTYCursorDecoder(
        #     h_dim=self.h_dim,
        #     device=device,
        # )

        self.decoders = OrderedDict()
        self.decoders["spatial"] = (
            self.spatial_decoder,
            self.h_dim * len(self.spatial_keys),
        )
        self.decoders["egocentric_view"] = (self.ego_decoder, self.h_dim)
        # self.decoders["inventory"] = (
        #     self.inventory_decoder,
        #     self.h_dim * len(self.inv_keys),
        # )
        # self.decoders["message"] = (self.message_decoder, self.h_dim)
        self.decoders["blstats"] = (self.blstats_decoder, self.h_dim)
        # self.decoders["screen_descriptions"] = (
        #     self.screen_descriptions_decoder,
        #     self.h_dim,
        # )
        # self.decoders["tty_cursor"] = (self.tty_cursor_decoder, self.h_dim)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        reconstructions = {}
        x = self.fc(z)  # (B, eo_dim)

        chunk_sizes = [size for _, size in self.decoders.values()]
        x_chunks = torch.split(x, chunk_sizes, dim=1)

        for (name, (decoder, _)), x_chunk in zip(self.decoders.items(), x_chunks):
            if name == "spatial":
                # spatial_decoder returns multiple keys
                spatial_outputs = decoder(x_chunk)
                reconstructions.update(spatial_outputs)
            # elif name == "inventory":
            #     # inventory_decoder returns multiple keys
            #     inventory_outputs = decoder(x_chunk)
            #     reconstructions.update(inventory_outputs)
            else:
                # other decoders return a single output
                reconstructions[name] = decoder(x_chunk)

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
        *,
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
            self.encoder.o_dim,
            hidden_dim=hidden_dim,
            device=device,
        )

        # it's convenient to keep these here to centralise key handling
        self.categorical_keys = [
            "glyphs",
            # "chars",
            # "colors",
            # "specials",
            # "tty_chars",
            # "tty_colors",
            "egocentric_view",
            # "inv_glyphs",
            # "inv_letters",
            # "inv_oclasses",
            # "inv_strs",
            # "message",
            # "screen_descriptions",
        ]
        self.continuous_keys = [
            "blstats",
            # "tty_cursor",
        ]

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        z, mu, logsigma = self.encoder(inputs)
        reconstructions = self.decoder(z)
        return reconstructions, mu, logsigma
