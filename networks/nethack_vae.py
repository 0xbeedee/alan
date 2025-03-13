from typing import Tuple, Dict
from tianshou.data import Batch
import gymnasium as gym

from collections import namedtuple, OrderedDict

import torch
from torch import nn

from .nethack_encoders_decoders import *
from .utils import Crop, reparameterise


class NetHackVAE(nn.Module):
    """A variational autoencoder for the NetHack Learning Environment."""

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
        self, inputs: Batch
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        _, z, dist = self.encoder(inputs)
        reconstructions = self.decoder(z)
        return reconstructions, z, dist

    def decode(self, z: torch.Tensor, is_dream: bool = False):
        """Decodes the latent vector into an observation compatible with the ones provided by the NetHack environment."""
        recons = self.decoder(z)

        observation = {}
        for key, recon in recons.items():
            if key in self.categorical_keys:
                # the second dim is only useful for internal purposes
                recon = (
                    torch.argmax(recon, dim=1)
                    if recon.dim() > 3
                    else torch.argmax(recon, dim=0)
                )
            if is_dream:
                # the first dimension is always 1 when dreaming
                recon = recon.squeeze(0) if recon.dim() > 1 else recon
            observation[key] = recon

        return observation


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
        self.latent_out = nn.Linear(self.o_dim, self.latent_dim).to(device)
        self.fc_mu = nn.Linear(self.o_dim, self.latent_dim).to(device)
        self.fc_logsigma = nn.Linear(self.o_dim, self.latent_dim).to(device)

    def forward(self, inputs: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_inputs = {key: inputs[key] for key in self.spatial_keys}

        glyphs_tr = torch.as_tensor(inputs["glyphs"], device=self.device)
        blstats_tr = torch.as_tensor(inputs["blstats"], device=self.device)
        cropped_inputs = self.crop(glyphs_tr, blstats_tr[:, :2])
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

        h_combined = torch.cat(
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

        latent_obs = self.latent_out(h_combined)
        mu = self.fc_mu(h_combined)  # (B, latent_dim)
        logsigma = self.fc_logsigma(h_combined)  # (B, latent_dim)
        # it's convenient to have the encoder return z as well
        z, dist = reparameterise(mu, logsigma)  # (B, latent_dim)
        return latent_obs, z, dist

    def _observation_data(self, key: str) -> Tuple[int, Tuple[int, ...]]:
        """Extracts the observation data corresponding to a key."""
        # we take the mean() because .high returns arrays of equal entries, and we only need one (this way is cleaner than indexing)
        data_tuple = namedtuple("obs_data", ["num_classes", "shape"])
        data = data_tuple(
            num_classes=int(self.observation_space[key].high.mean() + 1),
            shape=self.observation_space[key].shape,
        )
        return data


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
