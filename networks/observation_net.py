from typing import Dict
from tianshou.data import Batch

import torch
from torch import nn

import gymnasium as gym
import numpy as np

from nle import nethack

from .utils import Crop


class NetHackObsNet(nn.Module):
    """The main network for processing NetHack observations.

    It's the initial part of the network in the NLE paper (https://arxiv.org/abs/2006.13760), i.e., the CNNs and the MLP, without the RNN.
    I kept it mostly intact due to the interesting local/global interplay.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        embedding_dim: int = 32,
        crop_dim: int = 9,
        num_layers: int = 5,
    ):
        super().__init__()

        self.glyph_shape = observation_space["glyphs"].shape
        self.blstats_size = observation_space["blstats"].shape[0]

        self.H = self.glyph_shape[0]
        self.W = self.glyph_shape[1]

        self.k_dim = embedding_dim
        self.h_dim = 512
        self.o_dim = self.h_dim // 2

        self.crop_dim = crop_dim
        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)

        K = embedding_dim  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = num_layers  # number of convnet layers

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [Y]

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]
        self.extract_representation = nn.Sequential(
            *self._interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )

        conv_extract_crop = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]
        self.extract_crop_representation = nn.Sequential(
            *self._interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract))
        )

        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_size, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        )

        intermediate_out_dim = self.k_dim
        # CNN over full glyph map
        intermediate_out_dim += self.H * self.W * Y
        # CNN crop model
        intermediate_out_dim += self.crop_dim**2 * Y

        # final fully connected component
        self.fc = nn.Sequential(
            nn.Linear(intermediate_out_dim, self.h_dim * 2),
            nn.ReLU(),
            nn.Linear(self.h_dim * 2, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.o_dim),
            nn.ReLU(),
        )

    def forward(
        self, env_out_batch: Batch | Dict[str, np.ndarray | torch.Tensor]
    ) -> torch.Tensor:
        # -- [B x H x W]
        glyphs = torch.as_tensor(env_out_batch["glyphs"]).long()
        B, *_ = glyphs.shape

        # -- [B x F]
        blstats = torch.as_tensor(env_out_batch["blstats"]).float()
        # -- [B x 2]
        coordinates = blstats[:, :2]
        # -- [B x K]
        blstats_emb = self.embed_blstats(blstats)
        assert blstats_emb.shape[0] == B

        # -- [B x H' x W']
        crop = self.crop(glyphs, coordinates)
        # -- [B x H' x W' x K]
        crop_emb = self._select(self.embed, crop)
        # -- [B x K x W' x H']
        crop_emb = crop_emb.permute(0, 3, 1, 2)
        # -- [B x W' x H' x K]
        crop_rep = self.extract_crop_representation(crop_emb).flatten(1)
        # -- [B x K']
        assert crop_rep.shape[0] == B

        # -- [B x H x W x K]
        glyphs_emb = self._select(self.embed, glyphs)
        # -- [B x K x W x H]
        glyphs_emb = glyphs_emb.permute(0, 3, 1, 2)
        # -- [B x W x H x K]
        glyphs_rep = self.extract_representation(glyphs_emb).flatten(1)
        # -- [B x K']
        assert glyphs_rep.shape[0] == B

        st = torch.cat([blstats_emb, crop_rep, glyphs_rep], dim=1)

        # -- [B x K]
        logits = self.fc(st)
        return logits

    def _interleave(self, xs, ys):
        return [val for pair in zip(xs, ys) for val in pair]

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))
