from typing import Dict, Self
from tianshou.data import Batch

import torch
from torch import nn

import gymnasium as gym
import numpy as np

from nle import nethack

from .utils import Crop


class NetHackObsNet(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        embedding_dim: int = 32,
        crop_dim: int = 9,
        num_layers: int = 5,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device

        self.glyph_shape = observation_space["glyphs"].shape
        self.blstats_size = observation_space["blstats"].shape[0]

        self.H = self.glyph_shape[0]
        self.W = self.glyph_shape[1]

        self.k_dim = embedding_dim
        self.h_dim = 512
        self.o_dim = self.h_dim // 2

        self.crop_dim = crop_dim
        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim).to(device)

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
        ).to(device)

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
        ).to(device)

        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_size, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        ).to(device)

        intermediate_out_dim = self.k_dim
        intermediate_out_dim += self.H * self.W * Y
        intermediate_out_dim += self.crop_dim**2 * Y

        self.fc = nn.Sequential(
            nn.Linear(intermediate_out_dim, self.h_dim * 2),
            nn.ReLU(),
            nn.Linear(self.h_dim * 2, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.o_dim),
            nn.ReLU(),
        ).to(device)

    def forward(
        self, env_out_batch: Batch | Dict[str, np.ndarray | torch.Tensor]
    ) -> torch.Tensor:
        glyphs = torch.as_tensor(env_out_batch["glyphs"], device=self.device).long()
        B, *_ = glyphs.shape

        blstats = torch.as_tensor(env_out_batch["blstats"], device=self.device).float()
        coordinates = blstats[:, :2]
        blstats_emb = self.embed_blstats(blstats)
        assert blstats_emb.shape[0] == B

        crop = self.crop(glyphs, coordinates)
        crop_emb = self._select(self.embed, crop)
        crop_emb = crop_emb.permute(0, 3, 1, 2)
        crop_rep = self.extract_crop_representation(crop_emb).flatten(1)
        assert crop_rep.shape[0] == B

        glyphs_emb = self._select(self.embed, glyphs)
        glyphs_emb = glyphs_emb.permute(0, 3, 1, 2)
        glyphs_rep = self.extract_representation(glyphs_emb).flatten(1)
        assert glyphs_rep.shape[0] == B

        st = torch.cat([blstats_emb, crop_rep, glyphs_rep], dim=1)

        logits = self.fc(st)
        return logits

    def _interleave(self, xs, ys):
        return [val for pair in zip(xs, ys) for val in pair]

    def _select(self, embed, x):
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def to(self, device) -> Self:
        self.device = device
        return super().to(device)
