from typing import Any, Dict, Optional
from core.types import EnvModelProtocol

import gymnasium as gym
import numpy as np

import torch

from models.utils import sample_mdn


class DreamEnv(gym.Env):
    """
    An Gymnasium environment enabling an agent to act within its world model as if it were acting in the real environment (i.e., allowing the agent to learn from its "dreams"/latent imagination).

    The idea is the same as the one used for the VizDoom experiment in the World Models paper (https://arxiv.org/abs/1803.10122).
    """

    def __init__(
        self,
        env_model: EnvModelProtocol,
        observation_space: gym.Space,
        action_space: gym.Space,
        min_nsteps: int = 100,
        max_nsteps: int = 1000,
    ):
        super().__init__()

        self.obs_net = env_model.vae.encoder
        self.vae = env_model.vae
        self.mdnrnn = env_model.mdnrnn
        self.device = env_model.device

        # these need to be the same as in the original environment
        self.action_space = action_space
        self.observation_space = observation_space

        self.min_nsteps = min_nsteps
        self.max_nsteps = max_nsteps

        self.hidden_state = None
        self.z = None
        self.t = 0

    # TODO i don't correctly set this initial_obs!
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Any] = None,
        initial_obs: Optional[Dict[str, np.ndarray | torch.Tensor]] = None,
    ):
        super().reset(seed=seed)
        self.t = 0

        hidden_dim = self.mdnrnn.hidden_dim
        self.hidden_state = (
            torch.zeros(1, hidden_dim, device=self.device),  # h_0
            torch.zeros(1, hidden_dim, device=self.device),  # c_0
        )

        if initial_obs is not None:
            self.z, *_ = self.obs_net(initial_obs)
        else:
            # sample random latent vector from standard normal distribution
            latent_dim = self.mdnrnn.latent_dim
            self.z = torch.randn(1, latent_dim, device=self.device)
        obs = self.vae.decode(self.z, is_dream=True)
        # TODO how can I use info?
        info = {}

        return obs, info

    @torch.no_grad()
    def step(self, action: int):
        self.t += 1

        action = torch.tensor([[action]], device=self.device)  # (1, action_dim)
        mus, sigmas, logpi, r, d, self.hidden_state = self.mdnrnn(
            action, self.z, hidden=self.hidden_state
        )
        reward = r.item()

        _, self.z = sample_mdn(mus, sigmas, logpi)
        obs = self.vae.decode(self.z, is_dream=True)
        info = {}

        terminated = torch.sigmoid(d).item() > 0.5
        truncated = self.t > self.max_nsteps
        if self.t < self.min_nsteps:
            # take at least min_nsteps in the environment
            terminated, truncated = False, False

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        pass

    def close(self):
        pass
