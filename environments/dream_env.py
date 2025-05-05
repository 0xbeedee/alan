from typing import Any, Optional, Tuple, Dict, SupportsFloat
from core.types import EnvModelProtocol
from core.policy import EnvModelProtocol

import gymnasium as gym
from tianshou.data import Batch
from torch import Tensor
from torch import nn
import torch
import numpy as np
import logging

from models.utils import sample_mdn


class DreamEnv(gym.Env):
    """
    An Gymnasium environment enabling an agent to act within its world model as if it were acting in the real environment (i.e., allowing the agent to learn from its "dreams"/latent imagination).

    The idea is the same as the one used for the VizDoom experiment in the World Models paper (https://arxiv.org/abs/1803.10122).
    """

    # TODO: Add render modes if needed
    render_mode = None

    def __init__(
        self,
        env_model: EnvModelProtocol,
        observation_space: gym.Space,
        action_space: gym.Space,
        min_nsteps: int = 100,
        max_nsteps: int = 1000,
    ):
        super().__init__()

        self.obs_net: nn.Module = env_model.vae.encoder
        self.vae: nn.Module = env_model.vae  # VAE model (encoder/decoder)
        self.mdnrnn: nn.Module = env_model.mdnrnn  # MDN-RNN model
        self.device: torch.device = env_model.device

        # these need to be the same as in the original environment
        self.action_space = action_space
        self.observation_space = observation_space

        self.min_nsteps = min_nsteps
        self.max_nsteps = max_nsteps

        # Internal state
        self.hidden_state: Tuple[Tensor, Tensor] | None = (
            None  # MDN-RNN hidden state (h, c)
        )
        self.z: Tensor | None = None  # Current VAE latent state
        self.t: int = 0  # Current timestep within the dream

    def reset(
        self,
        *,  # Enforce keyword arguments for seed and options
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[Any, Dict[str, Any]]:  # Returns obs, info
        super().reset(seed=seed)
        self.t = 0

        hidden_dim = self.mdnrnn.hidden_dim
        self.hidden_state = (
            torch.zeros(1, hidden_dim, device=self.device),  # h_0
            torch.zeros(1, hidden_dim, device=self.device),  # c_0
        )

        # Get initial real observation to encode
        initial_obs_batch = self._get_initial_obs()
        # Encode initial observation to get starting latent state z
        _, self.z, _ = self.vae(initial_obs_batch)

        # Decode initial latent state to get first dreamed observation
        # is_dream=True might alter decoder behavior (e.g., disable stochasticity)
        obs = self.vae.decode(self.z, is_dream=True)
        info = {}  # Info dict currently unused

        return obs, info

    @torch.no_grad()
    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        self.t += 1

        # Ensure action and z are correctly shaped tensors
        # Assume action needs batch dim [1, action_dim] or similar based on MDNRNN input needs
        action_tensor = torch.as_tensor(action, device=self.device).unsqueeze(0)
        # Z should already be [1, latent_dim]
        assert self.z is not None and self.z.shape[0] == 1
        assert self.hidden_state is not None

        # Predict next latent state, reward, done using MDN-RNN
        mus, sigmas, logpi, r_pred, d_pred, self.hidden_state = self.mdnrnn(
            action_tensor, self.z, hidden=self.hidden_state
        )
        # Extract predicted reward (assuming scalar reward prediction)
        reward = r_pred.item()

        # Sample next latent state from the MDN mixture
        _, self.z = sample_mdn(mus, sigmas, logpi)
        # Decode next latent state to get dreamed observation
        # is_dream=True might alter decoder behavior
        obs = self.vae.decode(self.z, is_dream=True)
        info = {}  # Info dict currently unused

        # Determine termination based on predicted done probability
        terminated = torch.sigmoid(d_pred).item() > 0.5
        # Determine truncation based on max steps
        truncated = self.t >= self.max_nsteps
        # Ensure minimum steps are taken
        if self.t < self.min_nsteps:
            terminated, truncated = False, False

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        """Dream environment does not support rendering."""
        pass

    def close(self) -> None:
        """No resources to close."""
        pass

    def _get_initial_obs(self) -> Batch:
        """Samples an initial observation from the original env space and prepares it as a batch for VAE."""
        initial_obs = self.observation_space.sample()
        # Add batch dimension if needed (e.g., for Dict spaces)
        # Assumes VAE expects Batch(obs=..., ...) structure if space is Dict
        if isinstance(self.observation_space, gym.spaces.Dict):
            initial_obs = Batch(
                {k: np.expand_dims(v, 0) for k, v in initial_obs.items()}
            )
        elif isinstance(initial_obs, np.ndarray):
            # Add batch dim for simple Box/Discrete space
            initial_obs = Batch(obs=np.expand_dims(initial_obs, 0))
        else:
            # Fallback/warning for unexpected space types
            logging.warning(
                f"Initial observation sampling for space {type(self.observation_space)} might not be correctly batched."
            )
            initial_obs = Batch(obs=initial_obs)  # Hope for the best?

        # Move data to the correct device (assuming VAE expects tensors)
        initial_obs = initial_obs.to_torch(device=self.device)
        return initial_obs
