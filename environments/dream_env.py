from typing import Any, Dict, Optional
from core.types import EnvModelProtocol

import gymnasium as gym
import numpy as np

import torch


class DreamEnv(gym.Env):
    """
    An Gymnasium environment enabling an agent to act within its world model as if it were acting in the real environment (i.e., allowing the agent to learn from its "dreams"/latent imagination).

    The idea is the same as the one used for the VizDoom experiment in the World Models paper (https://arxiv.org/abs/1803.10122).
    """

    # TODO only works for NetHack at present

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
        obs = self._decode(self.z)
        # TODO how can I use info?
        info = {}

        return obs, info

    @torch.no_grad()
    def step(self, action: int):
        self.t += 1

        action = torch.tensor([[action]], device=self.device)  # (1, action_dim)
        z_t = self.z  # (1, latent_dim)
        mus, sigmas, logpi, r, d, self.hidden_state = self.mdnrnn(
            action, z_t, hidden=self.hidden_state
        )
        reward = r.item()

        self.z = self._sample_mdn(mus, sigmas, logpi)
        obs = self._decode(self.z)
        info = {}

        terminated = torch.sigmoid(d).item() > 0.5
        truncated = self.t > self.max_nsteps
        if self.t < self.min_nsteps:
            # take at least min_nsteps in the environment
            terminated, truncated = False, False

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        # TODO try rendering the latent space?
        pass

    def close(self):
        pass

    def _decode(self, z: torch.Tensor):
        """Decodes the reconstruction returned by the VAE decoder into an observation compatible with the ones provided by the real environment."""
        recons = self.vae.decoder(z)

        observation = {}
        for key, recon in recons.items():
            observation[key] = (
                torch.argmax(recon, dim=1).squeeze()
                if key in self.vae.categorical_keys
                else recon.squeeze()
            )
        return observation

    def _sample_mdn(
        self, mus: torch.Tensor, sigmas: torch.testing, logpi: torch.Tensor
    ):
        """Samples from the MDN output to get the next latent state."""
        # remove batch dimensions
        mus = mus.squeeze(0)  # (n_gaussian_comps, latent_dim)
        sigmas = sigmas.squeeze(0)  # (n_gaussian_comps, latent_dim)
        logpi = logpi.squeeze(0)  # (n_gaussian_comps,)

        # convert logpi to probabilities
        pi = torch.exp(logpi).cpu().numpy()
        pi = pi / np.sum(pi)

        component = np.random.choice(len(pi), p=pi)
        mu_c = mus[component]  # (latent_dim,)
        sigma_c = sigmas[component]  # (latent_dim,)

        z = mu_c + sigma_c * torch.randn_like(mu_c).to(self.device)
        # add batch dimension
        z = z.unsqueeze(0)  # (1, latent_dim)
        return z
