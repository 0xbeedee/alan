from core.types import EnvModelProtocol

import gymnasium as gym
import numpy as np

import torch


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
    ):
        super().__init__()

        self.obs_net = env_model.vae.encoder
        self.vae = env_model.vae
        self.mdnrnn = env_model.mdnrnn
        self.device = env_model.device

        # these are usually exactly the same as in the original environment
        self.action_space = action_space
        self.observation_space = observation_space

        self.hidden_state = None
        self.z = None

        self.t = 0
        # TODO possibly as input to init?
        self.max_episode_length = 1000

    def reset(self, initial_obs=None):
        self.t = 0

        hidden_dim = self.mdnrnn.hidden_dim
        self.hidden_state = (
            torch.zeros(1, hidden_dim, device=self.device),  # h_0
            torch.zeros(1, hidden_dim, device=self.device),  # c_0
        )

        if initial_obs is not None:
            obs_out = self._pass_through_obsnet(initial_obs)
            self.z, *_ = self.obs_net(obs_out)
        else:
            # sample random latent vector from standard normal distribution
            latent_dim = self.mdnrnn.latent_dim
            self.z = torch.randn(1, latent_dim, device=self.device)

        obs = self._decode(self.z)
        # TODO could I not use info?
        info = {}
        return obs, info

    @torch.no_grad()
    def step(self, action):
        self.t += 1

        action = (
            torch.as_tensor(action, device=self.device).unsqueeze(0).unsqueeze(1)
        )  # (1, action_dim)
        z_t = self.z  # (1, latent_dim)
        mus, sigmas, logpi, r, d, self.hidden_state = self.mdnrnn(
            action, z_t, hidden=self.hidden_state
        )

        self.z = self._sample_mdn(mus, sigmas, logpi)

        obs = self._decode(self.z)
        info = {}
        # TODO this might not be such a bad idea => if the ds estimate is too high we'll end things too soon!
        # TODO could add some counter to track the number of steps and only after that number check the d array
        done = self.t >= self.max_episode_length or torch.sigmoid(d).item() > 0.5
        return obs, r.item(), done, False, info

    def render(self, mode="human"):
        # TODO we could try rendering the latent space
        pass

    def close(self):
        pass

    def _pass_through_obsnet(self, env_obs):
        return env_obs

    def _decode(self, z):
        """Decodes the reconstruction returned by the VAE decoder into an observation compatible with the ones provided by the real environment."""
        # TODO only works for NetHack at present
        recons = self.vae.decoder(z)

        observation = {}
        for key, recon in recons.items():
            observation[key] = (
                torch.argmax(recon, dim=1).squeeze()
                if key in self.vae.categorical_keys
                else recon.squeeze()
            )
        return observation

    def _sample_mdn(self, mus, sigmas, logpi):
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
