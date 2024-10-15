from core.types import EnvModelProtocol

import gymnasium as gym
import numpy as np

import torch


# TODO this needs to be tested!
class DreamEnv(gym.Env):
    """
    An Gymnasium environment enabling an agent to act within its world model as if it were acting in the real environment (i.e., allowing the agent to learn from its "dreams"/latent imagination).

    The idea is the same as the one used for the VizDoom experiment in the World Models paper (https://arxiv.org/abs/1803.10122).
    """

    def __init__(self, model: EnvModelProtocol, action_space, observation_space):
        super().__init__()

        self.obs_net = model.mdnrnn_trainer.obs_net
        self.vae = model.vae
        # TODO make the necessary changes to adjust the MDNRNNCell code
        self.mdnrnn = model.mdnrnn
        self.device = model.device

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

        hidden_size = self.mdnrnn.hidden_size
        self.hidden_state = (
            torch.zeros(1, hidden_size, device=self.device),
            torch.zeros(1, hidden_size, device=self.device),
        )

        if initial_obs is not None:
            obs_mu, obs_logsigma = self.vae.encoder(self.obs_net(initial_obs))
            self.z = obs_mu + obs_logsigma.exp() * torch.randn_like(obs_mu)
        else:
            # sample random latent vector from standard normal distribution
            latent_size = self.mdnrnn.latent_size
            self.z = torch.randn(1, latent_size).to(self.device)

        recon_obs = self.vae.decoder(self.z)
        return recon_obs

    def step(self, action):
        self.t += 1

        mus, sigmas, logpi, r, d, self.hidden_state = self.mdnrnn(
            torch.Tensor(action), self.z, self.hidden_state
        )
        self.z = self._sample_mdn(mus, sigmas, logpi)
        recon_obs = self.vae.decoder(self.z)

        done = self.t >= self.max_episode_length or torch.sigmoid(d).item() > 0.5
        info = {}
        return recon_obs, r.item(), done, info

    def render(self, mode="human"):
        # TODO we could try rendering the latent space
        pass

    def close(self):
        pass

    def _sample_mdn(self, mus, sigmas, logpi):
        """Samples from the MDN output to get the next latent state."""
        # remove batch dimensions
        mus = mus.squeeze(0)  # (n_gaussian_comps, latent_size)
        sigmas = sigmas.squeeze(0)  # (n_gaussian_comps, latent_size)
        logpi = logpi.squeeze(0)  # (n_gaussian_comps,)

        # convert logpi to probabilities
        pi = torch.exp(logpi).cpu().numpy()
        pi = pi / np.sum(pi)

        component = np.random.choice(len(pi), p=pi)
        mu_c = mus[component]  # (latent_size,)
        sigma_c = sigmas[component]  # (latent_size,)

        z = mu_c + sigma_c * torch.randn_like(mu_c).to(self.device)
        # add batch dimension
        z = z.unsqueeze(0)  # (1, latent_size)
        return z
