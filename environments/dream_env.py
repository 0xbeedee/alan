from core.types import EnvModelProtocol

import gymnasium as gym
import numpy as np

import torch
import torch.nn.functional as F
from tianshou.data import Batch


class DreamEnv(gym.Env):
    """
    An Gymnasium environment enabling an agent to act within its world model as if it were acting in the real environment (i.e., allowing the agent to learn from its "dreams"/latent imagination).

    The idea is the same as the one used for the VizDoom experiment in the World Models paper (https://arxiv.org/abs/1803.10122).
    """

    def __init__(self, env_model: EnvModelProtocol, action_space, observation_space):
        super().__init__()

        self.obs_net = env_model.mdnrnn_trainer.obs_net
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

        hidden_size = self.mdnrnn.hidden_size
        self.hidden_state = (
            torch.zeros(1, hidden_size, device=self.device),  # h_0
            torch.zeros(1, hidden_size, device=self.device),  # c_0
        )

        if initial_obs is not None:
            obs_out = self._pass_through_obsnet(initial_obs)
            obs_mu, obs_logsigma = self.vae.encoder(obs_out)

            self.z = obs_mu + obs_logsigma.exp() * torch.randn_like(obs_mu)
        else:
            # sample random latent vector from standard normal distribution
            latent_size = self.mdnrnn.latent_size
            self.z = torch.randn(1, latent_size).to(self.device)

        recon_obs = self.vae.decoder(self.z)
        # TODO could I not use info?
        info = {}
        return recon_obs, info

    @torch.no_grad()
    def step(self, action):
        self.t += 1

        action = torch.Tensor(action)  # (1, action_size)
        z_t = self.z  # (1, latent_size)
        input_tensor = torch.cat([action, z_t], dim=1)

        output, self.hidden_state = self.mdnrnn.rnn(input_tensor, self.hidden_state)
        gmm_output = self.mdnrnn.gmm_linear(output.squeeze(0))  # remove seq dim

        mus, sigmas, logpi, r, d = self._parse_gmm_output(gmm_output)

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

    def _pass_through_obsnet(self, env_obs):
        # unlike in the Collector, simply wrapping an environment observation with Batch and passing it to the obs_net doesn't work
        env_obs_batched = {}
        for key, value in env_obs.items():
            if isinstance(value, np.ndarray):
                value = np.expand_dims(value, axis=0)
            elif isinstance(value, torch.Tensor):
                value = value.unsqueeze(0)
            env_obs_batched[key] = value

        return self.obs_net(env_obs_batched)

    def _parse_gmm_output(self, gmm_outs):
        stride = self.mdnrnn.n_gaussian_comps * self.mdnrnn.latent_size

        # gmm_outs must have shape (1, output_size)
        if gmm_outs.dim() == 1:
            gmm_outs = gmm_outs.unsqueeze(0)

        mus = gmm_outs[:, :stride]
        mus = mus.view(1, self.mdnrnn.n_gaussian_comps, self.mdnrnn.latent_size)

        sigmas = gmm_outs[:, stride : 2 * stride]
        sigmas = sigmas.view(1, self.mdnrnn.n_gaussian_comps, self.mdnrnn.latent_size)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, 2 * stride : 2 * stride + self.mdnrnn.n_gaussian_comps]
        logpi = F.log_softmax(pi, dim=-1)

        r = gmm_outs[:, -2]
        d = gmm_outs[:, -1]

        return mus, sigmas, logpi, r, d

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
