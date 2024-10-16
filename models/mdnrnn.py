import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Self


class MDNRNN(nn.Module):
    """
    Mixture Density Network Recurrent Neural Network (MDNRNN) model for predicting future latent states in a sequential manner.

    This model combines a recurrent neural network (RNN) with a Mixture Density Network (MDN) to handle the uncertainty in predictions by modeling the output distribution as a mixture of Gaussians.

    Adapted from https://github.com/ctallec/world-models/blob/master/models/mdrnn.py.
    """

    def __init__(
        self,
        *,
        latent_size: int,
        action_size: int,
        hidden_size: int,
        n_gaussian_comps: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.latent_size: int = latent_size
        self.action_size: int = action_size
        self.hidden_size: int = hidden_size
        self.n_gaussian_comps: int = n_gaussian_comps
        self.device = device

        # outputs parameters for the Gaussian Mixture Model (GMM)
        self.gmm_linear = nn.Linear(
            hidden_size, (2 * latent_size + 1) * n_gaussian_comps + 2
        ).to(device)

        self.rnn = nn.LSTM(latent_size + action_size, hidden_size).to(device)

    def forward(
        self,
        actions: torch.Tensor,
        latents: torch.Tensor,
        tau: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the MDNRNN for multiple time steps."""
        bs = latents.shape[0]  # batch size

        ins = torch.cat(
            [actions, latents], dim=1
        )  # (batch_size, action_size + latent_size)

        outs, _ = self.rnn(ins)  # (batch_size, hidden_size)

        # get GMM parameters and additional outputs
        gmm_outs = self.gmm_linear(
            outs
        )  # (batch_size, (2 * latent_size + 1) * n_gaussian_comps + 2)

        # to separate the GMM parameters
        stride = self.n_gaussian_comps * self.latent_size

        mus = gmm_outs[:, :stride].contiguous()  # (batch_size, stride)
        mus = mus.view(
            bs, self.n_gaussian_comps, self.latent_size
        )  # (batch_size, n_gaussian_comps, latent_size)

        sigmas = gmm_outs[:, stride : 2 * stride].contiguous()  # (batch_size, stride)
        sigmas = sigmas.view(
            bs, self.n_gaussian_comps, self.latent_size
        )  # (batch_size, n_gaussian_comps, latent_size)
        sigmas = torch.exp(sigmas)  # ensure positive standard deviations
        sigmas = sigmas * tau  # scale by the temperature

        # GMM coefficients
        pi = gmm_outs[
            :, 2 * stride : 2 * stride + self.n_gaussian_comps
        ].contiguous()  # (batch_size, n_gaussian_comps)
        pi = pi.view(bs, self.n_gaussian_comps)
        pi = pi / tau
        logpi = F.log_softmax(pi, dim=-1)

        # rewards and terminal (done) state indicators
        rs = gmm_outs[:, -2]  # (batch_size,)
        ds = gmm_outs[:, -1]  # (batch_size,)

        return mus, sigmas, logpi, rs, ds

    def to(self, device: torch.device) -> Self:
        self.device = device
        self.gmm_linear = self.gmm_linear.to(device)
        self.rnn = self.rnn.to(device)
        return super().to(device)
