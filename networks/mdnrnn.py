import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Self


class MDNRNN(nn.Module):
    """
    Mixture Density Network Recurrent Neural Network (MDNRNN) model for predicting future latent states in a sequential manner.

    This model combines a recurrent neural network (RNN) with a Mixture Density Network (MDN) to handle the uncertainty in predictions by modeling the output distribution as a mixture of Gaussians.

    Adapted from https://github.com/ctallec/world-models/blob/master/models/mdrnn.py.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        latent_dim: int,
        n_gaussian_comps: int,
        hidden_dim: int = 512,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.latent_dim: int = latent_dim
        self.action_dim: int = action_dim
        self.hidden_dim: int = hidden_dim
        self.n_gaussian_comps: int = n_gaussian_comps
        self.device = device

        # outputs parameters for the Gaussian Mixture Model (GMM)
        self.gmm_linear = nn.Linear(
            hidden_dim, (2 * latent_dim + 1) * n_gaussian_comps + 2
        ).to(device)

        self.rnn_cell = nn.LSTMCell(latent_dim + action_dim, hidden_dim).to(device)

    def forward(
        self,
        actions: torch.Tensor,
        latents: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        tau: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the MDNRNN for multiple time steps."""
        outs, hidden = self.pass_through_rnn(actions, latents, hidden=hidden)

        bs = latents.shape[0]  # batch size
        mus, sigmas, logpi, rs, ds = self._compute_gmm_parameters(outs, bs, tau=tau)

        return mus, sigmas, logpi, rs, ds, (outs, hidden)

    def pass_through_rnn(
        self,
        actions: torch.Tensor,
        latents: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ins = torch.cat(
            [actions, latents], dim=1
        )  # (batch_dim, action_dim + latent_dim)
        # h_t = hidden state at time t, c_t = cell state at time t
        h_t, c_t = self.rnn_cell(ins, hx=hidden)  # each (batch_dim, hidden_dim)

        return h_t, c_t

    def _compute_gmm_parameters(
        self, rnn_outs: torch.Tensor, bs: int, tau: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gmm_outs = self.gmm_linear(
            rnn_outs
        )  # (batch_dim, (2 * latent_dim + 1) * n_gaussian_comps + 2)

        # to separate the GMM parameters
        stride = self.n_gaussian_comps * self.latent_dim

        mus = gmm_outs[:, :stride].contiguous()  # (batch_dim, stride)
        mus = mus.view(
            bs, self.n_gaussian_comps, self.latent_dim
        )  # (batch_dim, n_gaussian_comps, latent_dim)

        sigmas = gmm_outs[:, stride : 2 * stride].contiguous()  # (batch_dim, stride)
        sigmas = sigmas.view(
            bs, self.n_gaussian_comps, self.latent_dim
        )  # (batch_dim, n_gaussian_comps, latent_dim)
        sigmas = torch.exp(sigmas)  # ensure positive standard deviations
        sigmas = sigmas * tau  # scale by the temperature

        pi = gmm_outs[
            :, 2 * stride : 2 * stride + self.n_gaussian_comps
        ].contiguous()  # (batch_dim, n_gaussian_comps)
        pi = pi.view(bs, self.n_gaussian_comps)
        pi = pi / tau  # scale by the temperature
        logpi = F.log_softmax(pi, dim=-1)

        # rewards and terminal (done) state indicators
        rs = gmm_outs[:, -2]  # (batch_dim,)
        ds = gmm_outs[:, -1]  # (batch_dim,)

        return mus, sigmas, logpi, rs, ds

    def to(self, device: torch.device) -> Self:
        self.device = device
        self.gmm_linear = self.gmm_linear.to(device)
        self.rnn_cell = self.rnn_cell.to(device)
        return super().to(device)
