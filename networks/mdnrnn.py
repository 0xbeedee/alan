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

        # initialize RNN with proper hidden state initialization
        self.rnn_cell = nn.LSTMCell(latent_dim + action_dim, hidden_dim).to(device)
        self.hidden = None

        # outputs parameters for the Gaussian Mixture Model (GMM)
        self.gmm_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, (2 * latent_dim + 1) * n_gaussian_comps + 2),
        ).to(device)

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with proper scaling."""
        for name, param in self.named_parameters():
            if "weight" in name and len(param.shape) >= 2:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name and len(param.shape) == 1:
                nn.init.constant_(param, 1.0)

    def reset_hidden(self, batch_size: int = 1):
        """Reset the hidden state of the RNN."""
        self.hidden = (
            torch.zeros(batch_size, self.hidden_dim, device=self.device),
            torch.zeros(batch_size, self.hidden_dim, device=self.device),
        )

    def forward(
        self,
        actions: torch.Tensor,
        latents: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        tau: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the MDNRNN for a single time step."""
        batch_size = latents.shape[0]

        if hidden is None:
            if self.hidden is None or self.hidden[0].shape[0] != batch_size:
                self.reset_hidden(batch_size)
            hidden = self.hidden

        outs, hidden = self.pass_through_rnn(actions, latents, hidden=hidden)
        self.hidden = hidden  # store the hidden state

        mus, sigmas, logpi, rs, ds = self._compute_gmm_parameters(
            outs, batch_size, tau=tau
        )

        return mus, sigmas, logpi, rs, ds, (outs, hidden)

    def pass_through_rnn(
        self,
        actions: torch.Tensor,
        latents: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass inputs through the RNN cell."""
        batch_size = latents.shape[0]

        if hidden is None:
            if self.hidden is None or self.hidden[0].shape[0] != batch_size:
                self.reset_hidden(batch_size)
            hidden = self.hidden

        ins = torch.cat(
            [actions, latents], dim=1
        )  # (batch_dim, action_dim + latent_dim)
        h_t, c_t = self.rnn_cell(ins, hx=hidden)  # each (batch_dim, hidden_dim)
        self.hidden = (h_t, c_t)
        return h_t, c_t

    def _compute_gmm_parameters(
        self, rnn_outs: torch.Tensor, bs: int, tau: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute GMM parameters from RNN outputs."""
        gmm_outs = self.gmm_linear(
            rnn_outs
        )  # (batch_dim, (2 * latent_dim + 1) * n_gaussian_comps + 2)

        # to separate the parameters
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
