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
        gaussian_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.latent_size: int = latent_size
        self.action_size: int = action_size
        self.hidden_size: int = hidden_size
        self.gaussian_size: int = gaussian_size
        self.device = device

        # outputs parameters for the Gaussian Mixture Model (GMM)
        self.gmm_linear = nn.Linear(
            hidden_size, (2 * latent_size + 1) * gaussian_size + 2
        ).to(device)

        self.rnn = nn.LSTM(latent_size + action_size, hidden_size).to(device)

    def forward(
        self, actions: torch.Tensor, latents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the MDNRNN for multiple time steps."""
        bs = latents.shape[0]

        ins = torch.cat(
            [actions, latents], dim=1
        )  # (batch_size, action_size + latent_size)

        outs, _ = self.rnn(ins)  # (batch_size, hidden_size)

        # get GMM parameters and additional outputs
        gmm_outs = self.gmm_linear(
            outs
        )  # (batch_size, (2 * latent_size + 1) * gaussian_size + 2)

        # to separate the GMM parameters
        stride = self.gaussian_size * self.latent_size

        mus = gmm_outs[:, :stride].contiguous()  # (batch_size, stride)
        mus = mus.view(
            bs, self.gaussian_size, self.latent_size
        )  # (batch_size, gaussian_size, latent_size)

        sigmas = gmm_outs[:, stride : 2 * stride].contiguous()  # (batch_size, stride)
        sigmas = sigmas.view(
            bs, self.gaussian_size, self.latent_size
        )  # (batch_size, gaussian_size, latent_size)
        sigmas = torch.exp(sigmas)  # ensure positive standard deviations

        pi = gmm_outs[
            :, 2 * stride : 2 * stride + self.gaussian_size
        ].contiguous()  # (batch_size, gaussian_size)
        pi = pi.view(bs, self.gaussian_size)
        logpi = F.log_softmax(pi, dim=-1)

        # rewards and terminal state indicators
        rs = gmm_outs[:, -2]  # (batch_size,)
        ds = gmm_outs[:, -1]  # (batch_size,)

        return mus, sigmas, logpi, rs, ds

    def to(self, device: torch.device) -> Self:
        self.device = device
        self.gmm_linear = self.gmm_linear.to(device)
        self.rnn = self.rnn.to(device)
        return super().to(device)


class MDNRNNCell(MDNRNN):
    """
    MDNRNN Cell for processing a single time step.

    This subclass modifies the MDNRNN to handle one step at a time, maintaining hidden states explicitly. Useful for scenarios where predictions are made sequentially.
    """

    def __init__(
        self,
        *,
        latent_size: int,
        action_size: int,
        hidden_size: int,
        gaussian_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(latent_size, action_size, hidden_size, gaussian_size, device)

        # replace the LSTM with an LSTMCell
        self.rnn = nn.LSTMCell(latent_size + action_size, hidden_size).to(device)

    def forward(
        self,
        action: torch.Tensor,
        latent: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """Performs a forward pass through the MDNRNNCell for a single time step."""
        hidden = tuple(h.to(self.device) for h in hidden)

        in_al = torch.cat(
            [action, latent], dim=1
        )  # (batch_size, action_size + latent_size)

        next_hidden = self.rnn(in_al, hidden)  # next_hidden is a tuple (h_next, c_next)
        out_rnn = next_hidden[0]  # (batch_size, hidden_size)

        out_full = self.gmm_linear(
            out_rnn
        )  # (batch_size, (2 * latent_size + 1) * gaussian_size + 2)

        stride = self.gaussian_size * self.latent_size

        mus = out_full[:, :stride].contiguous()  # (batch_size, stride)
        mus = mus.view(
            -1, self.gaussian_size, self.latent_size
        )  # (batch_size, gaussian_size, latent_size)

        sigmas = out_full[:, stride : 2 * stride].contiguous()  # (batch_size, stride)
        sigmas = sigmas.view(
            -1, self.gaussian_size, self.latent_size
        )  # (batch_size, gaussian_size, latent_size)
        sigmas = torch.exp(sigmas)

        pi = out_full[
            :, 2 * stride : 2 * stride + self.gaussian_size
        ].contiguous()  # (batch_size, gaussian_size)
        pi = pi.view(-1, self.gaussian_size)
        logpi = F.log_softmax(pi, dim=-1)

        r = out_full[:, -2]
        d = out_full[:, -1]

        return mus, sigmas, logpi, r, d, next_hidden

    def to(self, device: torch.device) -> Self:
        self.device = device
        self.rnn = self.rnn.to(device)
        return super().to(device)
