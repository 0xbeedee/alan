import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Self


class MDNRNN(nn.Module):
    """Mixture Density Network RNN (MDNRNN) model for multiple forward steps.

    Adapted from https://github.com/ctallec/world-models/blob/master/models/mdrnn.py.
    """

    def __init__(
        self,
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

        self.gmm_linear = nn.Linear(
            hidden_size, (2 * latent_size + 1) * gaussian_size + 2
        ).to(device)
        self.rnn = nn.LSTM(latent_size + action_size, hidden_size).to(device)

    def forward(
        self, actions: torch.Tensor, latents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Multiple steps forward.

        Args:
            actions (torch.Tensor): Tensor of shape (SEQ_LEN, BSIZE, ASIZE).
            latents (torch.Tensor): Tensor of shape (SEQ_LEN, BSIZE, LSIZE).

        Returns:
            Tuple containing:
                - mus (torch.Tensor): Shape (SEQ_LEN, BSIZE, N_GAUSS, LSIZE).
                - sigmas (torch.Tensor): Shape (SEQ_LEN, BSIZE, N_GAUSS, LSIZE).
                - logpi (torch.Tensor): Shape (SEQ_LEN, BSIZE, N_GAUSS).
                - rs (torch.Tensor): Shape (SEQ_LEN, BSIZE).
                - ds (torch.Tensor): Shape (SEQ_LEN, BSIZE).
        """
        actions = actions.to(self.device)
        latents = latents.to(self.device)

        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)

        stride = self.gaussian_size * self.latent_size

        mus = gmm_outs[:, :, :stride].contiguous()
        mus = mus.view(seq_len, bs, self.gaussian_size, self.latent_size)

        sigmas = gmm_outs[:, :, stride : 2 * stride].contiguous()
        sigmas = sigmas.view(seq_len, bs, self.gaussian_size, self.latent_size)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride : 2 * stride + self.gaussian_size].contiguous()
        pi = pi.view(seq_len, bs, self.gaussian_size)
        logpi = F.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]

        ds = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds

    def to(self, device: torch.device) -> Self:
        self.device = device
        self.gmm_linear = self.gmm_linear.to(device)
        self.rnn = self.rnn.to(device)
        return super().to(device)


class MDNRNNCell(MDNRNN):
    """MDNRNN model for a single forward step."""

    def __init__(
        self,
        latent_size: int,
        action_size: int,
        hidden_size: int,
        gaussian_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(latent_size, action_size, hidden_size, gaussian_size, device)
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
        """One step forward.

        Args:
            action (torch.Tensor): Tensor of shape (BSIZE, ASIZE).
            latent (torch.Tensor): Tensor of shape (BSIZE, LSIZE).
            hidden (Tuple[torch.Tensor, torch.Tensor]): Tuple of tensors each of shape (BSIZE, HSIZE).

        Returns:
            Tuple containing:
                - mus (torch.Tensor): Shape (BSIZE, N_GAUSS, LSIZE).
                - sigmas (torch.Tensor): Shape (BSIZE, N_GAUSS, LSIZE).
                - logpi (torch.Tensor): Shape (BSIZE, N_GAUSS).
                - r (torch.Tensor): Shape (BSIZE).
                - d (torch.Tensor): Shape (BSIZE).
                - next_hidden (Tuple[torch.Tensor, torch.Tensor]): Next hidden state.
        """
        action = action.to(self.device)
        latent = latent.to(self.device)
        hidden = tuple(h.to(self.device) for h in hidden)

        in_al = torch.cat([action, latent], dim=1)

        next_hidden = self.rnn(in_al, hidden)
        out_rnn = next_hidden[0]

        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussian_size * self.latent_size

        mus = out_full[:, :stride].contiguous()
        mus = mus.view(-1, self.gaussian_size, self.latent_size)

        sigmas = out_full[:, stride : 2 * stride].contiguous()
        sigmas = sigmas.view(-1, self.gaussian_size, self.latent_size)
        sigmas = torch.exp(sigmas)

        pi = out_full[:, 2 * stride : 2 * stride + self.gaussian_size].contiguous()
        pi = pi.view(-1, self.gaussian_size)
        logpi = F.log_softmax(pi, dim=-1)

        r = out_full[:, -2]

        d = out_full[:, -1]

        return mus, sigmas, logpi, r, d, next_hidden

    def to(self, device: torch.device) -> Self:
        super().to(device)
        return self
