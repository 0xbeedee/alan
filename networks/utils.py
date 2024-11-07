import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal


def reparameterise(
    mu: torch.Tensor, logsigma: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Uses the reparameterisation trick to obtain an observation in latent space, given the means and logsigmas."""
    # from https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/
    sigma = F.softplus(logsigma) + eps
    scale_tril = torch.diag_embed(sigma)
    dist = MultivariateNormal(mu, scale_tril=scale_tril)
    # we'll need the dist for training the VAE
    return dist.rsample(), dist


class Crop(nn.Module):
    """Helper class to provide the agent with an egocentric representation of NetHack."""

    def __init__(self, height: int, width: int, height_target: int, width_target: int):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        width_grid = self._step_to_range(2 / (self.width - 1), self.width_target)[
            None, :
        ].expand(self.height_target, -1)
        height_grid = self._step_to_range(2 / (self.height - 1), height_target)[
            :, None
        ].expand(-1, self.width_target)

        # "clone" necessary, https://github.com/pytorch/pytorch/issues/34880
        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def forward(self, inputs: torch.Tensor, coordinates: torch.Tensor):
        """Calculates centered crop around given x,y coordinates (usually the agent's position)."""
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        inputs = inputs[:, None, :, :].float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
            .squeeze(1)
            .long()
        )

    def _step_to_range(self, delta: float, num_steps: int):
        """Range of `num_steps` integers with distance `delta` centered around zero."""
        return delta * torch.arange(-num_steps // 2, num_steps // 2)
