from typing import Dict, Any

import torch
from torch.distributions import Normal, Categorical, MixtureSameFamily, Independent


def sample_mdn(mus: torch.Tensor, sigmas: torch.Tensor, logpi: torch.Tensor):
    """Samples from the MDN output to get the next latent state using PyTorch distributions."""
    # categorical for mixture weights
    mixture_dist = Categorical(logits=logpi)
    # multivariate normal mixture components
    component_dist = Independent(Normal(loc=mus, scale=sigmas), 1)
    mixture_model = MixtureSameFamily(
        mixture_distribution=mixture_dist, component_distribution=component_dist
    )
    z = mixture_model.sample()  # (batch_size, latent_dim)
    return mixture_model, z


def gmm_loss(
    batch: torch.Tensor,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    logpi: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    """Computes the Gaussian Mixture Model (GMM) loss.

    More precisely, it computes minus the log probability of the batch under the GMM model described by mus, sigmas and pi.
    """
    # to improve numerical stability
    sigmas = torch.clamp(sigmas, min=1e-6)
    gmm, _ = sample_mdn(mus, sigmas, logpi)
    log_prob = gmm.log_prob(batch)
    if reduce:
        return -torch.mean(log_prob)
    return -log_prob


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_filename: str,
    best_model_filename: str,
):
    """Saves model and training parameters at checkpoint_filename. If is_best==True, also saves best_model_filename."""
    torch.save(state, checkpoint_filename)
    if is_best:
        torch.save(state, best_model_filename)
