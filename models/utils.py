from typing import Dict, Any

import torch
from torch.distributions import Categorical, Normal, MixtureSameFamily, Independent


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
    # categorical for mixture weights
    mix = Categorical(logits=logpi)
    # multivariate normal mixture components
    comp = Independent(Normal(loc=mus, scale=sigmas), 1)
    gmm = MixtureSameFamily(mix, comp)

    # log probability of the batch under the GMM
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
