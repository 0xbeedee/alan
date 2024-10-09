from typing import Dict, Any

import torch
from torch.distributions.normal import Normal


def gmm_loss(
    batch: torch.Tensor,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    logpi: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    """Computes the Gaussian Mixture Model (GMM) loss.

    More precisely, it computes minus the log probability of batch under the GMM model described by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch dimensions (several batch dimensions are useful when you have both a batch axis and a time step axis), gs the number of mixtures, and fs the number of features.
    """
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs, _ = torch.max(g_log_probs, dim=-1, keepdim=True)
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze(-1) + torch.log(probs)
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
