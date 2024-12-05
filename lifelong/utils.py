import torch
from torch import nn
from tianshou.data import Batch


# TODO this is a bit too simplistic...
def is_similar(
    obs_net: nn.Module,
    batch_obs1: Batch,
    batch_obs2: Batch,
    thresh: float = 0.90,
) -> torch.Tensor:
    latent1, latent2 = obs_net(batch_obs1), obs_net(batch_obs2)
    # TODO cosine similarity (after normalisation) could also be used here
    # TODO we could use other methods to go from [0, inf) to [0, 1] (e.g., sigmoids, RBF kernels, exp decay)
    distance = torch.norm(latent1 - latent2, p=2, dim=1, keepdim=True)  # (B, 1)
    # TODO very high similarity for some observations in the beginning, why?
    similarity = 1.0 / (1.0 + distance)

    return similarity > thresh
