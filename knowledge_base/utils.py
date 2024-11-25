import torch


# TODO this is a bit too simplistic...
def is_similar(
    latent_obs1: torch.Tensor, latent_obs2: torch.Tensor, thresh: float = 0.90
) -> bool:
    # TODO cosine similarity (after normalisatin) could also be used here
    # TODO we could use other methods to go from [0, inf) to [0, 1] (e.g., sigmoids, RBF kernels, exp decay)
    similarity = 1.0 / (1.0 + torch.dist(latent_obs1, latent_obs2, p=2))

    if similarity > thresh:
        return True
    return False
