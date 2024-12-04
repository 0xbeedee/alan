import torch


# TODO this is a bit too simplistic...
def is_similar(
    batch_latent_obs1: torch.Tensor,
    batch_latent_obs2: torch.Tensor,
    thresh: float = 0.90,
) -> torch.Tensor:
    # TODO cosine similarity (after normalisatin) could also be used here
    # TODO we could use other methods to go from [0, inf) to [0, 1] (e.g., sigmoids, RBF kernels, exp decay)
    distance = torch.norm(
        batch_latent_obs1 - batch_latent_obs2, p=2, dim=1, keepdim=True
    )  # (B, 1)
    # TODO I get a very high similarity for some observations, especially in the beginning, why?
    similarity = 1.0 / (1.0 + distance)

    return similarity > thresh
