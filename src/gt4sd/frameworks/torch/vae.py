"""pytorch utils for VAEs."""

import torch


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Applies reparametrization trick to obtain sample from latent space.

    Args:
        mu: the latent means of shape batch_size x latent_size.
        logvar: latent log variances, shape batch_size x latent_size.

    Returns:
        torch.Tensor: sampled Z from the latent distribution.
    """
    return torch.randn_like(mu).mul_(torch.exp(0.5 * logvar)).add_(mu)  # type:ignore
