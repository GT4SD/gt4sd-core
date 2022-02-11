"""Model utilities."""


class KLAnnealer:
    """Annealer scaling KL weights (beta) linearly according to the number of epochs."""

    def __init__(
        self, kl_low: float, kl_high: float, n_epochs: int, start_epoch: int
    ) -> None:
        """Construct KLAnnealer.

        Args:
            kl_low: low KL weight.
            kl_high: high KL weight.
            n_epochs: number of epochs.
            start_epoch: starting epoch.
        """
        self.kl_low = kl_low
        self.kl_high = kl_high
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch
        self.kl = (self.kl_high - self.kl_low) / (self.n_epochs - self.start_epoch)

    def __call__(self, epoch: int) -> float:
        """Call the annealer.

        Args:
            epoch: current epoch number.

        Returns:
            the beta weight.
        """
        k = (epoch - self.start_epoch) if epoch >= self.start_epoch else 0
        beta = self.kl_low + k * self.kl
        if beta > self.kl_high:
            beta = self.kl_high
        return beta
