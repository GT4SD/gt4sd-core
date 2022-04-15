#
# MIT License
#
# Copyright (c) 2022 GT4SD team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
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
