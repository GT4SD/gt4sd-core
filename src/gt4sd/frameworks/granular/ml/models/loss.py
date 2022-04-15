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
"""Losses for granular models."""

from typing import Any, Dict

import torch
from torch import nn


class MSLELossNegMix9(nn.Module):
    """MSLE loss negative mix 9."""

    def __init__(self) -> None:
        """Initialize the loss."""
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> Any:
        """Forward pass in the loss.

        Args:
            prediction: predictions.
            target: groundtruth.

        Returns:
            loss value.
        """
        pred2 = prediction.clone()
        true2 = target.clone()
        pred2[pred2 < 0] = 0
        pred2 = pred2 + 1e-6
        true2 = true2 + 1e-6
        pred3 = prediction.clone()
        true3 = target.clone()
        pred3[target < 0.0001] = 0
        true3[target < 0.0001] = 0
        pred4 = prediction.clone()
        true4 = target.clone()
        pred4[pred4 > 0] = 0
        true4[true4 < 2] = 0
        l4_ = self.mse(pred4, true4)
        l1_ = self.mse(pred3 / (0.001 + true3), true3 / (0.001 + true3))
        l2_ = self.mse(prediction, target)
        l3_ = self.mse(torch.log(pred2), torch.log(true2))
        l0_ = torch.abs(l1_ * 0.1 + l2_ * 1.0 + l3_ * 1.0e-5 + l4_ * 10.0)
        return l0_


class MSLELossNegMix91(nn.Module):
    """MSLE loss negative mix 91."""

    def __init__(self):
        """Initialize the loss."""
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.mae = nn.L1Loss(reduction="sum")

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> Any:
        """Forward pass in the loss.

        Args:
            prediction: predictions.
            target: groundtruth.

        Returns:
            loss value.
        """
        pred2 = prediction.clone()
        true2 = target.clone()
        pred2[pred2 < 0] = 0
        pred2 = pred2 + 1e-6
        true2 = true2 + 1e-6
        l1_ = self.mae(prediction, target)
        l2_ = self.mse(prediction, target)
        l3_ = self.mse(torch.log(pred2), torch.log(true2))
        l0_ = torch.abs(l1_ * 0.3 + l2_ * 1.0 + l3_ * 1.0e-5)
        return l0_


class MseWithNans(nn.Module):
    """MSE with NaNs handling."""

    def __init__(self):
        """Initialize the loss."""
        super().__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> Any:
        """Forward pass in the loss.

        Args:
            prediction: predictions.
            target: groundtruth.

        Returns:
            loss value.
        """
        mask = torch.isnan(target)
        out = (prediction[~mask] - target[~mask]) ** 2
        loss = out.mean()
        return loss


class MaeWithNans(nn.Module):
    """MAE with NaNs handling."""

    def __init__(self):
        """Initialize the loss."""
        super().__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> Any:
        """Forward pass in the loss.

        Args:
            prediction: predictions.
            target: groundtruth.

        Returns:
            loss value.
        """
        mask = torch.isnan(target)
        if sum(mask) == len(target):
            return torch.tensor(0).type_as(prediction)
        out = abs((prediction[~mask] - target[~mask]))
        loss = sum(out) / len(prediction[~mask])
        return loss


class MSLELossNegMix92(nn.Module):
    """MSLE loss negative mix 92."""

    def __init__(self):
        """Initialize the loss."""
        super().__init__()
        self.mae = nn.L1Loss(reduction="mean")

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> Any:
        """Forward pass in the loss.

        Args:
            prediction: predictions.
            target: groundtruth.

        Returns:
            loss value.
        """
        l1_ = self.mae(prediction, target)
        mask = target.ge(0.001)
        l2_ = self.mae(prediction[mask], target[mask])
        l0_ = torch.abs(l1_ + l2_ * 10.0)
        return l0_


class MSLELossNegMix99(nn.Module):
    """MSLE loss negative mix 99."""

    def __init__(self):
        """Initialize the loss."""
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> Any:
        """Forward pass in the loss.

        Args:
            prediction: predictions.
            target: groundtruth.

        Returns:
            loss value.
        """
        mask = torch.isnan(target)
        out = (prediction[~mask] - target[~mask]) ** 2
        loss = out.mean()
        return loss


LOSS_FACTORY: Dict[str, nn.Module] = {
    "mse": nn.MSELoss(),
    "mse-sum": nn.MSELoss(reduction="sum"),
    "mse-mean": nn.MSELoss(reduction="mean"),
    "mse-with-nans": MseWithNans(),
    "msewithnans": MseWithNans(),
    "mae": nn.L1Loss(),
    "mae-sum": nn.L1Loss(reduction="sum"),
    "mae-mean": nn.L1Loss(reduction="mean"),
    "mae-with-nans": MaeWithNans(),
    "maewithnans": MaeWithNans(),
    "bce": nn.BCELoss(),
    "bce-with-logits": nn.BCEWithLogitsLoss(),
    "bcewl": nn.BCEWithLogitsLoss(),
    "loss9": MSLELossNegMix9(),
    "l9": MSLELossNegMix9(),
    "msle-neg-mix-9": MSLELossNegMix9(),
    "loss91": MSLELossNegMix91(),
    "l91": MSLELossNegMix91(),
    "msle-neg-mix-91": MSLELossNegMix91(),
    "loss92": MSLELossNegMix92(),
    "l92": MSLELossNegMix92(),
    "msle-neg-mix-92": MSLELossNegMix92(),
    "loss99": MSLELossNegMix99(),
    "l99": MSLELossNegMix99(),
    "msle-neg-mix-99": MSLELossNegMix99(),
    "crossentropyloss": nn.CrossEntropyLoss(),
    "ce": nn.CrossEntropyLoss(),
}
