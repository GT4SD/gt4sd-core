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
"""Core for Generative Flow Networks (https://arxiv.org/abs/2111.09266).
"""

from abc import abstractmethod
from argparse import ArgumentParser
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.distributions import Distribution

from gt4sd.frameworks.gflownet.train.core import GFNTrainer


class GFlowNetBaseModel(nn.Module):
    """Base model class. GFN is a generative policy at its core.
    pi(a' | s') - given the actual state (a partial molecular graph) select the best action (basic building block for molecules).
    p(s' | s, a) - transition distribution.
    The implementation for molecules relyes on graph-based models.
    The general idea is to start with the empty state, and add a basic building block at each (state, action) trajectory step.
    """

    steps: int
    trainer: GFNTrainer

    def __init__(self, name: str, data: Dict[str, str], *args, **kwargs) -> None:
        """Construct GFlowNetBaseModel.

        Args:
            name: model name.
            data: data name mappings.
        """
        super().__init__()
        self.name = name
        self.data = data

        self.policy = ""
        self.sampler = ""

    def forward(self, x: Any, *args, **kwargs) -> Any:
        """Forward pass in the model.

        Args:
            x: model input.

        Returns:
            model output.
        """
        return self._run_step(x)

    @abstractmethod
    def _run_step(self, x: Any, *args, **kwargs) -> Any:
        """Run a step in the model.

        Args:
            x: model input.

        Returns:
            model step output.
        """
        pass

    @abstractmethod
    def step(
        self,
        input_data: Any,
        target_data: Any,
        device: str = "cpu",
        current_epoch: int = 0,
        *args,
        **kwargs,
    ) -> Tuple[Any, Any, Any]:
        """Training step for the model.

        Args:
            input_data: input for the step.
            target_data: target for the step.
            device: string representing the device to use. Defaults to "cpu".
            current_epoch: current epoch. Defaults to 0.

        Returns:
            a tuple containing the step output, the loss and the logs for the module.
        """
        pass

    @abstractmethod
    def val_step(
        self,
        input_data: Any,
        target_data: Any,
        device: str = "cpu",
        current_epoch: int = 0,
        *args,
        **kwargs,
    ) -> Any:
        """Validation step for the model.

        Args:
            input_data: input for the step.
            target_data: target for the step.
            device: string representing the device to use. Defaults to "cpu".
            current_epoch: current epoch. Defaults to 0.

        Returns:
            a tuple containing the step output, the loss and the logs for the module.
        """
        pass

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser, name: str, *args, **kwargs
    ) -> ArgumentParser:
        """Adding to a parser model specific arguments.

        Args:
            parent_parser: patent parser.
            name: model name.

        Returns:
            updated parser.
        """
        return parent_parser


class GFLowNetSampler(GFlowNetBaseModel):
    """Autoencoder model class."""

    latent_size: int

    @abstractmethod
    def decode(self, z: Any, *args, **kwargs) -> Any:
        """Decode a latent space point.

        Args:
            z: latent point.

        Returns:
            decoded sample.
        """
        pass

    @abstractmethod
    def encode(self, x: Any, *args, **kwargs) -> Any:
        """Encode a sample.

        Args:
            x: input sample.

        Returns:
            latent encoding.
        """
        pass

    def encode_decode(self, x: Any, *args, **kwargs) -> Any:
        """Encode and decode a sample.

        Args:
            x: input sample.

        Returns:
            decoded sample.
        """
        z = self.encode(x)
        return self.decode(z)

    def inference(self, z: Any, *args, **kwargs) -> Any:
        """Run the model in inference mode.

        Args:
            z: sample.

        Returns:
            generated output.
        """
        return self.decode(z)

    def sample(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> Tuple[Distribution, Distribution, torch.Tensor]:
        """Sample a point from a given mean and average following a normal log-likelihood.

        Args:
            mu: mean tensor.
            log_var: log varian tensor.

        Returns:
            a tuple containing standard normal, localized normal and the sampled point.
        """
        std = torch.exp(log_var / 2.0)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z


class GFlowNetLoss(nn.Module):
    """Base loss class.
    Implements trajectory balance as proposed in https://arxiv.org/abs/2201.13259.
    Optionally we can include the standard TD flow loss in https://arxiv.org/abs/2106.04399
    """

    steps: int

    def __init__(self, name: str, *args, **kwargs) -> None:
        """
        Args:
            name: model name.
            data: data name mappings.
        """
        super().__init__()
        self.name = name


class GFlowNetReward(nn.Module):
    """Base model class."""

    steps: int

    def __init__(self, name: str, *args, **kwargs) -> None:
        """
        Args:
            name: model name.
            data: data name mappings.
        """
        super().__init__()
        self.name = name


class GFlowNetPolicy(nn.Module):
    """Base model class."""

    steps: int

    def __init__(self, name: str, *args, **kwargs) -> None:
        """
        Args:
            name: model name.
            data: data name mappings.
        """
        super().__init__()
        self.name = name
