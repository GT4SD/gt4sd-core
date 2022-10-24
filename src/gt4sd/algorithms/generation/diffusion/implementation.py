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
"""
Implementation details for huggingface diffusers generation algorithms.

Parts of the implementation inspired by: https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py.
"""

import os
from typing import Any, Dict, List, Optional, Union

import importlib_metadata
import numpy as np
import torch
from diffusers import (
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    LDMPipeline,
    LDMTextToImagePipeline,
    LMSDiscreteScheduler,
    ScoreSdeVePipeline,
    ScoreSdeVeScheduler,
    StableDiffusionPipeline,
)
from packaging import version

from ....frameworks.torch import device_claim
from .geodiff.core import GeoDiffPipeline

DIFFUSERS_VERSION_LT_0_6_0 = version.parse(
    importlib_metadata.version("diffusers")
) < version.parse("0.6.0")


def set_seed(seed: int = 42) -> None:
    """Set seed for all random number generators.

    Args:
        seed: seed to set. Defaults to 42.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)  # type:ignore


MODEL_TYPES = {
    "diffusion": DDPMPipeline,
    "diffusion_implicit": DDIMPipeline,
    "latent_diffusion": LDMPipeline,
    "latent_diffusion_conditional": LDMTextToImagePipeline,
    "stable_diffusion": StableDiffusionPipeline,
    "score_sde": ScoreSdeVePipeline,
    "geodiff": GeoDiffPipeline,
}

SCHEDULER_TYPES = {
    "ddpm": DDPMScheduler,
    "ddim": DDIMScheduler,
    "discrete": LMSDiscreteScheduler,
    "continuous": ScoreSdeVeScheduler,
}


class Generator:
    """Implementation of a generator."""

    def __init__(
        self,
        resources_path: str,
        model_type: str,
        model_name: str,
        scheduler_type: str,
        auth_token: bool = True,
        prompt: Optional[Union[str, Dict[str, Any]]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """A Diffusers generation algorithm.

        Args:
            resources_path: path to the cache.
            model_type: type of the model.
            model_name: name of the model weights/version.
            scheduler_type: type of the schedule.
            auth_token: authentication token for private models.
            prompt: target for conditional generation.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """
        self.device = device_claim(device)
        self.resources_path = resources_path
        self.model_type = model_type
        self.model_name = model_name
        self.scheduler_type = scheduler_type
        self.prompt = prompt
        self.auth_token = auth_token
        self.load_model()

    def load_model(self) -> None:
        """Load a pretrained diffusion generative model."""

        try:
            model_class = MODEL_TYPES[self.model_type]
        except KeyError:
            raise KeyError(f"model type: {self.model_type} not supported")

        if (
            os.path.exists(self.resources_path)
            and len(os.listdir(self.resources_path)) > 0
        ):
            model_name_or_path = self.resources_path
        else:
            model_name_or_path = self.model_name

        if self.model_type == "stable_diffusion":
            self.model = model_class.from_pretrained(
                model_name_or_path,
                use_auth_token=self.auth_token,
            )
        else:
            self.model = model_class.from_pretrained(model_name_or_path)

        self.model.to(self.device)

    def sample(self, number_samples: int = 1) -> List[Any]:
        """Sample images with optional conditioning.

        Args:
            number_samples: number of images to generate.
        Returns:
            generated samples.
        """
        # if prompt is provided, use it
        if self.prompt:
            item = self.model(batch_size=number_samples, prompt=self.prompt)
        else:
            item = self.model(batch_size=number_samples)

        # To support old diffusers versions (<0.6.0)
        if DIFFUSERS_VERSION_LT_0_6_0 or self.model_type in ["geodiff"]:
            item = item["sample"]
        else:
            item = item.images

        return item
