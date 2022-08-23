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

import logging
import os
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers import (
    DDIMPipeline,
    DDPMPipeline,
    DiffusionPipeline,
    LMSContinuousScheduler,
    LMSContinuousSchedulerWithDiscrete,
    LMSDiscreteScheduler,
    PNDMPipeline,
    StableDiffusionPipeline,
)

from ....frameworks.torch import device_claim

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
    "pndmp": PNDMPipeline,
    "latent_diffusion": DiffusionPipeline,
    "stable_diffusion": StableDiffusionPipeline,
}

SCHEDULER_TYPES = {
    "discrete": LMSDiscreteScheduler,
    "continuous": LMSContinuousScheduler,
    "continuous_with_discrete": LMSContinuousSchedulerWithDiscrete,
}


class Generator:
    """Implementation of a generator."""

    def __init__(
        self,
        resources_path: str,
        model_type: str,
        model_name: str,
        scheduler_type: str,
        temperature: float,
        repetition_penalty: float,
        k: int,
        p: float,
        prefix: str,
        number_of_sequences: int,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """An HuggingFace Difffuser generation algorithm.

        Args:
            resources_path: path to the cache.
            model_type: type of the model.
            model_name: name of the model weights/version.
            scheduler_type: type of the schedule.
            temperature: temperature for sampling, the lower the greedier the sampling.
            repetition_penalty: primarily useful for CTRL model, where 1.2 should be used.
            k: number of top-k probability token to keep.
            p: only tokens with cumulative probabilities summing up to this value are kept.
            prefix: text defining context provided prior to the prompt.
            number_of_sequences: number of generated sequences.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """
        self.device = device_claim(device)
        self.resources_path = resources_path
        self.model_type = model_type
        self.model_name = model_name
        self.scheduler_type = scheduler_type
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.k = k
        self.p = p
        self.prefix = prefix
        self.number_of_sequences = number_of_sequences
        self.load_model()

    def load_model(self) -> None:
        """Load a pretrained diffusion generative model."""

        try:
            model_class = MODEL_TYPES[self.model_type]

            noise_scheduler_class = SCHEDULER_TYPES[self.scheduler_type]

        except KeyError:
            raise KeyError(f"model type: {self.model_type} not supported")

        if (
            os.path.exists(self.resources_path)
            and len(os.listdir(self.resources_path)) > 0
        ):
            model_name_or_path = self.resources_path
        else:
            model_name_or_path = self.model_name

        self.noise_scheduler = noise_scheduler_class.from_pretrained(  # type:ignore
            model_name_or_path
        )

        self.model = model_class.from_pretrained(model_name_or_path)
        self.model.to(self.device)

    def sample(self) -> List[str]:
        """Sample text snippets.

        Returns:
            generated text snippets.
        """

        output = self.model()["sample"]
        return output

    def conditional_sample(self, cond) -> List[str]:
        """Sample text snippets.

        Args:
            cond: text to condition the model on.
        Returns:
            generated text snippets.
        """

        output = self.model(cond)["sample"]
        return output

    def conditional_text_sample(self, cond, prompt) -> List[str]:
        """Sample text snippets.

        Args:
            cond: text to condition the model on.
            prompt: text to condition the model on.
        Returns:
            generated text snippets.
        """

        output = self.model(cond, prompt)["sample"]
        return output
