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
"""Implementation of the zero-shot classifier."""

import json
import logging
import os
from typing import List, Optional, Union

import torch
from transformers import pipeline

from ....frameworks.torch import device_claim

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ZeroShotClassifier:
    """
    Zero-shot classifier based on the HuggingFace pipeline leveraging MLNI.
    """

    def __init__(
        self,
        resources_path: str,
        model_name: str,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """Initialize ZeroShotClassifier.

        Args:
            resources_path: path where to load hypothesis, candidate labels and, optionally, the model.
            model_name: name of the model to load from the cache or download from HuggingFace.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """
        device = device_claim(device)
        self.device = -1 if device.type == "cpu" else int(device.type.split(":")[1])
        self.resources_path = resources_path
        self.model_name = model_name
        self.load_pipeline()

    def load_pipeline(self) -> None:
        """Load zero shot classification MLNI pipeline."""
        metadata_filepath = os.path.join(self.resources_path, "metadata.json")
        if os.path.exists(metadata_filepath):
            with open(metadata_filepath) as fp:
                metadata = json.load(fp)
                self.labels = metadata["labels"]
                self.hypothesis_template = metadata["hypothesis_template"]
            self.model_name_or_path = os.path.join(self.resources_path, self.model_name)
            if not os.path.exists(self.model_name_or_path):
                logger.info(
                    f"no model named {self.model_name_or_path} in cache, using HuggingFace"
                )
                self.model_name_or_path = self.model_name
        else:
            message = f"could not retrieve the MLNI pipeline from the cache: {metadata_filepath} does not exists!"
            logger.error(message)
            raise ValueError(message)
        self.model = pipeline(
            "zero-shot-classification",
            model=self.model_name_or_path,
            device=self.device,
        )

    def predict(self, text: str) -> List[str]:
        """Get sorted classification labels.

        Args:
            text: text to classify.

        Returns:
            labels sorted by score from highest to lowest.
        """
        return self.model(
            text,
            candidate_labels=self.labels,
            hypothesis_template=self.hypothesis_template,
        )["labels"]
