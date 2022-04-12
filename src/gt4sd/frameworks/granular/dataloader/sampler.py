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
Sampler implementation.

Reimplemented starting from: https://github.com/ncullen93/torchsample/blob/ea4d1b3975f68be0521941e733887ed667a1b46e/torchsample/samplers.py.
The main reason for reimplementation is to avoid to add a dependency and to control better the logger.
"""

import logging
from typing import Iterator

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class StratifiedSampler(Sampler):
    """Implementation of a sampler for tensors based on scikit-learn StratifiedShuffleSplit."""

    def __init__(
        self, targets: torch.Tensor, batch_size: int, test_size: float = 0.5
    ) -> None:
        """Construct a StratifiedSampler.

        Args:
            targets: targets tensor.
            batch_size: size of the batch.
            test_size: proportion of samples in the test set. Defaults to 0.5.
        """
        self.targets = targets
        self.number_of_splits = int(self.targets.size(0) / batch_size)
        self.test_size = test_size

    def gen_sample_array(self) -> np.ndarray:
        """Get sample array.

        Returns:
            sample array.
        """
        splitter = StratifiedShuffleSplit(
            n_splits=self.number_of_splits, test_size=self.test_size
        )
        data_placeholder = torch.randn(self.targets.size(0), 2).numpy()
        targets = self.targets.numpy()
        splitter.get_n_splits(data_placeholder, targets)
        train_index, test_index = next(splitter.split(data_placeholder, targets))
        return np.hstack([train_index, test_index])

    def __iter__(self) -> Iterator[np.ndarray]:
        """Get an iterator over the sample array.

        Returns:
            sample array iterator.

        Yields:
            a sample array.
        """
        return iter(self.gen_sample_array())

    def __len__(self) -> int:
        """Length of the sampler.

        Returns:
            the sampler length.
        """
        return len(self.targets)
