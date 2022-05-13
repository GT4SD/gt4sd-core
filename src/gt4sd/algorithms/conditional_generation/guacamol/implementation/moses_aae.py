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
"""Moses AAE implementation."""

import argparse

from guacamol_baselines.moses_baselines.aae_distribution_learning import AaeGenerator


class AAE:
    def __init__(
        self,
        model_path: str,
        model_config_path: str,
        vocab_path: str,
        n_samples: int,
        n_batch: int,
        max_len: int,
        device: str = "cpu",
    ):
        """Initialize AAE.

        Args:
            model_path: path from where to load the model.
            model_config_path: path from where to load the model config.
            vocab_path: path from where to load the vocab.
            n_samples: number of samples to sample.
            n_batch: size of the batch.
            max_len: max length of SMILES.
            device: device used for computation. Defaults to cpu.
        """
        self.config = argparse.Namespace(
            model_load=model_path,
            config_load=model_config_path,
            vocab_load=vocab_path,
            n_samples=n_samples,
            n_batch=n_batch,
            max_len=max_len,
            device=device,
        )

    def get_generator(self) -> AaeGenerator:
        """Create an instance of the AaeGenerator.

        Returns:
            an instance of AaeGenerator.
        """
        optimiser = AaeGenerator(self.config)
        return optimiser
