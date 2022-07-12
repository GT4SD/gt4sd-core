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
"""Implementation of MoLeR conditional generators."""

import logging
from itertools import cycle, islice
from typing import List

from rdkit import Chem
from molecule_generation import VaeWrapper

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MoLeRGenerator:
    """Interface for MoLeR generator."""

    def __init__(
        self,
        resources_path: str,
        scaffolds: str,
        num_samples: int,
        beam_size: int,
        seed: int,
        num_workers: int,
    ) -> None:
        """Instantiate a MoLeR generator.

        Args:
            resources_path: path to the resources for model loading.
            scaffolds: scaffolds as '.'-separated SMILES. If empty, no scaffolds are used.
            num_samples: Number of molecules to sample per call.
            beam_size: beam size to use during decoding.
            seed: seed used for random number generation.
            num_workers: number of workers used for generation.

        Raises:
            RuntimeError: in the case extras are disabled.
        """
        # loading artifacts
        self.resources_path = resources_path
        self.scaffolds = scaffolds
        self.num_samples = num_samples
        self.beam_size = beam_size
        self.num_workers = num_workers
        self._seed = seed

    def generate(self) -> List[str]:
        """Sample molecules using MoLeR.

        Returns:
            sampled molecule (SMILES).
        """
        # process scaffolds
        valid_scaffolds = [
            scaffold
            for scaffold in self.scaffolds.split(".")
            if Chem.MolFromSmiles(scaffold) is not None
        ]
        # generate molecules
        logger.info("running MoLeR...")
        with VaeWrapper(
            self.resources_path,
            beam_size=self.beam_size,
            seed=self._seed,
            num_workers=self.num_workers,
        ) as model:
            latents = model.sample_latents(self.num_samples)
            scaffolds = list(islice(cycle(valid_scaffolds), self.num_samples))
            samples = model.decode(
                latents=latents,
                scaffolds=scaffolds if len(scaffolds) == self.num_samples else None,
            )
        # offset seed to guarantee uniqueness
        self._seed += 1
        logger.info("MoLeR run completed")
        return samples
