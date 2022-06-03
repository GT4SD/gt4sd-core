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
"""Implementation of Reinvent conditional generators."""

import logging
import os
from typing import List, NamedTuple, Optional, Set, Tuple

from reinvent_models.lib_invent.models.model import DecoratorModel

from .reinvent_core.core import ReinventBase, SampledSequencesDTO

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SampledTuple(NamedTuple):
    scaffold: str
    decoration: str
    nll: float


class ReinventConditionalGenerator(ReinventBase):
    def __init__(
        self,
        resources_path: str,
        batch_size: int,
        randomize: bool,
        sample_uniquely: bool,
        max_sequence_length: int,
    ):
        """Initialize Reinvent.

        Args:
            resources_path: path where to load hypothesis, candidate labels and, optionally, the model.
            batch_size: number of samples to generate per scaffold.
            randomize: randomize the scaffolds if set to true.
            sample_uniquely: generate unique sample sequences if set to true.
            max_sequence_length: maximum length of the generated sequences.
        """
        self.resources_path = resources_path
        self.batch_size = batch_size
        self.randomize = randomize
        self.sample_uniquely = sample_uniquely
        self.model_path = os.path.join(self.resources_path, "model.prior")
        self.target: Optional[str] = None

        if not os.path.isfile(self.model_path):
            logger.debug("reinvent model files does not exist locally")
            raise OSError(f"artifacts file {self.model_path} does not exist locally")

        self.model = DecoratorModel.load_from_file(path=self.model_path)
        self.model.max_sequence_length = max_sequence_length
        super().__init__(
            self.model, self.batch_size, self.randomize, self.sample_uniquely
        )

    def sample_unique_sequences(self, sampled_sequences: List[Tuple]) -> List[Tuple]:
        """
        Samples the model for the given number of SMILES.

        Args:
            scaffold_list: A list of SampledTuple.
        Returns:
            A list of SampledTuple.
        """
        sequences = [
            SampledSequencesDTO(scaffold, decoration, nll)
            for scaffold, decoration, nll in sampled_sequences
        ]
        logger.info("getting unique sample sequences from generated samples")
        return [
            (sample.scaffold, sample.decoration, sample.nll)
            for sample in self._sample_unique_sequences(sequences)
        ]

    def generate_sampled_tuples(self, scaffold: str) -> Set[SampledTuple]:
        """
        Samples the model for the given number of SMILES.
        Args:
            scaffold_list: A list of scaffold SMILES.
        Returns:
            A Set of SampledTuple.
        """
        if self.target != scaffold:
            self.target = scaffold
            batch = next(iter(self.get_dataloader([scaffold])))
            logger.info("initialization of the dataloader")
            scaffold_seqs, scaffold_seq_lengths = batch
            self.scaffold_seqs = scaffold_seqs.expand(
                self.batch_size - 1, scaffold_seqs.shape[1]
            )
            self.scaffold_seq_lengths = scaffold_seq_lengths.expand(self.batch_size - 1)
        logger.info("started generating samples with an nll score value")
        sampled_sequences = list(
            self.model.sample_decorations(self.scaffold_seqs, self.scaffold_seq_lengths)
        )
        if self.sample_uniquely:
            sampled_sequences = self.sample_unique_sequences(sampled_sequences)

        return set(
            [
                SampledTuple(scaffold, decoration, nll)
                for scaffold, decoration, nll in sampled_sequences
            ]
        )

    def generate_samples(self, scaffold: str) -> Set[str]:
        """
        Samples the model for the given number of SMILES.

        Args:
            scaffold: A scaffold SMILES.
        Returns:
            A Set of SMILES representing molecules.
        """
        return set(
            molecule for _, molecule, _ in self.generate_sampled_tuples(scaffold)
        )
