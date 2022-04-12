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
"""MolecularAI Implementation of sample generation, randomizing scaffolds as well as fetching unique sample sequences

The source of this file is
https://raw.githubusercontent.com/MolecularAI/Reinvent/982b26dd6cfeb8aa84b6d7e4a8c2a7edde2bad36/running_modes/lib_invent/rl_actions/sample_model.py
and it was only minimally changed. See README.md.
"""

__copyright__ = "Copyright 2021, MolecularAI"
__license__ = "Apache 2.0"

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import torch.utils.data as tud
from reinvent_chemistry import Conversions
from reinvent_chemistry.library_design import AttachmentPoints, BondMaker
from reinvent_chemistry.utils import get_indices_of_unique_smiles
from reinvent_models.lib_invent.models import dataset as md

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class SampledSequencesDTO:
    scaffold: str
    decoration: str
    nll: float


class ReinventBase:
    def __init__(
        self, model, batch_size: int, logger=None, randomize=False, sample_uniquely=True
    ):
        """
        Creates an instance of SampleModel.
        :params model: A model instance (better in scaffold_decorating mode).
        :params batch_size: Batch size to use.
        :return:
        """
        self.model = model
        self._batch_size = batch_size
        self._bond_maker = BondMaker()
        self._attachment_points = AttachmentPoints()
        self._randomize = randomize
        self._conversions = Conversions()
        self._sample_uniquely = sample_uniquely

    def get_dataloader(self, scaffold_list: List[str]) -> tud.DataLoader:
        """
        Get a dataloader for the list of scaffolds to use with reinvent.
        NOTE: This method was factored out of the `run` method from the original source.
        :params scaffold_list: A list of scaffold SMILES.
        :return: An instance of a torch dataloader.
        """
        scaffold_list = (
            self._randomize_scaffolds(scaffold_list)
            if self._randomize
            else scaffold_list
        )
        clean_scaffolds = [
            self._attachment_points.remove_attachment_point_numbers(scaffold)
            for scaffold in scaffold_list
        ]
        dataset = md.Dataset(
            clean_scaffolds,
            self.model.vocabulary.scaffold_vocabulary,
            self.model.vocabulary.scaffold_tokenizer,
        )
        dataloader = tud.DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            collate_fn=md.Dataset.collate_fn,
        )
        return dataloader

    def run(self, scaffold_list: List[str]) -> List[SampledSequencesDTO]:
        """
        Samples the model for the given number of SMILES.
        NOTE: this method was slightly adapted from the original source.
        :params scaffold_list: A list of scaffold SMILES.
        :return: A list of SampledSequencesDTO.
        """

        dataloader = self.get_dataloader(scaffold_list)

        sampled_sequences = []
        for batch in dataloader:

            for _ in range(self._batch_size):
                scaffold_seqs, scaffold_seq_lengths = batch
                packed = self.model.sample_decorations(
                    scaffold_seqs, scaffold_seq_lengths
                )
                for scaffold, decoration, nll in packed:
                    sampled_sequences.append(
                        SampledSequencesDTO(scaffold, decoration, nll)
                    )

            if self._sample_uniquely:
                sampled_sequences = self._sample_unique_sequences(sampled_sequences)

        return sampled_sequences

    def _sample_unique_sequences(
        self, sampled_sequences: List[SampledSequencesDTO]
    ) -> List[SampledSequencesDTO]:
        strings = [
            "".join([ss.scaffold, ss.decoration])
            for index, ss in enumerate(sampled_sequences)
        ]
        unique_idxs = get_indices_of_unique_smiles(strings)
        sampled_sequences_np = np.array(sampled_sequences)
        unique_sampled_sequences = sampled_sequences_np[unique_idxs]
        return unique_sampled_sequences.tolist()

    def _randomize_scaffolds(self, scaffolds: List[str]):
        scaffold_mols = [
            self._conversions.smile_to_mol(scaffold) for scaffold in scaffolds
        ]
        randomized = [self._bond_maker.randomize_scaffold(mol) for mol in scaffold_mols]
        return randomized
