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
"""Implementation of PaccMann^RL conditional generators."""

import logging
from typing import List

import torch
from rdkit import Chem

from ...conditional_generation.paccmann_rl.core import (
    PaccMannRL,
    PaccMannRLProteinBasedGenerator,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PaccMannVaeDefaultGenerator:
    """
    Molecular generator as implemented in https://doi.org/10.1016/j.isci.2021.102269
    """

    def __init__(
        self,
        temperature: float = 1.4,
        batch_size: int = 32,
        algorithm_version: str = "v0",
        generated_length: int = 100,
    ) -> None:
        """
        Initialize the generator.

        Args:
            batch_size: batch size used for generation.
            algorithm_version: algorithm version for the PaccMannRLProteinBasedGenerator.
                NOTE: Only the decoder of that model is used here.
            temperature: temperature for the sampling. Defaults to 1.4.
            generated_length: maximum length of the generated molecules.
                Defaults to 100.
        """
        self.configuration = PaccMannRLProteinBasedGenerator(
            algorithm_version=algorithm_version,
            temperature=temperature,  # type: ignore
            generated_length=generated_length,  # type: ignore
            batch_size=batch_size,  # type: ignore
        )
        self.batch_size = batch_size

        self.algorithm = PaccMannRL(configuration=self.configuration, target="")
        self.model = self.configuration.get_conditional_generator(
            self.algorithm.local_artifacts
        )

    def generate(self) -> List[str]:
        """
        Generate a given number of samples (molecules) from a given protein.

        Args:
            number_of_molecules: number of molecules to sample.

        Returns:
            list of SMILES generated.
        """
        smiles: List = []
        while len(smiles) < self.batch_size:
            # Define latent code
            latent = torch.randn(1, self.batch_size, self.model.encoder_latent_size)
            # Bypass algorithm.sample by decoding SMILES directly from latent
            generated_smiles = self.model.get_smiles_from_latent(latent)
            _, valid_ids = self.model.validate_molecules(generated_smiles)
            valid_ids = [
                i
                for i in valid_ids
                if len(
                    Chem.DetectChemistryProblems(
                        Chem.MolFromSmiles(generated_smiles[i])
                    )
                )
                == 0
            ]
            generated_molecules = list([generated_smiles[index] for index in valid_ids])
            smiles.extend(generated_molecules)
        return smiles
