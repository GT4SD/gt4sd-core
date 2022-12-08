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

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from paccmann_chemistry.models import StackGRUDecoder, StackGRUEncoder, TeacherVAE
from paccmann_chemistry.utils.search import SamplingSearch
from paccmann_omics.encoders import ENCODER_FACTORY
from pytoda.smiles.smiles_language import SMILESLanguage

from ....domains.materials import validate_molecules
from ....domains.materials.protein_encoding import PrimarySequenceEncoder
from ....frameworks.torch import device_claim
from ....frameworks.torch.vae import reparameterize

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ConditionalGenerator(ABC):
    """Abstract interface for a conditional generator."""

    #: device where the inference is running.
    device: torch.device
    #: temperature for the sampling.
    temperature: float
    #: maximum length of the generated molecules.
    generated_length: int

    #: parameters for the SELFIES generator.
    selfies_conditional_generator_params: dict
    #: SELFIES generator.
    selfies_conditional_generator: TeacherVAE
    #: SMILES language instance.
    smiles_language: SMILESLanguage

    generator_latent_size: int
    encoder_latent_size: int

    def get_smiles_from_latent(self, latent: torch.Tensor) -> List[str]:
        """Take samples from the latent space.

        Args:
            latent: latent vector tensor.

        Returns:
            SMILES list and indexes for the valid ones.
        """
        if self.generator_latent_size == 2 * self.encoder_latent_size:
            latent = latent.repeat(1, 1, 2)

        # generate molecules as tokens list
        generated_molecules = self.selfies_conditional_generator.generate(
            latent,
            prime_input=torch.tensor(
                [self.smiles_language.start_index], device=self.device
            ).long(),
            end_token=torch.tensor(
                [self.smiles_language.stop_index], device=self.device
            ).long(),
            generate_len=self.generated_length,
            search=SamplingSearch(temperature=self.temperature),
        )

        molecules = [
            self.smiles_language.token_indexes_to_smiles(generated_molecule.tolist())
            for generated_molecule in iter(generated_molecules)
        ]

        # convert SELFIES to SMILES
        if "selfies" in self.smiles_language.name:
            molecules = [
                self.smiles_language.selfies_to_smiles(a_selfies)
                for a_selfies in molecules
            ]
        return molecules

    @staticmethod
    def validate_molecules(smiles) -> Tuple[List[Chem.rdchem.Mol], List[int]]:
        return validate_molecules(smiles_list=smiles)

    @abstractmethod
    def get_latent(self, condition: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def generate_valid(self, condition: Any, number_of_molecules: int) -> List[str]:
        """
        Generate a given number of samples (molecules) from a given condition.

        Args:
            protein: the protein used as context/condition.
            number_of_molecules: number of molecules to sample.

        Returns:
            list of SMILES generated.
        """
        # prepare the molecule set
        generated_molecules: Set[str] = set()
        logger.info("embedding condition and getting reparametrized latent samples")
        latent = self.get_latent(condition)
        logger.info("starting generation of molecules")
        while len(generated_molecules) < number_of_molecules:
            # generate the molecules
            generated_smiles = self.get_smiles_from_latent(latent)
            _, valid_ids = self.validate_molecules(generated_smiles)
            generated_molecules |= set([generated_smiles[index] for index in valid_ids])
        logger.info("completed generation of molecules")
        # return the molecules listed by length
        return sorted(list(generated_molecules), key=len, reverse=True)[
            :number_of_molecules
        ]

    def generate_batch(self, condition: Any) -> List[str]:
        logger.info("embedding condition and getting reparametrized latent samples")
        latent = self.get_latent(condition)
        logger.info("starting generation of molecules")
        # generate the molecules
        return self.get_smiles_from_latent(latent)


class ProteinSequenceConditionalGenerator(ConditionalGenerator):
    """
    Protein conditional generator as implemented in https://doi.org/10.1088/2632-2153/abe808
    (originally https://arxiv.org/abs/2005.13285).
    It generates highly binding and low toxic ligands.

    Attributes:
        samples_per_protein: number of points sampled per protein.
            It has to be greater than 1.
        protein_embedding_encoder_params: parameter for the protein embedding encoder.
        protein_embedding_encoder: protein embedding encoder.
    """

    def __init__(
        self,
        resources_path: str,
        temperature: float = 1.4,
        generated_length: int = 100,
        samples_per_protein: int = 100,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """
        Initialize the generator.

        Args:
            resources_path: directory where to find models and parameters.
            temperature: temperature for the sampling. Defaults to 1.4.
            generated_length: maximum length of the generated molecules.
                Defaults to 100.
            samples_per_protein: number of points sampled per protein.
                It has to be greater than 1. Defaults to 10.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """
        # device
        self.device = device_claim(device)
        # setting sampling parameters
        self.temperature = temperature
        self.generated_length = generated_length
        self.samples_per_protein = samples_per_protein
        # instantiate protein embedding encoder
        with open(os.path.join(resources_path, "protein_embedding_params.json")) as fp:
            self.protein_embedding_encoder_params = json.load(fp)
        self.protein_embedding_encoder = ENCODER_FACTORY["dense"](
            self.protein_embedding_encoder_params
        ).to(self.device)
        self.protein_embedding_encoder.load(
            os.path.join(resources_path, "protein_embedding_encoder.pt"),
            map_location=self.device,
        )
        self.protein_embedding_encoder.eval()
        self.encoder_latent_size = self.protein_embedding_encoder.latent_size
        # instantiate selfies conditional generator
        with open(
            os.path.join(resources_path, "selfies_conditional_generator.json")
        ) as fp:
            self.selfies_conditional_generator_params = json.load(fp)
        self.selfies_conditional_generator = TeacherVAE(
            StackGRUEncoder(self.selfies_conditional_generator_params),
            StackGRUDecoder(self.selfies_conditional_generator_params),
        ).to(self.device)
        self.selfies_conditional_generator.load(
            os.path.join(resources_path, "selfies_conditional_generator.pt"),
            map_location=self.device,
        )
        self.selfies_conditional_generator.eval()
        self.generator_latent_size = (
            self.selfies_conditional_generator.decoder.latent_dim
        )
        # loading SMILES language for decoding and conversion of SELFIES to SMILES
        self.smiles_language = SMILESLanguage.load(
            os.path.join(resources_path, "selfies_language.pkl")
        )
        # protein embedding from primary sequence (via tape)
        self.primary_sequence_embedder = PrimarySequenceEncoder(
            model_type="transformer",
            from_pretrained="bert-base",
            model_config_file=None,
            tokenizer="iupac",
        ).to(self.device)

    def get_latent(self, protein: str) -> torch.Tensor:
        """
        Given a protein generate the latent representation.

        Args:
            protein: the protein used as context/condition.

        Returns:
            the latent representation for the given context. It contains
                self.samples_per_protein repeats.
        """
        # encode embedded sequence once, ignore the returned dummy ids
        embeddings, _ = self.primary_sequence_embedder.forward([[protein]])
        protein_mu, protein_logvar = self.protein_embedding_encoder(
            embeddings.to(self.device)
        )

        # now stack as batch to generate different samples
        proteins_mu = torch.cat([protein_mu] * self.samples_per_protein, dim=0)
        proteins_logvar = torch.cat([protein_logvar] * self.samples_per_protein, dim=0)
        # get latent representation
        return torch.unsqueeze(reparameterize(proteins_mu, proteins_logvar), 0)

    def generate_valid(self, protein: str, number_of_molecules: int) -> List[str]:
        """
        Generate a given number of samples (molecules) from a given protein.

        Args:
            protein: the protein used as context/condition.
            number_of_molecules: number of molecules to sample.

        Returns:
            list of SMILES generated.
        """
        return super().generate_valid(
            condition=protein, number_of_molecules=number_of_molecules
        )

    def generate_batch(self, protein: str) -> List[str]:
        return super().generate_batch(condition=protein)


class TranscriptomicConditionalGenerator(ConditionalGenerator):
    """
    Transcriptomic conditional generator as implemented in https://doi.org/10.1016/j.isci.2021.102269
    (originally https://doi.org/10.1007/978-3-030-45257-5_18, https://arxiv.org/abs/1909.05114).
    It generates highly effective small molecules against transcriptomic progiles.

    Attributes:
        samples_per_profile: number of points sampled per profile.
            It has to be greater than 1.
        transcriptomic_encoder_params: parameter for the protein embedding encoder.
        transcriptomic_encoder: protein embedding encoder.
    """

    def __init__(
        self,
        resources_path: str,
        temperature: float = 1.4,
        generated_length: int = 100,
        samples_per_profile: int = 100,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """
        Initialize the generator.

        Args:
            resources_path: directory where to find models and parameters.
            temperature: temperature for the sampling. Defaults to 1.4.
            generated_length: maximum length of the generated molecules.
                Defaults to 100.
            samples_per_profile: number of points sampled per protein.
                It has to be greater than 1. Defaults to 10.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """
        # device
        self.device = device_claim(device)
        # setting sampling parameters
        self.temperature = temperature
        self.generated_length = generated_length
        self.samples_per_profile = samples_per_profile
        with open(os.path.join(resources_path, "genes.txt")) as fp:
            self.genes = [gene.strip() for gene in fp if gene]
        # instantiate protein embedding encoder
        with open(os.path.join(resources_path, "transcriptomic_params.json")) as fp:
            self.transcriptomic_encoder_params = json.load(fp)
        self.transcriptomic_encoder = ENCODER_FACTORY["dense"](
            self.transcriptomic_encoder_params
        ).to(self.device)
        self.transcriptomic_encoder.load(
            os.path.join(resources_path, "transcriptomic_encoder.pt"),
            map_location=self.device,
        )
        self.transcriptomic_encoder.eval()
        self.encoder_latent_size = self.transcriptomic_encoder.latent_size
        # instantiate selfies conditional generator
        with open(
            os.path.join(resources_path, "selfies_conditional_generator.json")
        ) as fp:
            self.selfies_conditional_generator_params = json.load(fp)
        self.selfies_conditional_generator = TeacherVAE(
            StackGRUEncoder(self.selfies_conditional_generator_params),
            StackGRUDecoder(self.selfies_conditional_generator_params),
        ).to(self.device)
        self.selfies_conditional_generator.load(
            os.path.join(resources_path, "selfies_conditional_generator.pt"),
            map_location=self.device,
        )
        self.selfies_conditional_generator.eval()
        self.generator_latent_size = (
            self.selfies_conditional_generator.decoder.latent_dim
        )
        # loading SMILES language for decoding and conversion of SELFIES to SMILES
        self.smiles_language = SMILESLanguage.load(
            os.path.join(resources_path, "selfies_language.pkl")
        )

    def get_latent(self, profile: Union[np.ndarray, pd.Series, str]) -> torch.Tensor:
        """
        Given a profile generate the latent representation.

        Args:
            profile: the profile used as context/condition.

        Raises:
            ValueError: in case the profile has a size mismatch with the genes panel.

        Returns:
            the latent representation for the given context. It contains
                self.samples_per_profile repeats.
        """
        if isinstance(profile, pd.Series):
            # make sure genes are sorted
            profile = profile[self.genes].values
        elif isinstance(profile, str):
            logger.warning("profile passed as string, serializing it to a list")
            profile = np.array(json.loads(profile))
        if profile.size != len(self.genes):
            raise ValueError(
                f"provided profile size ({profile.size}) does not match required size {len(self.genes)}"
            )
        # encode embedded progiles
        transcriptomic_mu, transcriptomic_logvar = self.transcriptomic_encoder(
            torch.from_numpy(
                np.vstack([profile] * self.samples_per_profile),
            )
            .float()
            .to(self.device)
        )
        # get latent representation
        return torch.unsqueeze(
            reparameterize(transcriptomic_mu, transcriptomic_logvar), 0
        )

    def generate_valid(
        self, profile: Union[np.ndarray, pd.Series], number_of_molecules: int
    ) -> List[str]:
        """
        Generate a given number of samples (molecules) from a given transcriptomic profile.

        Args:
            profile: the profile used as context/condition.
            number_of_molecules: number of molecules to sample.

        Returns:
            list of SMILES generated.
        """
        return super().generate_valid(
            condition=profile, number_of_molecules=number_of_molecules
        )

    def generate_batch(self, profile: Union[np.ndarray, pd.Series]) -> List[str]:
        return super().generate_batch(condition=profile)
