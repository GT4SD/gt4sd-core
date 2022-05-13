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
"""Implementation details for PaccMann vanilla generator trained on polymer building blocks (catalysts/monomers)."""

import json
import os
from typing import List, Optional, Union

import torch
from rdkit import Chem, RDLogger
from paccmann_chemistry.models.vae import StackGRUDecoder, StackGRUEncoder, TeacherVAE
from paccmann_chemistry.utils import get_device
from paccmann_chemistry.utils.search import SamplingSearch
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.smiles.transforms import Selfies, SMILESToTokenIndexes
from pytoda.transforms import Compose, ToTensor

from ....frameworks.torch import device_claim

RDLogger.DisableLog("rdApp.*")


class Generator:
    def __init__(
        self,
        resources_path: str,
        generated_length: int = 100,
        batch_size: int = 32,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """Initialize the encoder/decoder generative model.

        Args:
            resources_path: directory where to find models and parameters.
            generated_length: length of the generated molecule in tokens. Defaults to 100.
            batch_size: size of the batch. Defaults to 1.
            device: device where the inference is running either as a dedicated class or a string.
                If not provided is inferred.
        """
        self.device = device_claim(device)
        self.generated_length = generated_length
        self.batch_size = batch_size
        self.resources_path = resources_path
        self.load_pretrained_paccmann(
            os.path.join(self.resources_path, "params.json"),
            os.path.join(self.resources_path, "smiles_language.pkl"),
            os.path.join(self.resources_path, "weights.pt"),
            self.batch_size,
        )

    def load_pretrained_paccmann(
        self, params_file: str, lang_file: str, weights_file: str, batch_size: int
    ) -> None:
        """Load a pretrained PaccMann model.

        Args:
            params_file: file for the parameters.
            lang_file: language file.
            weights_file: serialized weights file.
            batch_size: size of the batch.
        """
        params = dict()
        with open(params_file, "r") as f:
            params.update(json.load(f))
        params["batch_mode"] = "Padded"
        params["batch_size"] = batch_size

        self.selfies = params.get("selfies", False)

        self.device = get_device()
        self.smiles_language = SMILESLanguage.load(lang_file)

        self.gru_encoder = StackGRUEncoder(params).to(self.device)
        self.gru_decoder = StackGRUDecoder(params).to(self.device)
        self.gru_vae = TeacherVAE(self.gru_encoder, self.gru_decoder).to(self.device)
        self.gru_vae.load_state_dict(torch.load(weights_file, map_location=self.device))
        self.gru_vae.eval()

        transforms = []
        if self.selfies:
            transforms += [Selfies()]
        transforms += [SMILESToTokenIndexes(smiles_language=self.smiles_language)]
        transforms += [ToTensor(device=self.device)]
        self.transform = Compose(transforms)

    def decode(
        self, latent_z: torch.Tensor, search: SamplingSearch = SamplingSearch()
    ) -> List[int]:
        """Decodes a sequence of tokens given a position in the latent space.

        Args:
            latent_z: a batch size x latent size tensor.
            search: defaults to sampling multinomial search.

        Returns:
            list of list of token indices.
        """
        latent_z = latent_z.view(1, latent_z.shape[0], latent_z.shape[1]).float()
        molecule_iter = self.gru_vae.generate(
            latent_z,
            prime_input=torch.tensor([self.smiles_language.start_index]).to(
                self.device
            ),
            end_token=torch.tensor([self.smiles_language.stop_index]).to(self.device),
            generate_len=self.generated_length,
            search=search,
        )
        return [
            [self.smiles_language.start_index] + m.cpu().detach().tolist()
            for m in molecule_iter
        ]

    def sample(self) -> List[str]:
        """Sample random molecules.

        Returns:
            sampled molecule (SMILES).
        """
        mol: List[str] = []
        while len(mol) < 1:
            indexes = self.decode(
                torch.randn(
                    self.batch_size, self.gru_decoder.latent_dim, device=self.device
                )
            )
            mol = [self.smiles_language.token_indexes_to_smiles(m) for m in indexes]
            mol = [m for m in mol if Chem.MolFromSmiles(m) is not None and m != ""]
        return mol
