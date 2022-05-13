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
from typing import Any, List, Optional, Union

import torch
from paccmann_predictor.models import MODEL_FACTORY
from pytoda.proteins.protein_language import ProteinLanguage
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.transforms import LeftPadding, ToTensor

from ....frameworks.torch import device_claim

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MCAPredictor:
    """Base implementation of an MCAPredictor."""

    def predict(self) -> Any:
        """Get prediction.

        Returns:
            predicted affinity
        """
        raise NotImplementedError("No prediction implemented for base MCAPredictor")

    def predict_values(self) -> Any:
        """Get prediction for algorithm sample method.

        Returns:
            predicted values as list.
        """
        raise NotImplementedError(
            "No values prediction implemented for base MCAPredictor"
        )


class BimodalMCAAffinityPredictor(MCAPredictor):
    """Bimodal MCA (Multiscale Convolutional Attention) affinity prediction model.

    For details see: https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.9b00520
    and https://iopscience.iop.org/article/10.1088/2632-2153/abe808.
    """

    def __init__(
        self,
        resources_path: str,
        protein_targets: List[str],
        ligands: List[str],
        confidence: bool,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """Initialize BimodalMCAAffinityPredictor.

        Args:
            resources_path: path where to load model weights and cofiguration.
            protein_targets: list of protein targets as AA sequences.
            ligands: list of ligands in SMILES format.
            confidence: whether the confidence for the prediction should be returned.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """
        self.device = device_claim(device)
        self.resources_path = resources_path
        self.protein_targets = protein_targets
        self.ligands = ligands
        self.confidence = confidence

        # setting affinity predictor parameters
        with open(os.path.join(resources_path, "mca_model_params.json")) as f:
            self.predictor_params = json.load(f)
        self.affinity_predictor = MODEL_FACTORY["bimodal_mca"](self.predictor_params)
        self.affinity_predictor.load(
            os.path.join(resources_path, "mca_weights.pt"),
            map_location=self.device,
        )
        affinity_protein_language = ProteinLanguage.load(
            os.path.join(resources_path, "protein_language.pkl")
        )
        affinity_smiles_language = SMILESLanguage.load(
            os.path.join(resources_path, "smiles_language.pkl")
        )
        self.affinity_predictor._associate_language(affinity_smiles_language)
        self.affinity_predictor._associate_language(affinity_protein_language)
        self.affinity_predictor.eval()

        self.pad_smiles_predictor = LeftPadding(
            self.affinity_predictor.smiles_padding_length,
            self.affinity_predictor.smiles_language.padding_index,
        )

        self.pad_protein_predictor = LeftPadding(
            self.affinity_predictor.protein_padding_length,
            self.affinity_predictor.protein_language.padding_index,
        )

        self.to_tensor = ToTensor()

    def predict(self) -> Any:
        """Get predicted affinity.

        Returns:
            predicted affinity.
        """
        # prepare ligand representation
        ligand_tensor = torch.cat(
            [
                torch.unsqueeze(
                    self.to_tensor(
                        self.pad_smiles_predictor(
                            self.affinity_predictor.smiles_language.smiles_to_token_indexes(
                                ligand_smiles
                            )
                        )
                    ),
                    0,
                )
                for ligand_smiles in self.ligands
            ],
            dim=0,
        )

        # prepare target protein representation
        target_tensor = torch.cat(
            [
                torch.unsqueeze(
                    self.to_tensor(
                        self.pad_protein_predictor(
                            self.affinity_predictor.protein_language.sequence_to_token_indexes(
                                protein_target
                            )
                        )
                    ),
                    0,
                )
                for protein_target in self.protein_targets
            ],
            dim=0,
        )

        with torch.no_grad():
            predictions, predictions_dict = self.affinity_predictor(
                ligand_tensor,
                target_tensor,
                confidence=self.confidence,
            )

        return predictions, predictions_dict

    def predict_values(self) -> List[float]:
        """Get prediction for algorithm sample method.

        Returns:
            predicted values as list.
        """
        predictions, _ = self.predict()
        return list(predictions[:, 0])
