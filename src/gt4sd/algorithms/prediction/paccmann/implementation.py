
"""Implementation of the zero-shot classifier."""

import json
import logging
import os
from typing import List, Optional, Union

import torch
from transformers import pipeline

from paccmann_predictor.models import MODEL_FACTORY
from pytoda.proteins.protein_language import ProteinLanguage
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.transforms import LeftPadding, ToTensor

from ....frameworks.torch import device_claim

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BimodalMCAAffinityPredictor:
    """
    Bimodal MCA (Multiscale Convolutional Attention) affinity prediction model 
    See https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.9b00520
    and https://iopscience.iop.org/article/10.1088/2632-2153/abe808    
    """

    def __init__(
        self,
        resources_path: str,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """Initialize BimodalMCAAffinityPredictor.

        Args:
            resources_path: path where to load hypothesis, candidate labels and, optionally, the model.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """
        self.device = device_claim(device)        
        self.resources_path = resources_path

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

    #fixme: target seems to usually be part of the configuration (for most generative models...)
    #todo: add batch support
    def predict(self, target: str, ligand_smiles: str, confidence=False) -> List[str]:
        """Get predicted affinity.

        Args:
            target: target sequence
            ligand_smiles: SMILES representation of the ligand
            confidence: set True to calculate confidence in two ways - 
                monte carl droput based
                and test time augmentation based

        Returns:
            predicted affinity
        """

        if isinstance(target, str):
            target = [target]

        if isinstance(ligand_smiles, str):
            ligand_smiles = [ligand_smiles]
        assert isinstance(target, list)
        assert isinstance(ligand_smiles, list)

        assert len(target)>0
        assert len(target) == len(ligand_smiles)

        ### prepare ligand representation
        ligand_tensor = torch.cat(
            [
                torch.unsqueeze(
                    self.to_tensor(
                        self.pad_smiles_predictor(
                            self.affinity_predictor.smiles_language.smiles_to_token_indexes(
                                smile
                            )
                        )
                    ),
                    0,
                )
                for smile in ligand_smiles
            ],
            dim=0,
        )
        

        ### prepare target representation
        # target_tensor = torch.unsqueeze(
        #     self.to_tensor(
        #         self.pad_protein_predictor(
        #             self.affinity_predictor.protein_language.sequence_to_token_indexes(
        #                 target
        #             )
        #         )
        #     ),
        #     0,
        # )

        target_tensor = torch.cat(
            [
                torch.unsqueeze(
                    self.to_tensor(
                        self.pad_protein_predictor(
                            self.affinity_predictor.protein_language.sequence_to_token_indexes(
                                curr_target
                            )
                        )
                    ),
                    0,
                )
                for curr_target in target
            ],
            dim=0,
        )

        with torch.no_grad():
            model_ans = self.affinity_predictor(
                ligand_tensor,
                target_tensor,
                confidence=confidence,
            )

        return model_ans
