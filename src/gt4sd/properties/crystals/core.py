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
import argparse
import os
from typing import Dict, List

import torch
from pydantic import Field
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ...algorithms.core import (
    ConfigurablePropertyAlgorithmConfiguration,
    Predictor,
    PredictorAlgorithm,
)
from ...frameworks.cgcnn.data import CIFData, collate_pool
from ...frameworks.cgcnn.model import CrystalGraphConvNet, Normalizer
from ..core import DomainSubmodule, S3Parameters


class S3ParametersCrystals(S3Parameters):
    domain: DomainSubmodule = DomainSubmodule("crystals")


class CGCNNParameters(S3ParametersCrystals):
    algorithm_name: str = "cgcnn"
    batch_size: int = Field(description="Prediction batch size", default=256)
    workers: int = Field(description="Number of data loading workers", default=0)


class FormationEnergyParameters(CGCNNParameters):
    algorithm_application: str = "FormationEnergy"


class AbsoluteEnergyParameters(CGCNNParameters):
    algorithm_application: str = "AbsoluteEnergy"


class BandGapParameters(CGCNNParameters):
    algorithm_application: str = "BandGap"


class FermiEnergyParameters(CGCNNParameters):
    algorithm_application: str = "FermiEnergy"


class BulkModuliParameters(CGCNNParameters):
    algorithm_application: str = "BulkModuli"


class ShearModuliParameters(CGCNNParameters):
    algorithm_application: str = "ShearModuli"


class PoissonRatioParameters(CGCNNParameters):
    algorithm_application: str = "PoissonRatio"


class MetalSemiconductorClassifierParameters(CGCNNParameters):
    algorithm_application: str = "MetalSemiconductorClassifier"


class _CGCNN(PredictorAlgorithm):
    """Base class for all cgcnn-based predictive algorithms."""

    def __init__(self, parameters: CGCNNParameters):

        # Set up the configuration from the parameters
        configuration = ConfigurablePropertyAlgorithmConfiguration(
            algorithm_type=parameters.algorithm_type,
            domain=parameters.domain,
            algorithm_name=parameters.algorithm_name,
            algorithm_application=parameters.algorithm_application,
            algorithm_version=parameters.algorithm_version,
        )

        self.batch_size = parameters.batch_size
        self.workers = parameters.workers

        # The parent constructor calls `self.get_model`.
        super().__init__(configuration=configuration)

    def get_model(self, resources_path: str) -> Predictor:
        """Instantiate the actual model.

        Args:
            resources_path: local path to model files.

        Returns:
            Predictor: the model.
        """

        existing_models = os.listdir(resources_path)
        existing_models = [
            file for file in existing_models if file.endswith(".pth.tar")
        ]

        if len(existing_models) > 1:
            raise ValueError(
                "Only one model should be located in the specified model path."
            )
        elif len(existing_models) == 0:
            raise ValueError("Model does not exist in the specified model path.")

        model_path = os.path.join(resources_path, existing_models[0])

        model_checkpoint = torch.load(
            model_path, map_location=lambda storage, loc: storage
        )

        model_args = argparse.Namespace(**model_checkpoint["args"])

        normalizer = Normalizer(torch.zeros(3))

        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        normalizer.load_state_dict(checkpoint["normalizer"])

        # Wrapper to get toxicity-endpoint-level predictions
        def informative_model(cif_path: str) -> Dict[str, List[float]]:

            dataset = CIFData(cif_path)
            test_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.workers,
                collate_fn=collate_pool,
            )

            # build model
            structures, _, _ = dataset[0]
            orig_atom_fea_len = structures[0].shape[-1]
            nbr_fea_len = structures[1].shape[-1]  # type: ignore

            model = CrystalGraphConvNet(
                orig_atom_fea_len,
                nbr_fea_len,
                atom_fea_len=model_args.atom_fea_len,
                n_conv=model_args.n_conv,
                h_fea_len=model_args.h_fea_len,
                n_h=model_args.n_h,
                classification=True if model_args.task == "classification" else False,
            )

            model.load_state_dict(checkpoint["state_dict"])

            test_preds = []
            test_cif_ids = []

            for i, (input, target, batch_cif_ids) in enumerate(test_loader):
                with torch.no_grad():
                    input_var = (
                        Variable(input[0]),
                        Variable(input[1]),
                        input[2],
                        input[3],
                    )

                # compute output
                output = model(*input_var)

                # record loss
                if model_args.task == "classification":
                    test_pred = torch.exp(output.data.cpu())
                    test_preds += test_pred[:, 1].tolist()
                else:
                    test_pred = normalizer.denorm(output.data.cpu())
                    test_preds += test_pred.view(-1).tolist()
                test_cif_ids += batch_cif_ids

            return {"cif_ids": test_cif_ids, "predictions": test_preds}  # type: ignore

        return informative_model


class FormationEnergy(_CGCNN):
    @classmethod
    def get_description(cls) -> str:
        text = """
        This model predicts the formation energy per atom using the CGCNN framework.
        For more details see: https://doi.org/10.1103/PhysRevLett.120.145301.
        """
        return text


class AbsoluteEnergy(_CGCNN):
    @classmethod
    def get_description(cls) -> str:
        text = """
        This model predicts the absolute energy of crystals using the CGCNN framework.
        For more details see: https://doi.org/10.1103/PhysRevLett.120.145301.
        """
        return text


class BandGap(_CGCNN):
    @classmethod
    def get_description(cls) -> str:
        text = """
        This model predicts the band gap of crystals using the CGCNN framework.
        For more details see: https://doi.org/10.1103/PhysRevLett.120.145301.
        """
        return text


class FermiEnergy(_CGCNN):
    @classmethod
    def get_description(cls) -> str:
        text = """
        This model predicts the Fermi energy of crystals using the CGCNN framework.
        For more details see: https://doi.org/10.1103/PhysRevLett.120.145301.
        """
        return text


class BulkModuli(_CGCNN):
    @classmethod
    def get_description(cls) -> str:
        text = """
        This model predicts the bulk moduli of crystals using the CGCNN framework.
        For more details see: https://doi.org/10.1103/PhysRevLett.120.145301.
        """
        return text


class ShearModuli(_CGCNN):
    @classmethod
    def get_description(cls) -> str:
        text = """
         This model predicts the shear moduli of crystals using the CGCNN framework.
        For more details see: https://doi.org/10.1103/PhysRevLett.120.145301.
         """
        return text


class PoissonRatio(_CGCNN):
    @classmethod
    def get_description(cls) -> str:
        text = """
        This model predicts the Poisson ratio of crystals using the CGCNN framework.
        For more details see: https://doi.org/10.1103/PhysRevLett.120.145301.
         """
        return text


class MetalSemiconductorClassifier(_CGCNN):
    @classmethod
    def get_description(cls) -> str:
        text = """
         This model predicts whether a given crystal is metal or semiconductor using the CGCNN framework.
        For more details see: https://doi.org/10.1103/PhysRevLett.120.145301.
         """
        return text
