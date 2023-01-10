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
from typing import List

from pydantic import Field

import torch
import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable

from ...algorithms.core import (
    ConfigurablePropertyAlgorithmConfiguration,
    Predictor,
    PredictorAlgorithm,
)
from ..core import (
    DomainSubmodule,
    PropertyValue,
    S3Parameters,
)

from ...frameworks.cgcnn.model import CrystalGraphConvNet, Normalizer
from ...frameworks.cgcnn.data import CIFData, collate_pool


class S3ParametersCrystals(S3Parameters):
    domain: DomainSubmodule = DomainSubmodule("crystals")


class CgcnnParameters(S3ParametersCrystals):
    algorithm_name: str = "cgcnn"
    batch_size: int = Field(description="Prediction batch size", default=256)
    workers: int = Field(description="Number of data loading workers", default=0)


class FormationEnergyParameters(CgcnnParameters):
    algorithm_application: str = "FormationEnergy"


class AbsoluteEnergyParameters(CgcnnParameters):
    algorithm_application: str = "AbsoluteEnergy"


class BandGapParameters(CgcnnParameters):
    algorithm_application: str = "BandGap"


class FermiEnergyParameters(CgcnnParameters):
    algorithm_application: str = "FermiEnergy"


class BulkModuliParameters(CgcnnParameters):
    algorithm_application: str = "BulkModuli"


class ShearModuliParameters(CgcnnParameters):
    algorithm_application: str = "ShearModuli"


class PoissonRationParameters(CgcnnParameters):
    algorithm_application: str = "PoissonRatio"


class MetalClassifierParameters(CgcnnParameters):
    algorithm_application: str = "MetalClassifier"


class _CGCNN(PredictorAlgorithm):
    """Base class for all cgcnn-based predictive algorithms."""

    def __init__(self, parameters: CgcnnParameters):

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
        model_checkpoint = torch.load(
            resources_path, map_location=lambda storage, loc: storage
        )
        model_args = argparse.Namespace(**model_checkpoint["args"])

        normalizer = Normalizer(torch.zeros(3))

        checkpoint = torch.load(
            resources_path, map_location=lambda storage, loc: storage
        )
        normalizer.load_state_dict(checkpoint["normalizer"])

        # Wrapper to get toxicity-endpoint-level predictions
        def informative_model(cif_path: str) -> List[PropertyValue]:

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
            nbr_fea_len = structures[1].shape[-1]

            model = CrystalGraphConvNet(
                orig_atom_fea_len,
                nbr_fea_len,
                atom_fea_len=model_args.atom_fea_len,
                n_conv=model_args.n_conv,
                h_fea_len=model_args.h_fea_len,
                n_h=model_args.n_h,
                classification=False,
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
                test_pred = normalizer.denorm(output.data.cpu())
                test_preds += test_pred.view(-1).tolist()
                test_cif_ids += batch_cif_ids

            return model.predictions.detach().tolist()

        return informative_model

    @classmethod
    def get_description(cls) -> str:
        text = """
        This model predicts the 12 endpoints from the Tox21 challenge.
        The endpoints are: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
        For details on the data see: https://tripod.nih.gov/tox21/challenge/.
        """
        return text


class FormationEnergy(_CGCNN):
    @classmethod
    def get_description(cls) -> str:
        text = """
        This model predicts the 12 endpoints from the Tox21 challenge.
        The endpoints are: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
        For details on the data see: https://tripod.nih.gov/tox21/challenge/.
        """
        return text


class AbsoluteEnergy(_CGCNN):
    @classmethod
    def get_description(cls) -> str:
        text = """
        This model predicts the 12 endpoints from the Tox21 challenge.
        The endpoints are: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
        For details on the data see: https://tripod.nih.gov/tox21/challenge/.
        """
        return text


class BandGap(_CGCNN):
    @classmethod
    def get_description(cls) -> str:
        text = """
        This model predicts the 12 endpoints from the Tox21 challenge.
        The endpoints are: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
        For details on the data see: https://tripod.nih.gov/tox21/challenge/.
        """
        return text


class FermiEnergy(_CGCNN):
    @classmethod
    def get_description(cls) -> str:
        text = """
        This model predicts the 12 endpoints from the Tox21 challenge.
        The endpoints are: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
        For details on the data see: https://tripod.nih.gov/tox21/challenge/.
        """
        return text


class BulkModuli(_CGCNN):
    @classmethod
    def get_description(cls) -> str:
        text = """
        This model predicts the 12 endpoints from the Tox21 challenge.
        The endpoints are: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
        For details on the data see: https://tripod.nih.gov/tox21/challenge/.
        """
        return text


class ShearModuli(_CGCNN):
    @classmethod
    def get_description(cls) -> str:
        text = """
         This model predicts the 12 endpoints from the Tox21 challenge.
         The endpoints are: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
         For details on the data see: https://tripod.nih.gov/tox21/challenge/.
         """
        return text


class PoissonRatio(_CGCNN):
    @classmethod
    def get_description(cls) -> str:
        text = """
         This model predicts the 12 endpoints from the Tox21 challenge.
         The endpoints are: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
         For details on the data see: https://tripod.nih.gov/tox21/challenge/.
         """
        return text


class MetalClassifer(_CGCNN):
    @classmethod
    def get_description(cls) -> str:
        text = """
         This model predicts the 12 endpoints from the Tox21 challenge.
         The endpoints are: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
         For details on the data see: https://tripod.nih.gov/tox21/challenge/.
         """
        return text
