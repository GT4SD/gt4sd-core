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
"""Implementation of MolGX conditional generators."""

import logging
import os
from typing import Any, Dict, List

from ....extras import EXTRAS_ENABLED

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if EXTRAS_ENABLED:
    from AMD_Analytics.amdsdk import AMDsdk

    class MolGXGenerator:
        """Interface for MolGX generator."""

        def __init__(
            self,
            resources_path: str,
            tag_name: str,
            homo_energy_value: float = -0.25,
            lumo_energy_value: float = 0.08,
            use_linear_model: bool = True,
            number_of_candidates: int = 2,
            maximum_number_of_candidates: int = 3,
            maximum_number_of_solutions: int = 3,
            maximum_number_of_nodes: int = 50000,
            beam_size: int = 2000,
            without_estimate: bool = True,
            use_specific_rings: bool = True,
            use_fragment_const: bool = False,
        ) -> None:
            """Instantiate a MolGX generator.

            Args:
                resources_path: path to the resources for model loading.
                tag_name: tag for the pretrained model.
                homo_energy_value: target HOMO energy value. Defaults to -0.25.
                lumo_energy_value: target LUMO energy value. Defaults to 0.08.
                use_linear_model: linear model usage. Defaults to True.
                number_of_candidates: number of candidates to consider. Defaults to 2.
                maximum_number_of_candidates: maximum number of candidates to consider. Defaults to 3.
                maximum_number_of_solutions: maximum number of solutions. Defaults to 3.
                maximum_number_of_nodes: maximum number of nodes in the graph exploration. Defaults to 50000.
                beam_size: size of the beam during search. Defaults to 2000.
                without_estimate: disable estimates. Defaults to True.
                use_specific_rings: flag to indicate whether specific rings are used. Defaults to True.
                use_fragment_const: using constant fragments. Defaults to False.

            Raises:
                RuntimeError: in the case extras are disabled.
            """
            if not EXTRAS_ENABLED:
                raise RuntimeError("Can't instantiate MolGXGenerator, extras disabled!")

            # loading artifacts
            self.resources_path = resources_path
            self.tag_name = tag_name
            self.amd = self.load_molgx(self.resources_path, self.tag_name)
            self.molecules_data, self.target_property = self.amd.LoadPickle("model")
            # algorithm parameters
            self._homo_energy_value = homo_energy_value
            self._lumo_energy_value = lumo_energy_value
            self._use_linear_model = use_linear_model
            self._number_of_candidates = number_of_candidates
            self._maximum_number_of_candidates = maximum_number_of_candidates
            self._maximum_number_of_solutions = maximum_number_of_solutions
            self._maximum_number_of_nodes = maximum_number_of_nodes
            self._beam_size = beam_size
            self._without_estimate = without_estimate
            self._use_specific_rings = use_specific_rings
            self._use_fragment_const = use_fragment_const
            self._parameters = self._create_parameters_dictionary()

        @staticmethod
        def load_molgx(resource_path: str, tag_name: str) -> AMDsdk:
            """Load MolGX model.

            Args:
                resource_path: path to the resources for model loading.
                tag_name: tag for the pretrained model.

            Returns:
                MolGX model SDK.
            """
            return AMDsdk(
                dir_pickle=os.path.join(resource_path, "pickle"),
                dir_data=os.path.join(resource_path, "data"),
                tag_data=tag_name,
            )

        def _create_parameters_dictionary(self) -> Dict[str, Any]:
            """Create parameters dictionary.

            Returns:
                the parameters to run MolGX.
            """
            self.target_property["homo"] = (self.homo_energy_value,) * 2
            self.target_property["lumo"] = (self.lumo_energy_value,) * 2
            parameters: Dict[str, Any] = {}
            parameters["target_property"] = self.target_property
            parameters["use_linear_model"] = self.use_linear_model
            parameters["num_candidate"] = self.number_of_candidates
            parameters["max_candidate"] = self.maximum_number_of_candidates
            parameters["max_solution"] = self.maximum_number_of_solutions
            parameters["max_node"] = self.maximum_number_of_nodes
            parameters["beam_size"] = self.beam_size
            parameters["without_estimate"] = self.without_estimate
            parameters["use_specific_rings"] = self.use_specific_rings
            parameters["use_fragment_const"] = self.use_fragment_const
            return parameters

        @property
        def homo_energy_value(self) -> float:
            return self._homo_energy_value

        @homo_energy_value.setter
        def homo_energy_value(self, value: float) -> None:
            self._homo_energy_value = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def lumo_energy_value(self) -> float:
            return self._lumo_energy_value

        @lumo_energy_value.setter
        def lumo_energy_value(self, value: float) -> None:
            self._lumo_energy_value = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def use_linear_model(self) -> bool:
            return self._use_linear_model

        @use_linear_model.setter
        def use_linear_model(self, value: bool) -> None:
            self._use_linear_model = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def number_of_candidates(self) -> int:
            return self._number_of_candidates

        @number_of_candidates.setter
        def number_of_candidates(self, value: int) -> None:
            self._number_of_candidates = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def maximum_number_of_candidates(self) -> int:
            return self._maximum_number_of_candidates

        @maximum_number_of_candidates.setter
        def maximum_number_of_candidates(self, value: int) -> None:
            self._maximum_number_of_candidates = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def maximum_number_of_solutions(self) -> int:
            return self._maximum_number_of_solutions

        @maximum_number_of_solutions.setter
        def maximum_number_of_solutions(self, value: int) -> None:
            self._maximum_number_of_solutions = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def maximum_number_of_nodes(self) -> int:
            return self._maximum_number_of_nodes

        @maximum_number_of_nodes.setter
        def maximum_number_of_nodes(self, value: int) -> None:
            self._maximum_number_of_nodes = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def beam_size(self) -> int:
            return self._beam_size

        @beam_size.setter
        def beam_size(self, value: int) -> None:
            self._beam_size = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def without_estimate(self) -> bool:
            return self._without_estimate

        @without_estimate.setter
        def without_estimate(self, value: bool) -> None:
            self._without_estimate = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def use_specific_rings(self) -> bool:
            return self._use_specific_rings

        @use_specific_rings.setter
        def use_specific_rings(self, value: bool) -> None:
            self._use_specific_rings = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def use_fragment_const(self) -> bool:
            return self._use_fragment_const

        @use_fragment_const.setter
        def use_fragment_const(self, value: bool) -> None:
            self._use_fragment_const = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def parameters(self) -> Dict[str, Any]:
            return self._parameters

        @parameters.setter
        def parameters(self, value: Dict[str, Any]) -> None:
            parameters = self._create_parameters_dictionary()
            parameters.update(value)
            self._parameters = parameters

        def generate(self) -> List[str]:
            """Sample random molecules.

            Returns:
                sampled molecule (SMILES).
            """
            # generate molecules
            logger.info(
                f"running MolGX with the following parameters: {self.parameters}"
            )
            molecules_df = self.amd.GenMols(self.molecules_data, self.parameters)
            logger.info("MolGX run completed")
            return molecules_df["SMILES"].tolist()

else:
    logger.warning("install AMD_analytcs extras to use MolGX")
