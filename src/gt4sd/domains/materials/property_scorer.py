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
from typing import Any, Dict, List, Union

from rdkit import Chem

from ...properties import (
    MOLECULE_PROPERTY_PREDICTOR_FACTORY,
    PROTEIN_PROPERTY_PREDICTOR_FACTORY,
    PropertyPredictorRegistry,
)
from .scorer import TargetValueScorer


class PropertyPredictorScorer(TargetValueScorer):
    """Property Predictor Scorer."""

    def __init__(
        self,
        name: str,
        target: Union[float, int],
        parameters: Dict[str, Any] = {},
    ) -> None:
        """Scoring function that calculates a generic score for a property.

        Args:
            name: name of the property to score.
            target: target score that will be used to get the distance to the score of a molecule or protein (not be confused with parameters["target"]).
            parameters: parameters for the scoring function.
        """
        self.name = name
        self.target = target
        self.parameters = parameters

        self.scoring_function = PropertyPredictorRegistry.get_property_predictor(  # type: ignore
            name=self.name, parameters=self.parameters
        )
        super().__init__(target=target, scoring_function=self.scoring_function)

    def score(self, sample: str) -> float:
        """Generates a score for a given molecule or protein.

        Args:
            sample: molecule or protein.

        Returns:
            A score for the given molecule or protein.
        """

        self.validate_input(sample=sample, property_name=self.name)
        return self.get_distance(self.scoring_function(sample) - self.target)

    def score_list(self, sample_list: List[str]) -> List[float]:
        """Generates a list of scores for a given molecule or protein list.

        Args:
            samples_list: A List of molecules or proteins.

        Returns:
            A List of scores
        """
        scores = []
        for sample in sample_list:
            scores.append(self.score(sample))
        return scores

    def predictor(self, sample: str) -> Union[float, int]:
        """Generates a prediction for a given molecule or protein.

        Args:
            sample: molecule or protein.

        Returns:
            A score for the given SMILES
        """

        self.validate_input(sample=sample, property_name=self.name)
        return self.scoring_function(sample)

    def validate_input(self, sample: str, property_name: str) -> None:
        """Validates the sample in input.
        If self.name is a property available for molecules, check that sample is a SMILES.
        If self.name is a property available for proteins, check that sample is a protein.

        Args:
            sample: molecule or protein.

        Raises:
            ValueError: if the sample is not a valid SMILES or protein given a certain property.

        Returns:
            True if the input is valid.
        """
        # if selected property is available for molecules, check that sample is a SMILES
        if self.name in MOLECULE_PROPERTY_PREDICTOR_FACTORY:
            if Chem.MolFromSmiles(sample) is None:
                raise ValueError(
                    f"{property_name} is a property available for molecules and {sample} is not a valid SMILES. Please input a molecule."
                )
        else:
            # if property is available for proteins, check that sample is a FASTA
            if Chem.MolFromFASTA(sample) is None:
                raise ValueError(
                    f"{property_name} is a property available for proteins and {sample} is not a valid FASTA. Plese input a protein."
                )


class MoleculePropertyPredictorScorer(PropertyPredictorScorer):
    """Property Predictor Scorer for molecules."""

    def __init__(
        self,
        name: str,
        target: Union[float, int],
        parameters: Dict[str, Any] = {},
    ) -> None:
        """Scoring function that calculates a generic score for a property in molecules.

        Args:
            name: name of the property to score.
            target: target score that will be used to get the distance to the score of a molecule or protein (not be confused with parameters["target"]).
            parameters: parameters for the scoring function.
        """
        self.name = name
        self.target = target
        self.parameters = parameters

        if name not in MOLECULE_PROPERTY_PREDICTOR_FACTORY:
            raise ValueError(f"property {name} not available for molecules.")

        super().__init__(name=name, target=target, parameters=parameters)


class ProteinPropertyPredictorScorer(PropertyPredictorScorer):
    """Property Predictor Scorer for protein."""

    def __init__(
        self,
        name: str,
        target: Union[float, int],
        parameters: Dict[str, Any] = {},
    ) -> None:
        """Scoring function that calculates a generic score for a property in proteins.

        Args:
            name: name of the property to score.
            target: target score that will be used to get the distance to the score of a molecule or protein (not be confused with parameters["target"]).
            parameters: parameters for the scoring function.
        """
        self.name = name
        self.target = target
        self.parameters = parameters

        if name not in PROTEIN_PROPERTY_PREDICTOR_FACTORY:
            raise ValueError(f"property {name} not available for proteins.")

        super().__init__(name=name, target=target, parameters=parameters)
