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
"""Implementation of scorers."""
from functools import partial
from typing import Any, Callable, Dict, List, Type

import numpy as np
from rdkit import Chem
from guacamol.common_scoring_functions import (
    IsomerScoringFunction,
    RdkitScoringFunction,
    SMARTSScoringFunction,
    TanimotoScoringFunction,
)
from guacamol.score_modifier import (
    ClippedScoreModifier,
    GaussianModifier,
    MaxGaussianModifier,
    MinGaussianModifier,
)
from guacamol.scoring_function import ScoringFunction
from guacamol.utils.descriptors import (
    bertz,
    logP,
    mol_weight,
    num_aromatic_rings,
    num_rings,
    num_rotatable_bonds,
    qed,
    tpsa,
)

MODIFIERS: Dict[str, Callable[..., Any]] = {
    "gaussian_modifier": GaussianModifier,
    "min_gaussian_modifier": MinGaussianModifier,
    "max_gaussian_modifier": MaxGaussianModifier,
    "clipped_score_modifier": ClippedScoreModifier,
}
MODIFIERS_PARAMETERS: Dict[str, Dict[str, float]] = {
    "gaussian_modifier": {"mu": 2, "sigma": 0.5},
    "min_gaussian_modifier": {"mu": 0.75, "sigma": 0.1},
    "max_gaussian_modifier": {"mu": 100, "sigma": 10},
    "clipped_score_modifier": {"upper_x": 0.8},
}
DESCRIPTOR: Dict[str, Callable[..., Any]] = {
    "num_rotatable_bonds": num_rotatable_bonds,
    "num_aromatic_rings": num_aromatic_rings,
    "log_p": logP,
    "tpsa": tpsa,
    "bertz": bertz,
    "qed": qed,
    "mol_weight": mol_weight,
    "num_rings": num_rings,
}


def distance_to_score(distance: float, beta: float) -> float:
    """calculating exponential for a given distance

    Args:
        distance: A float.

    Returns:
        An exponential score value for a given SMILES
    """
    return np.exp(-beta * distance**2)


class DistanceScorer(ScoringFunction):
    def __init__(self, beta: float = 0.00000001) -> None:
        """DistanceScorer is used to call a partial copy of distance_to_score function.

        Args:
            beta: A float value used for getting an exponential score value
        """
        self.partical_distance_score = partial(distance_to_score, beta=beta)

    def get_distance(self, smile_distance: float) -> float:
        """Generates a partial copy of distance_to_score function

        Args:
            smiles: SMILES.

        Returns:
            An exponential score value for a given SMILES
        """
        return self.partical_distance_score(smile_distance)


class TargetValueScorer(DistanceScorer):
    def __init__(self, target: float, scoring_function: Callable[[str], float]) -> None:
        """Scoring function which is used to generate a score based on a taget and a scoring function.

        Args:
            target: target score that will be used to get the distance to the score of the SMILES
            scoring_function: an instance of a scoring class
        """
        super().__init__()
        self.target = target
        self.scoring_function = scoring_function

    def score(self, smiles: str) -> float:
        """Generates a score for a given SMILES

        Args:
            smiles: SMILES.

        Returns:
            A score for the given SMILES
        """
        return self.get_distance(self.scoring_function(smiles) - self.target)

    def score_list(self, smiles_list: List[str]) -> List[float]:
        """Generates a list of scores for a given SMILES List

        Args:
            smiles_list: A List of SMILES.

        Returns:
            A List of scores
        """
        return [
            self.score(smiles)
            for smiles in smiles_list
            if Chem.MolFromSmiles(smiles) and smiles
        ]


class CombinedScorer:
    def __init__(
        self,
        scorer_list: List[Type[Any]],
        weights: List[float] = None,
    ) -> None:
        """Scoring function which generates a combined score for a SMILES as per the given scoring functions.

        Args:
            scorer_list: A list of the scoring functions
            weights: A list of weights
        """
        self.scorer_list = scorer_list
        self.weights = self._normalize_weights(weights)

    def _normalize_weights(self, weights=None) -> List[float]:
        """It is used for normalizing weights.

        Args:
            weights: A list of weights.

        Returns:
            Sum of all the scores generated by the given scoring functions
        """
        weights = weights if weights else [1.0] * len(self.scorer_list)
        offsetted_weights = [weight + min(weights) for weight in weights]
        return [weight / float(sum(offsetted_weights)) for weight in offsetted_weights]

    def score(self, smiles: str):
        """Generates a score for a given SMILES

        Args:
            smiles: SMILES.

        Returns:
            Sum of all the scores generated by the given scoring functions
        """
        return sum(
            [
                scorer.score(smiles) * weight
                for scorer, weight in zip(self.scorer_list, self.weights)
            ]
        )

    def score_list(self, smiles_list: List[str]) -> List[float]:
        """Generates a list of scores for a given SMILES List

        Args:
            smiles_list: A List of SMILES.

        Returns:
            A List of scores
        """
        return [self.score(smiles) for smiles in smiles_list]


class RDKitDescriptorScorer(TargetValueScorer):
    def __init__(
        self,
        target: float,
        modifier: str = "gaussian_modifier",
        descriptor: str = "num_rotatable_bonds",
    ) -> None:
        """Scoring function wrapping RDKit descriptors.

        Args:
            target: target score that will be used to get the distance to the score of the SMILES
            modifier: score modifier
            descriptor:  molecular descriptors
        """
        self.target = target
        self.modifier = MODIFIERS[modifier](**MODIFIERS_PARAMETERS[modifier])
        self.descriptor = DESCRIPTOR[descriptor]
        super().__init__(target=target, scoring_function=self.score)

    def score(self, smiles: str) -> float:
        """Generates a score for a given SMILES

        Args:
            smiles: SMILES.

        Returns:
            A score for the given SMILES
        """
        scoring_function = RdkitScoringFunction(
            descriptor=self.descriptor,
            score_modifier=self.modifier,
        )
        return scoring_function.score_mol(Chem.MolFromSmiles(smiles))


class TanimotoScorer(TargetValueScorer):
    def __init__(
        self,
        target: float,
        target_smile: str,
        fp_type: str = "ECFP4",
        modifier: str = "gaussian_modifier",
    ) -> None:
        """Scoring function that looks at the fingerprint similarity against a target molecule.

        Args:
            target: target score that will be used to get the distance to the score of the SMILES
            target_smile: target molecule to compare similarity
            fp_type: fingerprint type
            modifier: score modifier
        """
        self.target = target
        self.target_smile = target_smile
        self.fp_type = fp_type
        self.modifier = MODIFIERS[modifier](**MODIFIERS_PARAMETERS[modifier])
        super().__init__(target=target, scoring_function=self.score)

    def score(self, smiles: str) -> float:
        """Generates a score for a given SMILES

        Args:
            smiles: SMILES.

        Returns:
            A score for the given SMILES
        """
        scoring_function = TanimotoScoringFunction(
            self.target_smile,
            fp_type=self.fp_type,
            score_modifier=self.modifier,
        )
        return scoring_function.score_mol(Chem.MolFromSmiles(smiles))


class IsomerScorer(TargetValueScorer):
    def __init__(self, target: float, target_smile: str) -> None:
        """Scoring function for closeness to a molecular formula.

        Args:
            target: target score that will be used to get the distance to the score of the SMILES
            target_smile: targeted SMILES to compare closeness with
        """
        self.target = target
        self.target_smile = target_smile
        super().__init__(target=target, scoring_function=self.score)

    def score(self, smiles: str) -> float:
        """Generates a score for a given SMILES

        Args:
            smiles: SMILES.

        Returns:
            A score for the given SMILES
        """
        scoring_function = IsomerScoringFunction(self.target_smile)
        return scoring_function.raw_score(smiles)


class SMARTSScorer(TargetValueScorer):
    def __init__(self, target: float, target_smile: str, inverse: bool = True) -> None:
        """Scoring function that looks at the fingerprint similarity against a target molecule.

        Args:
            target: target score that will be used to get the distance to the score of the SMILES
            target_smile: The SMARTS string to match
            inverse: If True then SMARTS is desired else it is not desired in the molecules
        """
        self.target = target
        self.target_smile = target_smile
        self.inverse = inverse
        super().__init__(target=target, scoring_function=self.score)

    def score(self, smiles: str) -> float:
        """Generates a score for a given SMILES

        Args:
            smiles: SMILES.

        Returns:
            A score for the given SMILES
        """
        scoring_function = SMARTSScoringFunction(self.target_smile, self.inverse)
        return scoring_function.score_mol(Chem.MolFromSmiles(smiles))


class QEDScorer(TargetValueScorer):
    def __init__(self, target: float) -> None:
        """Scoring function that calculates the weighted sum of ADS mapped properties using QED module of rdkit

        Args:
            target: target score that will be used to get the distance to the score of the SMILES
        """
        self.target = target
        super().__init__(target=target, scoring_function=self.score)

    def score(self, smiles: str) -> float:
        """Generates a score for a given SMILES

        Args:
            smiles: SMILES.

        Returns:
            A score for the given SMILES
        """
        return Chem.QED.qed(Chem.MolFromSmiles(smiles))
