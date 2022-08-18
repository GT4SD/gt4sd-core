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
from typing import Any, Dict, Union

from ...properties import PropertyPredictorRegistry
from .scorer import TargetValueScorer


class PropertyPredictorScorer(TargetValueScorer):
    def __init__(
        self,
        name: str,
        target: Union[float, int],
        parameters: Dict[str, Any] = {},
    ) -> None:
        """Scoring function that calculates a generic score for a property

        Args:
            name: name of the property to score.
            target: target score that will be used to get the distance to the score of the SMILES (not be confused with parameters["target"]).
        """
        self.target = target
        self.name = name
        self.parameters = parameters

        self.scoring_function = PropertyPredictorRegistry.get_property_predictor(  # type: ignore
            name=self.name, parameters=self.parameters
        )
        super().__init__(target=target, scoring_function=self.scoring_function)

    def predictor(self, smiles: str) -> Union[float, int, bool]:
        """Generates a prediction for a given SMILES.

        Args:
            smiles: SMILES.

        Returns:
            A score for the given SMILES
        """

        return self.scoring_function(smiles)
