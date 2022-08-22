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
import json
from typing import Any, Dict, List, Tuple, Type, Union

from .core import (
    CombinedScorer,
    IsomerScorer,
    QEDScorer,
    RDKitDescriptorScorer,
    SMARTSScorer,
    TanimotoScorer,
    TargetValueScorer,
)

SCORING_FUNCTIONS_FACTORY = {
    "target_value_scorer": TargetValueScorer,
    "combined_scorer": CombinedScorer,
    "rdkit_scorer": RDKitDescriptorScorer,
    "tanimoto_scorer": TanimotoScorer,
    "isomer_scorer": IsomerScorer,
    "smarts_scorer": SMARTSScorer,
    "qed_scorer": QEDScorer,
}


def get_target_parameters(
    target: Union[str, Dict[str, Any]]
) -> Tuple[List[Type[Any]], List[float]]:
    """Generates a tuple of scorers and weight list

    Args:
        target: scoring functions and parameters related to it

    Return:
        A tuple containing scoring functions and weight list
    """
    score_list = []
    weights = []
    target_dictionary: Dict[str, Any] = {}
    if isinstance(target, str):
        target_dictionary = json.loads(target)
    elif isinstance(target, dict):
        target_dictionary = target
    else:
        raise ValueError(
            f"{target} of type {type(target)} is not supported: provide 'str' or 'Dict[str, Any]'"
        )
    for scoring_function_name, parameters in target_dictionary.items():
        weight = 1.0
        if "weight" in parameters:
            weight = parameters.pop("weight")
        score_list.append(
            SCORING_FUNCTIONS_FACTORY[scoring_function_name](**parameters)
        )
        weights.append(weight)
    return (score_list, weights)
