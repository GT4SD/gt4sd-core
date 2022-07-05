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
from typing import Any, Callable, Union

from pydantic import BaseModel

from ..domains.materials import MacroMolecule, Property, Protein, SmallMolecule


class PropertyPredictorConfiguration(BaseModel):
    """Abstract class for property prediction in molecules and proteins."""

    pass


class PropertyPredictor(BaseModel):
    """Property predictor."""

    def __init__(self, parameters: PropertyPredictorConfiguration = None) -> None:
        """
        Args:
            parameters
        """
        self.parameters = parameters

    @staticmethod
    def from_json(json_file: str) -> PropertyPredictor:
        """Instantiate from json configuration.

        # pydantic from dict

        Args:
            json_file (str): configuration file

        Returns:
            PropertyPredictor
        """
        pass

    def to_json(self) -> str:
        """Convert instance PropertyPrediction in json configuration.

        Returns:
            str: json file
        """
        pass

    def __call__(self, sample: Any) -> Property:
        """generic call method for all properties.

        pp = PropertyPredictor(params)
        property_value = pp(sample)

        Args:
            sample:

        Returns:
            Property:
        """
        pass


class CallablePropertyPredictor(PropertyPredictor):
    def __init__(
        self, parameters: PropertyPredictorConfiguration, callable_fn: Callable
    ) -> None:
        self.callable_fn = callable_fn
        super().__init__(parameters=parameters)

    def __call__(
        self, sample: Union[SmallMolecule, MacroMolecule, Protein]
    ) -> Property:
        """generic call method for all properties.

        pp = PropertyPredictor(params)
        property_value = pp(sample)

        Args:
            sample: input string (SMILES)

        Returns:
            Property: callable function to compute a generic property
        """
        return self.callable_fn(sample)


class SmallMoleculeProperty(PropertyPredictor):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, molecule: SmallMolecule) -> Property:
        pass


class ProteinProperty(PropertyPredictor):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, protein: MacroMolecule) -> Property:
        pass
