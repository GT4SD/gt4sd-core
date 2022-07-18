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
from typing import Optional
from ..domains.materials import (
    MacroMolecule,
    PropertyValue,
    Protein,
    SmallMolecule,
    Molecule,
)


class PropertyConfiguration(BaseModel):
    """Abstract class for property computation."""

    pass


class Property:
    """Property base class."""

    def __init__(self, parameters: PropertyConfiguration = None) -> None:
        """
        Args:
            parameters
        """
        self.parameters = parameters

    @staticmethod
    def from_json(json_file: str) -> PropertyConfiguration:
        """Instantiate from json configuration.

        # pydantic from dict

        Args:
            json_file (str): configuration file

        Returns:
            Property
        """
        # TODO: Not exactly sure how to
        pass

    def to_json(self) -> str:
        """Convert instance PropertyPrediction in json configuration.

        Returns:
            str: json file
        """
        # TODO
        pass

    def __call__(self, sample: Any) -> PropertyValue:
        """generic call method for all properties.

        pp = Property(params)
        property_value = pp(sample)

        Args:
            sample:

        Returns:
            Property:
        """
        raise NotImplementedError


class CallableProperty(Property):
    def __init__(
        self,
        callable_fn: Callable,
        parameters: Optional[PropertyConfiguration] = None,
    ) -> None:
        self.callable_fn = callable_fn
        super().__init__(parameters=parameters)

    def __call__(self, sample: Molecule) -> Property:
        """generic call method for all properties.

        ```python
        scorer = Property(params)
        property_value = scorer(sample)
        ```

        Args:
            sample: input string.

        Returns:
            Property: callable function to compute a generic property.
        """
        return self.callable_fn(sample)


class SmallMoleculeProperty(Property):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, molecule: SmallMolecule) -> Property:
        raise NotImplementedError


class ProteinProperty(Property):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, protein: MacroMolecule) -> Property:
        raise NotImplementedError
