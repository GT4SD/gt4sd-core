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
from pydantic import Field

from ..core import (
    CallablePropertyPredictor,
    ConfigurableCallablePropertyPredictor,
    PropertyPredictorParameters,
)
from .functions import (
    aliphatic_index,
    aromaticity,
    boman_index,
    charge,
    charge_density,
    hydrophobic_ratio,
    instability,
    isoelectric_point,
    length,
    molecular_weight,
)


# NOTE: property prediction parameters
class AmideConfiguration(PropertyPredictorParameters):
    amide: bool = Field(
        False,
        example=False,
        description="whether the sequences are C-terminally amidated.",
    )


class PhConfiguration(PropertyPredictorParameters):
    ph: float = 7.0


class AmidePhConfiguration(PropertyPredictorParameters):
    amide: bool = Field(
        False,
        example=False,
        description="whether the sequences are C-terminally amidated.",
    )
    ph: float = 7.0


# NOTE: property prediction classes
class Length(CallablePropertyPredictor):
    """Retrieves the number of residues of a protein."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=length, parameters=parameters)


class MolecularWeight(ConfigurableCallablePropertyPredictor):
    """Computes the molecular weight of a protein."""

    def __init__(self, parameters: AmideConfiguration) -> None:
        super().__init__(callable_fn=molecular_weight, parameters=parameters)


class BomanIndex(CallablePropertyPredictor):
    """Computes the Boman index of a protein (sum of solubility values of all residues)."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=boman_index, parameters=parameters)


class AliphaticIndex(CallablePropertyPredictor):
    """Computes the aliphatic index of a protein. Measure of thermal stability."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=aliphatic_index, parameters=parameters)


class HydrophobicRatio(CallablePropertyPredictor):
    """Computes the hydrophobicity of a protein, relative freq. of **A,C,F,I,L,M & V**."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=hydrophobic_ratio, parameters=parameters)


class Charge(ConfigurableCallablePropertyPredictor):
    """Computes the charge of a protein."""

    def __init__(self, parameters: AmidePhConfiguration) -> None:
        super().__init__(callable_fn=charge, parameters=parameters)


class ChargeDensity(ConfigurableCallablePropertyPredictor):
    """Computes the charge density of a protein."""

    def __init__(self, parameters: AmidePhConfiguration) -> None:
        super().__init__(callable_fn=charge_density, parameters=parameters)


class IsoelectricPoint(ConfigurableCallablePropertyPredictor):
    """Computes the isoelectric point of every residue and aggregates."""

    def __init__(self, parameters: AmideConfiguration) -> None:
        super().__init__(callable_fn=isoelectric_point, parameters=parameters)


class Aromaticity(CallablePropertyPredictor):
    """Computes aromaticity of the protein (relative frequency of Phe+Trp+Tyr)."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=aromaticity, parameters=parameters)


class Instability(CallablePropertyPredictor):
    """Calculates the protein instability."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=instability, parameters=parameters)
