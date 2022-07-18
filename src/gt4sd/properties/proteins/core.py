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

from ..core import CallableProperty, PropertyConfiguration
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

# class CallableFactory:

#     @staticmethod
#     def get(callable: Callable) -> PropertyPredictor:
#         return Callable


# Parameter classes
class AmideParameter(PropertyConfiguration):
    amide: bool = False  # whether the sequences are C-terminally amidated.


class PhParameter(PropertyConfiguration):
    ph: float = 7.0


class AmidePhParameters(AmideParameter, PhParameter):
    pass


# Property classes
class Length(CallableProperty):
    """Retrieves the number of residues of a protein."""

    def __init__(self) -> None:
        super().__init__(callable_fn=length)


class MolecularWeight(CallableProperty):
    """Computes the molecular weight of a protein."""

    def __init__(self, parameters: AmideParameter) -> None:
        super().__init__(callable_fn=molecular_weight, parameters=parameters)


class BomanIndex(CallableProperty):
    """Computes the Boman index of a protein (sum of solubility values of all residues)."""

    def __init__(self) -> None:
        super().__init__(callable_fn=boman_index)


class AliphaticIndex(CallableProperty):
    """Computes the aliphatic index of a protein. Measure of thermal stability."""

    def __init__(self) -> None:
        super().__init__(callable_fn=aliphatic_index)


class HydrophobicRatio(CallableProperty):
    """Computes the hydrophobicity of a protein, relative freq. of **A,C,F,I,L,M & V**."""

    def __init__(self) -> None:
        super().__init__(callable_fn=hydrophobic_ratio)


class Charge(CallableProperty):
    """Computes the charge of a protein."""

    def __init__(self, parameters: AmidePhParameters) -> None:
        super().__init__(callable_fn=charge, parameters=parameters)


class ChargeDensity(CallableProperty):
    """Computes the charge density of a protein."""

    def __init__(self, parameters: AmidePhParameters) -> None:
        super().__init__(callable_fn=charge_density, parameters=parameters)


class IsoelectricPoint(CallableProperty):
    """Computes the isoelectric point of every residue and aggregates."""

    def __init__(self, parameters: AmideParameter) -> None:
        super().__init__(callable_fn=isoelectric_point, parameters=parameters)


class Aromaticity(CallableProperty):
    """Computes aromaticity of the protein (relative frequency of Phe+Trp+Tyr)."""

    def __init__(self) -> None:
        super().__init__(callable_fn=aromaticity)


class Instability(CallableProperty):
    """Calculates the protein instability."""

    def __init__(self) -> None:
        super().__init__(callable_fn=instability)
