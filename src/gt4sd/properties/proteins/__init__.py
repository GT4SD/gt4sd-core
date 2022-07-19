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
from typing import Callable, Dict, Union, Tuple

from rdkit.Chem import Mol

from ...domains.materials import MacroMolecule, PropertyValue, SmallMolecule
from ..core import CallableProperty, Property, PropertyConfiguration
from .core import (
    AliphaticIndex,
    AmideConfiguration,
    AmidePhConfiguration,
    Aromaticity,
    BomanIndex,
    Charge,
    ChargeDensity,
    HydrophobicRatio,
    Instability,
    IsoelectricPoint,
    Length,
    MolecularWeight,
)

# All functions can be called with either a SMILES or a Mol object.
PROTEIN_FACTORY: Dict[
    str, Union[CallableProperty, Tuple[CallableProperty, PropertyConfiguration]]
] = {
    # Inherent properties
    "length": (Length, PropertyConfiguration),
    "weight": (MolecularWeight, AmideConfiguration),
    # Rule-based properties
    "boman_index": (BomanIndex, PropertyConfiguration),
    "charge_density": (ChargeDensity, AmidePhConfiguration),
    "charge": (Charge, AmidePhConfiguration),
    "aliphaticity": (AliphaticIndex, PropertyConfiguration),
    "hydrophobicity": (HydrophobicRatio, PropertyConfiguration),
    "isoelectric_point": (IsoelectricPoint, AmideConfiguration),
    "aromaticity": (Aromaticity, PropertyConfiguration),
    "instability": (Instability, PropertyConfiguration),
    # Properties predicted by ML models
}