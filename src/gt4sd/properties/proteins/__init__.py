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
from typing import Callable, Dict, Union

from rdkit.Chem import Mol

from ...domains.materials import MacroMolecule, PropertyValue, SmallMolecule
from ..core import CallableProperty, Property
from .core import (
    AliphaticIndex,
    AmideParameter,
    AmidePhParameters,
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
PROTEIN_FACTORY: Dict[str, PropertyValue] = {
    # Inherent properties
    "length": (Length),
    "weight": (MolecularWeight),
    # Rule-based properties
    "boman_index": (BomanIndex),
    "charge_density": (ChargeDensity, AmidePhParameters),
    "charge": (Charge, AmidePhParameters),
    "aliphaticity": (AliphaticIndex),
    "hydrophobicity": (HydrophobicRatio),
    "isoelectric_point": (IsoelectricPoint, AmideParameter),
    "aromaticity": (Aromaticity),
    "instability": (Instability),
    # Properties predicted by ML models
}
