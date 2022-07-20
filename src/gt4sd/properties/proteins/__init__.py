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
from typing import Dict, Tuple, Type

from ..core import PropertyPredictor, PropertyPredictorParameters
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

# NOTE: all functions can be called with either an AA sequence or a rdkit.Chem.Mol object.
PROTEIN_PROPERTY_PREDICTOR_FACTORY: Dict[
    str, Tuple[Type[PropertyPredictor], Type[PropertyPredictorParameters]]
] = {
    # inherent properties
    "length": (Length, PropertyPredictorParameters),
    "protein_weight": (MolecularWeight, AmideConfiguration),
    # rule-based properties
    "boman_index": (BomanIndex, PropertyPredictorParameters),
    "charge_density": (ChargeDensity, AmidePhConfiguration),
    "charge": (Charge, AmidePhConfiguration),
    "aliphaticity": (AliphaticIndex, PropertyPredictorParameters),
    "hydrophobicity": (HydrophobicRatio, PropertyPredictorParameters),
    "isoelectric_point": (IsoelectricPoint, AmideConfiguration),
    "aromaticity": (Aromaticity, PropertyPredictorParameters),
    "instability": (Instability, PropertyPredictorParameters),
    # properties predicted by ML models
}
