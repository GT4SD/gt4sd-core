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
from typing import Callable, Dict

from rdkit.Chem import Mol

from ...domains.materials import Property, MacroMolecule
from .core import (
    length,
    boman_index,
    charge_density,
    charge,
    aliphatic_index,
    hydrophobic_ratio,
    isoelectric_point,
    aromaticity,
    instability,
)

# All functions can be called with either a SMILES or a Mol object.
PROTEIN_FACTORY: Dict[str, Callable[[MacroMolecule], Property]] = {
    # Inherent properties
    'length': length,
    # Rule-based properties
    'boman_index': boman_index,
    'charge_density': charge_density,
    'charge': charge,
    'aliphaticity': aliphatic_index,
    'hydrophobicity': hydrophobic_ratio,
    'isoelectric_point': isoelectric_point,
    'aromaticity': aromaticity,
    'instability': instability,
    # Properties predicted by ML models
}
