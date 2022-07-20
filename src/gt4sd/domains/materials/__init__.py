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
"""Types, classes, validation, etc. for the material domain."""

from typing import Any, Dict, List, NewType, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol

# TODO: setting to str directly requires no wrapping, so wrong strings could be passed
Protein = str  # NewType('Protein', str)
SMILES = str  # NewType('SMILES', str)
SmallMolecule = Union[SMILES, Mol]
MacroMolecule = Union[Protein, Mol]
Omics = Union[np.ndarray, pd.Series]
PAG = SMILES
Molecule = Union[SmallMolecule, MacroMolecule]
Sequence = str


def validate_molecules(
    smiles_list: List[SMILES],
) -> Tuple[List[Chem.rdchem.Mol], List[int]]:
    """Validate molecules.

    Args:
        smiles_list: list of SMILES representing molecules.

    Returns:
        a tuple containing RDKit molecules and valid indexes.
    """
    # generate molecules from SMILES
    molecules = [
        Chem.MolFromSmiles(a_smiles, sanitize=True) for a_smiles in smiles_list
    ]
    # valid ids
    valid_ids = [
        index
        for index, molecule in enumerate(molecules)
        if molecule is not None and molecule != ""
    ]
    return molecules, valid_ids


Bounds = Tuple[int, int]  # NewType('Bounds', Tuple[int, int])

PhotoacidityCondition = NewType("PhotoacidityCondition", Bounds)
# photoacidity_condition = PhotoacidityCondition(
#     (0, 1)
# )  # PhotoacidityCondition(Bounds((0, 1))) if Bound was a new type

ConditionPAG = Union[PhotoacidityCondition]
