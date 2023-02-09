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

from enum import Enum
from typing import Any, List, NewType, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol

# TODO: setting to str directly requires no wrapping, so wrong strings could be passed
Protein = str  # NewType('Protein', str)
SMILES = str  # NewType('SMILES', str)
SELFIES = str  # NewType('SELFIES', str)
Copolymer = str  # NewType('Copolymer', str)
SmallMolecule = Union[SMILES, Mol]
MacroMolecule = Union[Protein, Mol]
Omics = Union[np.ndarray, pd.Series]
PAG = SMILES
Molecule = Union[SmallMolecule, MacroMolecule]
Sequence = str


class MoleculeFormat(str, Enum):
    selfies = "SELFIES"
    smiles = "SMILES"
    copolymer = "Copolymer"


def validate_smiles(
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


def validate_selfies(
    selfies_list: List[SELFIES],
) -> Tuple[List[SELFIES], List[int]]:
    """Validate molecules.

    Args:
        selfies_list: list of SELFIES representing molecules.

    Returns:
        a tuple containing RDKit molecules and valid indexes.
    """
    # selfies is not valid if it contains -1 so return no valid ids
    valid_ids = [i for i, s in enumerate(selfies_list) if s != -1]
    selfies = [s for i, s in enumerate(selfies_list) if i in valid_ids]
    return selfies, valid_ids


def validate_copolymer(
    copolymers_list: List[Copolymer],
) -> Tuple[List[Copolymer], List[int]]:
    """Validate copolymers.

    Args:
        copolymers_list: list of Copolymer representing molecules.

    Returns:
        a tuple containing RDKit molecules and valid indexes.
    """
    # TODO implement actual validation

    # Remove duplicates
    copolymers, idxs = [], []
    for i, copolymer in enumerate(copolymers_list):
        if copolymer not in copolymers:
            copolymers.append(copolymer)
            idxs.append(i)
    return copolymers, idxs


MOLECULE_FORMAT_VALIDATOR_FACTORY = {
    MoleculeFormat.selfies: validate_selfies,
    MoleculeFormat.smiles: validate_smiles,
    MoleculeFormat.copolymer: validate_copolymer,
}


def validate_molecules(
    pattern_list: List[str],
    input_type: str,
) -> Tuple[List[Any], List[int]]:
    """Validate molecules.

    Args:
        pattern_list: list of patterns representing molecules.
        input_type: type of patter (SELFIES, SMILES OR Copolymer).

    Returns:
        a tuple containing RDKit molecules and valid indexes.
    """
    return MOLECULE_FORMAT_VALIDATOR_FACTORY[input_type](pattern_list)  # type: ignore


Bounds = Tuple[int, int]  # NewType('Bounds', Tuple[int, int])

PhotoacidityCondition = NewType("PhotoacidityCondition", Bounds)

ConditionPAG = Union[PhotoacidityCondition]
