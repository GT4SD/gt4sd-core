"""Types, classes, validation, etc. for the material domain."""

from typing import List, NewType, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem

from gt4sd.exceptions import InvalidItem

# TODO setting to str directly requires no wrapping, so wrong strings could be passed
Protein = str  # NewType('Protein', str)
SMILES = str  # NewType('SMILES', str)
SmallMolecule = SMILES
Omics = Union[np.ndarray, pd.Series]
PAG = SMILES
Molecule = Union[SmallMolecule, Protein]
Sequence = str
Property = float


def check_smiles(smiles: SMILES):
    try:
        pass  # TODO
    except Exception:
        raise InvalidItem(title="invalid SMILES", detail="Validation as SMILES failed.")


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
        index for index, molecule in enumerate(molecules) if molecule is not None
    ]
    return molecules, valid_ids


Bounds = Tuple[int, int]  # NewType('Bounds', Tuple[int, int])

PhotoacidityCondition = NewType("PhotoacidityCondition", Bounds)
# photoacidity_condition = PhotoacidityCondition(
#     (0, 1)
# )  # PhotoacidityCondition(Bounds((0, 1))) if Bound was a new type

ConditionPAG = Union[PhotoacidityCondition]
