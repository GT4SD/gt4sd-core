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
from typing import Callable

from rdkit import Chem
from modlamp.descriptors import GlobalDescriptor
from tdc import Oracle
from tdc.chem_utils.oracle.oracle import fp2fpfunc
from tdc.metadata import download_oracle_names

from ..domains.materials import MacroMolecule, SmallMolecule
from .core import PropertyValue


def to_mol(mol: SmallMolecule) -> Chem.Mol:
    """Safely convert a string or a rdkit.Chem.Mol to a rdkit.Chem.Mol.

    Args:
        mol: a string or a rdkit.Chem.Mol object.

    Raises:
        TypeError: if wrong type is given.

    Returns:
        a rdkit.Chem.Mol object.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    elif isinstance(mol, Chem.Mol):
        pass
    else:
        raise TypeError(
            f"Please provide SMILES string or rdkit.Chem.Mol object not {type(mol)}"
        )
    return mol


def to_smiles(mol: SmallMolecule) -> str:
    """Safely returns a SMILES string from a SMILES or a rdkit.Chem.Mol object.

    Args:
        SmallMolecule: either a SMILES or a rdkit.Chem.Mol object.

    Returns:
        a SMILES string.
    """
    if isinstance(mol, str):
        try:
            mol = Chem.MolFromSmiles(mol)
        except Exception:
            raise ValueError(
                f"Could not convert SMILES string to rdkit.Chem.Mol: {mol}"
            )
    elif isinstance(mol, Chem.Mol):
        pass
    else:
        raise TypeError(
            f"Pass a SMILES string or rdkit.Chem.Mol object not {type(mol)}"
        )

    return Chem.MolToSmiles(mol, canonical=True)


def get_similarity_fn(
    target_mol: SmallMolecule, fp_key: str = "FCFP4"
) -> Callable[[SmallMolecule], PropertyValue]:
    """Get a similarity function for a target molecule.

    Args:
        target_mol: a target molecule as SMILES or rdkit.Chem.Mol object.
        fp_key: The type of fingerprint to use. One of `ECFP4`, `ECFP6`, `FCFP4` and `AP`.

    Returns:
        a similarity function that can be called with a `SmallMolecule`.
    """
    if fp_key not in fp2fpfunc.keys():
        raise ValueError(f"Choose fp_key from {fp2fpfunc.keys()}.")
    target_smiles = to_smiles(target_mol)
    return Oracle(name="similarity_meta", target_smiles=target_smiles, fp=fp_key)


def get_activity_fn(target: str) -> Callable[[SmallMolecule], PropertyValue]:
    """Get a function to measure activity/affinity against a protein target.

    Args:
        target: name of the target protein.

    Returns:
        an affinity function that can be called with a `SmallMolecule`.
    """
    if target not in download_oracle_names:
        raise ValueError(
            f"Supported targets are: {download_oracle_names}, not {target}"
        )
    return Oracle(name=target)


# for proteins
def get_sequence(protein: MacroMolecule) -> str:
    """Safely returns an amino acid sequence of a macromolecule

    Args:
        protein: either an AA sequence or a rdkit.Chem.Mol object that can be converted to FASTA.

    Raises:
        TypeError: if the input was none of the above types.
        ValueError: if the sequence was empty or could not be parsed into FASTA.

    Returns:
        an AA sequence.
    """
    if isinstance(protein, str):
        seq = protein.upper().strip()
        return seq
    elif isinstance(protein, Chem.Mol):
        seq = Chem.MolToFASTA(protein).split()
    else:
        raise TypeError(f"Pass a string or rdkit.Chem.Mol object not {type(protein)}")
    if seq == []:
        raise ValueError(
            f"Sequence was empty or rdkit.Chem.Mol could not be converted: {protein}"
        )
    return seq[-1]


def get_descriptor(protein: MacroMolecule) -> GlobalDescriptor:
    """Convert a macromolecule to a modlamp GlobalDescriptor object.

    Args:
        protein: either an AA sequence or a rdkit.Chem.Mol object that can be converted to FASTA.

    Returns:
        GlobalDescriptor object.
    """
    seq = get_sequence(protein)
    return GlobalDescriptor(seq)
