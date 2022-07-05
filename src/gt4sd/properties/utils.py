from typing import Callable

from rdkit import Chem
from modlamp.descriptors import GlobalDescriptor
from tdc import Oracle
from tdc.chem_utils.oracle.oracle import fp2fpfunc
from tdc.metadata import download_oracle_names

from .core import MacroMolecule, Property, SmallMolecule


def to_mol(mol: SmallMolecule) -> Chem.Mol:
    """
    Safely convert a string or a `rdkit.Chem.Mol` to a `rdkit.Chem.Mol`.

    Args:
        mol: A string or a `rdkit.Chem.Mol` object.

    Raises:
        TypeError: If wrong type is given.

    Returns:
        A `rdkit.Chem.Mol` object.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    elif isinstance(mol, Chem.Mol):
        pass
    else:
        raise TypeError(f"Please provide SMILES string or Mol object not {type(mol)}")
    return mol


def to_smiles(mol: SmallMolecule) -> str:
    """
    Safely returns a SMILES string from a SMILES or a `rdkit.Chem.Mol` object.

    Args:
        SmallMolecule: Either a SMILES or a rdkit.Chem.Mol object.

    Returns:
        A SMILES string.
    """
    if isinstance(mol, str):
        try:
            mol = Chem.MolFromSmiles(mol)
        except Exception:
            raise ValueError(f"Could not convert SMILES string to Mol: {mol}")
    elif isinstance(mol, Chem.Mol):
        pass
    else:
        raise TypeError(f"Pass a SMILES string or Mol object not {type(mol)}")

    return Chem.MolToSmiles(mol, canonical=True)


def get_similarity_fn(
    target_mol: SmallMolecule, fp_key: str = "FCFP4"
) -> Callable[[SmallMolecule], Property]:
    """
    Get a similarity function for a target molecule.

    Args:
        target_mol: A target molecule as SMILES or `rdkit.Chem.Mol` object.
        fp_key: The type of fingerprint to use. One of `ECFP4`, `ECFP6`, `FCFP4` and `AP`.

    Returns:
        A similarity function that can be called with a `SmallMolecule`.
    """
    if fp_key not in fp2fpfunc.keys():
        raise ValueError(f"Choose `fp_key` from {fp2fpfunc.keys()}.")
    target_smiles = to_smiles(target_mol)
    return Oracle(name="similarity_meta", target_smiles=target_smiles, fp=fp_key)


def get_activity_fn(target: str) -> Callable[[SmallMolecule], Property]:
    """
    Get a function to measure activity/affinity against a protein target.

    Args:
        target: Name of the target protein.

    Returns:
        An affinity function that can be called with a `SmallMolecule`.
    """
    if target not in download_oracle_names:
        raise ValueError(
            f"Supported targets are: {download_oracle_names}, not {target}"
        )
    return Oracle(name=target)


# for proteins
def get_sequence(protein: MacroMolecule) -> str:
    """
    Safely returns an amino acid sequence of a macromolecule

    Args:
        protein: Either an AAS or a rdkit.Chem.Mol object that can be converted to FASTA.

    Raises:
        TypeError: If the input was none of the above types.
        ValueError: If the sequence was empty or could not be parsed into FASTA.

    Returns:
        An AAS.
    """
    if isinstance(protein, str):
        seq = protein.upper().strip()
    elif isinstance(protein, Chem.Mol):
        seq = Chem.MolToFASTA(protein).split()
    else:
        raise TypeError(f"Pass a string or Mol object not {type(protein)}")
    if seq == []:
        raise ValueError(f"Sequence was empty or mol could not be converted: {protein}")
    return seq[-1]


def get_descriptor(protein: MacroMolecule) -> GlobalDescriptor:
    """
    Convert a macromolecule to a modlamp GlobalDescriptor object.

    Args:
        protein: Either an AAS or a rdkit.Chem.Mol object that can be converted to FASTA.

    Returns:
        GlobalDescriptor object.
    """
    seq = get_sequence(protein)
    return GlobalDescriptor(seq)
