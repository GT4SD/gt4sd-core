from rdkit import Chem
from modlamp.descriptors import GlobalDescriptor

from ...domains.materials import MacroMolecule


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
