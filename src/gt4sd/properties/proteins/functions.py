from ...domains.materials import MacroMolecule
from .utils import get_descriptor


def length(protein: MacroMolecule) -> int:
    """Retrieves the number of residues of a protein."""
    desc = get_descriptor(protein)
    desc.length()
    return int(desc.descriptor)


def boman_index(protein: MacroMolecule) -> float:
    """Computes the Boman index of a protein (sum of solubility values of all residues).

    Boman, H. G. (2003).
    Antibacterial peptides: basic facts and emerging concepts.
    Journal of internal medicine, 254(3), 197-215.
    """
    desc = get_descriptor(protein)
    desc.boman_index()
    return float(desc.descriptor)


def aliphatic_index(protein: MacroMolecule) -> float:
    """Computes the aliphatic index of a protein. Measure of thermal stability.

    Ikai, A. (1980).
    Thermostability and aliphatic index of globular proteins.
    The Journal of Biochemistry, 88(6), 1895-1898.
    """
    desc = get_descriptor(protein)
    desc.aliphatic_index()
    return float(desc.descriptor)


def hydrophobic_ratio(protein: MacroMolecule) -> float:
    """Computes the hydrophobicity of a protein, relative freq. of **A,C,F,I,L,M & V**."""
    desc = get_descriptor(protein)
    desc.hydrophobic_ratio()
    return float(desc.descriptor)


def charge(protein: MacroMolecule, ph: float = 7.4, amide: bool = True) -> float:
    """Computes the charge of a protein.

    Bjellqvist, B., Hughes, G. J., Pasquali, C., Paquet, N., Ravier, F., Sanchez, J. C., ... & Hochstrasser, D. (1993).
    The focusing positions of polypeptides in immobilized pH gradients can be predicted from their amino acid sequences.
    Electrophoresis, 14(1), 1023-1031.
    """
    desc = get_descriptor(protein)
    desc.calculate_charge(ph=ph, amide=amide)
    return float(desc.descriptor)


def charge_density(
    protein: MacroMolecule, ph: float = 7.4, amide: bool = True
) -> float:
    """Computes the charge density of a protein.

    Bjellqvist, B., Hughes, G. J., Pasquali, C., Paquet, N., Ravier, F., Sanchez, J. C., ... & Hochstrasser, D. (1993).
    The focusing positions of polypeptides in immobilized pH gradients can be predicted from their amino acid sequences.
    Electrophoresis, 14(1), 1023-1031.
    """
    desc = get_descriptor(protein)
    desc.charge_density(ph=ph, amide=amide)
    return float(desc.descriptor)


def isoelectric_point(protein: MacroMolecule, amide: bool = True) -> float:
    """Computes the isoelectric point of every residue and aggregates."""
    desc = get_descriptor(protein)
    desc.isoelectric_point(amide=amide)
    return float(desc.descriptor)


def aromaticity(protein: MacroMolecule) -> float:
    """Computes aromaticity of the protein (relative frequency of Phe+Trp+Tyr).

    Lobry, J. R., & Gautier, C. (1994).
    Hydrophobicity, expressivity and aromaticity are the major trends of amino-acid usage
        in 999 Escherichia coli chromosome-encoded genes.
    Nucleic acids research, 22(15), 3174-3180.
    """
    desc = get_descriptor(protein)
    desc.aromaticity()
    return float(desc.descriptor)


def instability(protein: MacroMolecule) -> float:
    """Calculates the protein instability.

    Guruprasad, K., Reddy, B. B., & Pandit, M. W. (1990).
    Correlation between stability of a protein and its dipeptide composition: a novel
        approach for predicting in vivo stability of a protein from its primary sequence.
    Protein Engineering, Design and Selection, 4(2), 155-161.
    """
    desc = get_descriptor(protein)
    desc.instability_index()
    return float(desc.descriptor)
