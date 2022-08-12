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
from guacamol.utils.descriptors import bertz as _bertz
from guacamol.utils.descriptors import (
    logP,
    mol_weight,
    num_aromatic_rings,
    num_atoms,
    num_H_acceptors,
    num_H_donors,
    num_rings,
    num_rotatable_bonds,
)
from guacamol.utils.descriptors import qed as _qed
from guacamol.utils.descriptors import tpsa as _tpsa
from paccmann_generator.drug_evaluators import (
    ESOL,
    SAS,
    Lipinski,
    PenalizedLogP,
    SCScore,
)
from rdkit.Chem.rdMolDescriptors import CalcNumAtomStereoCenters, CalcNumHeterocycles
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from ...domains.materials import SmallMolecule
from ..utils import to_mol, to_smiles

# Instantiate classes for faster inference
_sas = SAS()
_sccore = SCScore(score_scale=5, fp_len=1024, fp_rad=2)
_esol = ESOL()
_lipinski = Lipinski()
_penalized_logp = PenalizedLogP()


def plogp(mol: SmallMolecule) -> float:
    """Calculate the penalized logP of a molecule. This is the logP minus the number of
    rings with > 6 atoms minus the SAS.

    Gómez-Bombarelli, R., Wei, J. N., Duvenaud, D., Hernández-Lobato, J. M., Sánchez-Lengeling, B., Sheberla, D., ... & Aspuru-Guzik, A. (2018).
    Automatic chemical design using a data-driven continuous representation of molecules.
    ACS central science, 4(2), 268-276.

    NOTE: Check the initial arXiv for the plogp reference: https://arxiv.org/abs/1610.02415v1
    """
    return _penalized_logp(mol)


def lipinski(mol: SmallMolecule) -> int:
    """
    Calculate whether a molecule adheres to the Lipinski-rule-of-5.
    A crude approximation of druglikeness.

    Lipinski, C. A., Lombardo, F., Dominy, B. W., & Feeney, P. J. (1997).
    Experimental and computational approaches to estimate solubility and permeability in
    drug discovery and development settings.
    Advanced drug delivery reviews, 23(1-3), 3-25.
    """
    return int(_lipinski(mol)[0])


def esol(mol: SmallMolecule) -> float:
    """Estimate the water solubility of a molecule.

    Delaney, J. S. (2004).
    ESOL: estimating aqueous solubility directly from molecular structure.
    Journal of chemical information and computer sciences, 44(3), 1000-1005.

    """
    return _esol(mol)


def scscore(mol: SmallMolecule) -> float:
    """Calculate the synthetic complexity score (SCScore) of a molecule.

    Coley, C. W., Rogers, L., Green, W. H., & Jensen, K. F. (2018).
    SCScore: synthetic complexity learned from a reaction corpus.
    Journal of chemical information and modeling, 58(2), 252-261.

    """
    return _sccore(mol)


def sas(mol: SmallMolecule) -> float:
    """Calculate the synthetic accessibility score (SAS) for a molecule.

    Ertl, P., & Schuffenhauer, A. (2009).
    Estimation of synthetic accessibility score of drug-like molecules based on molecular
    complexity and fragment contributions.
    Journal of cheminformatics, 1(1), 1-11.
    """
    return _sas(mol)


def bertz(mol: SmallMolecule) -> float:
    """Calculate Bertz index of a molecule.

    Bertz, S. H. (1981).
    The first general index of molecular complexity.
    Journal of the American Chemical Society, 103(12), 3599-3601.
    """
    return _bertz(to_mol(mol))


def tpsa(mol: SmallMolecule) -> float:
    """
    Calculate the total polar surface area of a molecule.

    Ertl, P., Rohde, B., & Selzer, P. (2000).
    Fast calculation of molecular polar surface area as a sum of fragment-based
    contributions and its application to the prediction of drug transport properties.
    Journal of medicinal chemistry, 43(20), 3714-3717.
    """
    return _tpsa(to_mol(mol))


def logp(mol: SmallMolecule) -> float:
    """
    Calculates the partition coefficient of a molecule.

    Wildman, S. A., & Crippen, G. M. (1999).
    Prediction of physicochemical parameters by atomic contributions.
    Journal of chemical information and computer sciences, 39(5), 868-873.

    """
    return logP(to_mol(mol))


def qed(mol: SmallMolecule) -> float:
    """
    Calculate the quantitative estimate of drug-likeness (QED) of a molecule.

    Bickerton, G. R., Paolini, G. V., Besnard, J., Muresan, S., & Hopkins, A. L. (2012).
    Quantifying the chemical beauty of drugs.
    Nature chemistry, 4(2), 90-98.
    """
    return _qed(to_mol(mol))


def number_of_h_acceptors(mol: SmallMolecule) -> int:
    """Calculate number of H acceptors of a molecule."""
    return num_H_acceptors(to_mol(mol))


def number_of_atoms(mol: SmallMolecule) -> int:
    """Calculate number of atoms of a molecule."""
    return num_atoms(to_mol(mol))


def number_of_h_donors(mol: SmallMolecule) -> int:
    """Calculate number of H donors of a molecule."""
    return num_H_donors(to_mol(mol))


def number_of_aromatic_rings(mol: SmallMolecule) -> int:
    """Calculate number of aromatic rings of a molecule."""
    return num_aromatic_rings(to_mol(mol))


def number_of_rings(mol: SmallMolecule) -> int:
    """Calculate number of rings of a molecule."""
    return num_rings(to_mol(mol))


def number_of_rotatable_bonds(mol: SmallMolecule) -> int:
    """Calculate number of rotatable bonds of a molecule."""
    return num_rotatable_bonds(to_mol(mol))


def number_of_large_rings(mol: SmallMolecule) -> int:
    """Calculate the amount of large rings (> 6 atoms) of a molecule."""
    mol = to_mol(mol)
    ringinfo = mol.GetRingInfo()
    return len([x for x in ringinfo.AtomRings() if len(x) > 6])


def molecular_weight(mol: SmallMolecule) -> float:
    """Calculate molecular weight of a molecule."""
    return mol_weight(to_mol(mol))


def is_scaffold(mol: SmallMolecule) -> int:
    """Whether a molecule is identical to its Murcko Scaffold."""
    mol = to_mol(mol)
    smi = Chem.MolToSmiles(mol, canonical=True)
    return int(smi == MurckoScaffoldSmiles(mol=mol))


def number_of_heterocycles(mol: SmallMolecule) -> int:
    """The amount of heterocycles of a molecule."""
    return CalcNumHeterocycles(to_mol(mol))


def number_of_stereocenters(mol: SmallMolecule) -> int:
    """The amount of stereo centers of a molecule."""
    return CalcNumAtomStereoCenters(to_mol(mol))


def similarity_to_seed(
    mol: SmallMolecule, similarity_fn: Callable[[SmallMolecule], float]
) -> float:
    """Calculate the similarity of a molecule to a seed molecule.

    Example:
        An example::

            from gt4sd.properties.molecules import similarity_to_seed, get_similarity_fn
            func = get_similarity_fn(target_mol='CCO', fp_key='FCFP4')
            similarity_to_seed(mol='CCC', similarity_fn=func)
    """
    return similarity_fn(to_smiles(mol))


def activity_against_target(
    mol: SmallMolecule, affinity_fn: Callable[[SmallMolecule], float]
) -> float:
    """Calculate the activity of a molecule against a target molecule.

    Example:
        An example::

            from gt4sd.properties.molecules import activity_against_target, get_activity_fn
            func = get_activity_fn(target_mol='DRD2')
            activity_against_target(mol='CCC', affinity_fn=func)
    """
    return affinity_fn(to_smiles(mol))
