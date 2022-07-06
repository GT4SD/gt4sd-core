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
from typing import Any, Dict

from rdkit.Chem import Mol

from ..core import CallablePropertyPredictor, PropertyPredictor
from .core import (
    ActivityAgainstTarget,
    ActivityAgainstTargetParameters,
    Bertz,
    BertzParameters,
    Esol,
    EsolParameters,
    IsScaffold,
    IsScaffoldParameters,
    Lipinski,
    LipinskiParameters,
    Logp,
    LogpParameters,
    MolecularWeight,
    MolecularWeightParameters,
    NumberAromaticRings,
    NumberAromaticRingsParameters,
    NumberAtoms,
    NumberAtomsParameters,
    NumberHAcceptors,
    NumberHAcceptorsParameters,
    NumberHDonors,
    NumberHDonorsParameters,
    NumberHeterocycles,
    NumberHeterocyclesParameters,
    NumberLargeRings,
    NumberLargeRingsParameters,
    NumberRings,
    NumberRingsParameters,
    NumberRotatableBonds,
    NumberRotatableBondsParameters,
    NumberStereocenters,
    NumberStereocentersParameters,
    Plogp,
    PlogpParameters,
    Qed,
    QedParameters,
    Sas,
    SasParameters,
    Scscore,
    ScscoreParameters,
    SimilaritySeed,
    SimilaritySeedParameters,
    Tpsa,
    TpsaParameters,
)

# All functions can be called with either a SMILES or a Mol object.
MOLECULE_FACTORY: Dict[str, Any] = {
    # Inherent properties
    "activity_against_target": (ActivityAgainstTarget, ActivityAgainstTargetParameters),
    "molecular_weight": (MolecularWeight, MolecularWeightParameters),
    "number_of_aromatic_rings": (NumberAromaticRings, NumberAromaticRingsParameters),
    "number_of_h_acceptors": (NumberHAcceptors, NumberHAcceptorsParameters),
    "number_of_h_donors": (NumberHDonors, NumberHDonorsParameters),
    "number_of_atoms": (NumberAtoms, NumberAtomsParameters),
    "number_of_rings": (NumberRings, NumberRingsParameters),
    "number_of_rotatable_bonds": (NumberRotatableBonds, NumberRotatableBondsParameters),
    "number_of_large_rings": (NumberLargeRings, NumberLargeRingsParameters),
    "number_of_heterocycles": (NumberHeterocycles, NumberHeterocyclesParameters),
    "number_of_stereocenters": (NumberStereocenters, NumberStereocentersParameters),
    "is_scaffold": (IsScaffold, IsScaffoldParameters),
    # Rule-based properties
    "bertz": (Bertz, BertzParameters),
    "tpsa": (Tpsa, TpsaParameters),
    "logp": (Logp, LogpParameters),
    "qed": (Qed, QedParameters),
    "plogp": (Plogp, PlogpParameters),
    "penalized_logp": (Plogp, PlogpParameters),
    "lipinski": (Lipinski, LipinskiParameters),
    "sas": (Sas, SasParameters),
    "esol": (Esol, EsolParameters),
    "similarity_seed": (SimilaritySeed, SimilaritySeedParameters),
    # Properties predicted by ML models
    "scscore": (Scscore, ScscoreParameters),
}
