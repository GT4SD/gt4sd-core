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
from typing import Dict

from rdkit.Chem import Mol
from ...domains.materials import PropertyValue
from ..core import CallableProperty, Property, PropertyConfiguration
from .core import (
    ActivityAgainstTarget,
    ActivityAgainstTargetParameters,
    Bertz,
    Esol,
    IsScaffold,
    Lipinski,
    Logp,
    MolecularWeight,
    NumberAromaticRings,
    NumberAtoms,
    NumberHAcceptors,
    NumberHDonors,
    NumberHeterocycles,
    NumberLargeRings,
    NumberRings,
    NumberRotatableBonds,
    NumberStereocenters,
    Plogp,
    Qed,
    Sas,
    Scscore,
    ScscoreConfiguration,
    SimilaritySeed,
    SimilaritySeedParameters,
    Tpsa,
)

# All functions can be called with either a SMILES or a Mol object.
MOLECULE_FACTORY: Dict[str, PropertyValue] = {
    # Inherent properties
    "weight": (MolecularWeight, PropertyConfiguration),
    "number_of_aromatic_rings": (NumberAromaticRings, PropertyConfiguration),
    "number_of_h_acceptors": (NumberHAcceptors, PropertyConfiguration),
    "number_of_h_donors": (NumberHDonors, PropertyConfiguration),
    "number_of_atoms": (NumberAtoms, PropertyConfiguration),
    "number_of_rings": (NumberRings, PropertyConfiguration),
    "number_of_rotatable_bonds": (NumberRotatableBonds, PropertyConfiguration),
    "number_of_large_rings": (NumberLargeRings, PropertyConfiguration),
    "number_of_heterocycles": (NumberHeterocycles, PropertyConfiguration),
    "number_of_stereocenters": (NumberStereocenters, PropertyConfiguration),
    "is_scaffold": (IsScaffold, PropertyConfiguration),
    # Rule-based properties
    "bertz": (Bertz, PropertyConfiguration),
    "tpsa": (Tpsa, PropertyConfiguration),
    "logp": (Logp, PropertyConfiguration),
    "qed": (Qed, PropertyConfiguration),
    "plogp": (Plogp, PropertyConfiguration),
    "penalized_logp": (Plogp, PropertyConfiguration),
    "lipinski": (Lipinski, PropertyConfiguration),
    "sas": (Sas, PropertyConfiguration),
    "esol": (Esol, PropertyConfiguration),
    "similarity_seed": (SimilaritySeed, SimilaritySeedParameters),
    # Properties predicted by ML models
    "scscore": (Scscore, ScscoreConfiguration),
    "activity_against_target": (ActivityAgainstTarget, ActivityAgainstTargetParameters),
}
