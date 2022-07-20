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
from typing import Dict, Tuple, Type

from ..core import PropertyPredictor, PropertyPredictorParameters
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

# NOTE: all functions can be called with either a SMILES or a rdkit.Chem.Mol object.
MOLECULE_PROPERTY_PREDICTOR_FACTORY: Dict[
    str, Tuple[Type[PropertyPredictor], Type[PropertyPredictorParameters]]
] = {
    # inherent properties
    "molecular_weight": (MolecularWeight, PropertyPredictorParameters),
    "number_of_aromatic_rings": (NumberAromaticRings, PropertyPredictorParameters),
    "number_of_h_acceptors": (NumberHAcceptors, PropertyPredictorParameters),
    "number_of_h_donors": (NumberHDonors, PropertyPredictorParameters),
    "number_of_atoms": (NumberAtoms, PropertyPredictorParameters),
    "number_of_rings": (NumberRings, PropertyPredictorParameters),
    "number_of_rotatable_bonds": (NumberRotatableBonds, PropertyPredictorParameters),
    "number_of_large_rings": (NumberLargeRings, PropertyPredictorParameters),
    "number_of_heterocycles": (NumberHeterocycles, PropertyPredictorParameters),
    "number_of_stereocenters": (NumberStereocenters, PropertyPredictorParameters),
    "is_scaffold": (IsScaffold, PropertyPredictorParameters),
    # rule-based properties
    "bertz": (Bertz, PropertyPredictorParameters),
    "tpsa": (Tpsa, PropertyPredictorParameters),
    "logp": (Logp, PropertyPredictorParameters),
    "qed": (Qed, PropertyPredictorParameters),
    "plogp": (Plogp, PropertyPredictorParameters),
    "penalized_logp": (Plogp, PropertyPredictorParameters),
    "lipinski": (Lipinski, PropertyPredictorParameters),
    "sas": (Sas, PropertyPredictorParameters),
    "esol": (Esol, PropertyPredictorParameters),
    "similarity_seed": (SimilaritySeed, SimilaritySeedParameters),
    # properties predicted by models
    "scscore": (Scscore, ScscoreConfiguration),
    "activity_against_target": (ActivityAgainstTarget, ActivityAgainstTargetParameters),
}
