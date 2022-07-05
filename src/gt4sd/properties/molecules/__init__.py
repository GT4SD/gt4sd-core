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
from typing import Callable, Dict, Type, Union
from ..core import PropertyPredictor, CallablePropertyPredictor

from rdkit.Chem import Mol

from .core import (
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
    SimilaritySeed,
    Tpsa,
)
from .utils import get_similarity_fn  # type: ignore

# All functions can be called with either a SMILES or a Mol object.
MOLECULE_FACTORY: Dict[str, Union[CallablePropertyPredictor, PropertyPredictor]] = {
    # Inherent properties
    "molecular_weight": MolecularWeight,
    "number_of_aromatic_rings": NumberAromaticRings,
    "number_of_h_acceptors": NumberHAcceptors,
    "number_of_h_donors": NumberHDonors,
    "number_of_atoms": NumberAtoms,
    "number_of_rings": NumberRings,
    "number_of_rotatable_bonds": NumberRotatableBonds,
    "number_of_large_rings": NumberLargeRings,
    "number_of_heterocycles": NumberHeterocycles,
    "number_of_stereocenters": NumberStereocenters,
    "is_scaffold": IsScaffold,
    # Rule-based properties
    "bertz": Bertz,
    "tpsa": Tpsa,
    "logp": Logp,
    "qed": Qed,
    "plogp": Plogp,
    "penalized_logp": Plogp,
    "lipinski": Lipinski,
    "sas": Sas,
    "esol": Esol,
    "similarity_seed": SimilaritySeed,
    # Properties predicted by ML models
    "scscore": Scscore,
}
