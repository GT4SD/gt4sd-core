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
from typing import Callable, Dict

from rdkit.Chem import Mol

from ...domains.materials import Property, SmallMolecule
from .core import (
    bertz,
    esol,
    is_scaffold,
    lipinski,
    logp,
    molecular_weight,
    number_of_aromatic_rings,
    number_of_atoms,
    number_of_h_acceptors,
    number_of_h_donors,
    number_of_heterocycles,
    number_of_large_rings,
    number_of_rings,
    number_of_rotatable_bonds,
    number_of_stereocenters,
    plogp,
    qed,
    sas,
    scscore,
    similarity_to_seed,
    tpsa,
)
from .utils import get_similarity_fn  # type: ignore

# All functions can be called with either a SMILES or a Mol object.
MOLECULE_FACTORY: Dict[str, Callable[[SmallMolecule], Property]] = {
    # Inherent properties
    "molecular_weight": molecular_weight,
    "number_of_aromatic_rings": number_of_aromatic_rings,
    "number_of_h_acceptors": number_of_h_acceptors,
    "number_of_h_donors": number_of_h_donors,
    "number_of_atoms": number_of_atoms,
    "number_of_rings": number_of_rings,
    "number_of_rotatable_bonds": number_of_rotatable_bonds,
    "number_of_large_rings": number_of_large_rings,
    "number_of_heterocycles": number_of_heterocycles,
    "number_of_stereocenters": number_of_stereocenters,
    "is_scaffold": is_scaffold,
    # Rule-based properties
    "bertz": bertz,
    "tpsa": tpsa,
    "logp": logp,
    "qed": qed,
    "plog": plogp,
    "penalized_logp": plogp,
    "lipinski": lipinski,
    "sas": sas,
    "esol": esol,
    "similarity": similarity_to_seed,
    # Properties predicted by ML models
    "scscore": scscore,
}
