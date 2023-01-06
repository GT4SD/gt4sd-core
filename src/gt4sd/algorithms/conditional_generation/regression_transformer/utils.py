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
from typing import List, Tuple, cast

from rdkit import Chem


def get_substructure_indices(
    full_sequence: List[str], substructure: List[str]
) -> List[int]:
    """
    Args:
        full_sequence: A list of strings, each representing a token from the full sequence
        substructure: A list of strings, each representing a token from the substructure that
            is contained in the full sequence.

    Returns:
        A list of integers, corresponding to all the indices of the tokens in the full sequence
        that match the substructure.

    """
    substructure_indices: List = []
    for i in range(len(full_sequence)):
        if full_sequence[i] == substructure[0]:
            if full_sequence[i : i + len(substructure)] == substructure:
                substructure_indices.extend(range(i, i + len(substructure)))
    return substructure_indices


def filter_stubbed(
    property_sequences: Tuple[Tuple[str, str]], target: str, threshold: float = 0.5
) -> Tuple[Tuple[str, str]]:
    """
    Remove stub-like molecules that are substantially smaller than the target.

    Args:
        sequences: List of generated molecules.
        properties: Properties of the molecules. Only used to be returned after filtering.
        target: Seed molecule.
        threshold: Fraction of size of generated molecule compared to seed determining the
            threshold under which molecules are discarded. Defaults to 0.5.

    Returns:
        Tuple of tuples of length 2 with filtered, generated molecule and its properties.
    """

    seed = Chem.MolFromSmiles(target)

    seed_atoms = len(list(seed.GetAtoms()))
    seed_bonds = seed.GetNumBonds()

    smis: List[str] = []
    props: List[str] = []
    for smi, prop in property_sequences:
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            continue

        num_atoms = len(list(mol.GetAtoms()))
        num_bonds = mol.GetNumBonds()

        if num_atoms > (threshold * seed_atoms) and num_bonds > (
            threshold * seed_bonds
        ):
            smis.append(smi)
            props.append(prop)

    successes = cast(Tuple[Tuple[str, str]], tuple(zip(smis, props)))
    return successes
