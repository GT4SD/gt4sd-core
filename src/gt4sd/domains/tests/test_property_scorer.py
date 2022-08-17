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
"""Test for properties."""
import numpy as np
import pytest

from gt4sd.domains.materials.scorer import SCORING_FUNCTIONS
from gt4sd.properties import PROTEIN_PROPERTY_PREDICTOR_FACTORY

protein = "KFLIYQMECSTMIFGL"
molecule = "C1=CC(=CC(=C1)Br)CN"
seed = "CCO"
_target = "drd2"
ground_truths = {
    "length": 16,
    "protein_weight": 1924.36,
    "boman_index": -0.534375,
    "aliphaticity": 97.5,
    "hydrophobicity": 0.5625,
    "charge": -0.071,
    "charge_density": -3.6895383400195386e-05,
    "isoelectric_point": 6.125,
    "aromaticity": 0.1875,
    "instability": 36.8375,
    "plogp": 0.25130060815905964,
    "molecular_weight": 186.05199999999996,
    "lipinski": 1,
    "esol": -2.6649954522215555,
    "scscore": 4.391860681753299,
    "sas": 1.6564993918409403,
    "bertz": 197.86256853719752,
    "tpsa": 26.02,
    "logp": 1.9078,
    "qed": 0.7116306772337652,
    "number_of_h_acceptors": 1,
    "number_of_atoms": 17,
    "number_of_h_donors": 1,
    "number_of_aromatic_rings": 1,
    "number_of_rings": 1,
    "number_of_large_rings": 0,
    "number_of_rotatable_bonds": 1,
    "is_scaffold": 0,
    "number_of_heterocycles": 0,
    "number_of_stereocenters": 0,
}

molecule_further_ground_truths = {
    "activity_against_target": 0.0049787821116125345,
    "similarity_seed": 0.03333333333333333,
}
protein_further_ground_truths = {"charge": 1.123}

artifact_model_data = {
    "tox21": {
        "parameters": {"algorithm_version": "v0"},
        "ground_truth": [
            6.422001024475321e-05,
            0.00028556393226608634,
            0.0027144483756273985,
            0.03775344416499138,
            0.000604992441367358,
            0.00027705798856914043,
            0.10752066224813461,
            0.5733309388160706,
            0.001268531079404056,
            0.10181115567684174,
            0.8995946049690247,
            0.1677667647600174,
        ],
    },
    "organtox": {
        "parameters": {
            "algorithm_version": "v0",
            "site": "Heart",
            "toxicity_type": "all",
        },
        "ground_truth": [0.06142323836684227, 0.07934761792421341],
    },
}

distance = 0.0


def select_sample(property_key):
    """select a sample from the protein or molecule factory."""
    return protein if property_key in PROTEIN_PROPERTY_PREDICTOR_FACTORY else molecule


@pytest.mark.parametrize(
    "property_key", [(property_key) for property_key in ground_truths.keys()]
)
def test_property_scorer(property_key):
    scorer = SCORING_FUNCTIONS["property_scorer"](
        name=property_key, target=ground_truths[property_key]
    )
    sample = select_sample(property_key)
    assert np.isclose(scorer(sample), distance, atol=1e-2)  # type: ignore


def test_similarity_seed_scorer():
    scorer = SCORING_FUNCTIONS["property_scorer"](
        name="similarity_seed",
        target=molecule_further_ground_truths["similarity_seed"],
        parameters={"smiles": seed},
    )
    assert np.isclose(scorer(molecule), distance, atol=1e-2)  # type: ignore


def test_activity_against_target_scorer():
    """target for the scorer is a number (for example the predicted property value). Target for the property is a string (molecule)."""
    scorer = SCORING_FUNCTIONS["property_scorer"](
        name="activity_against_target",
        target=molecule_further_ground_truths["activity_against_target"],
        parameters={"target": _target},
    )
    assert np.isclose(scorer(molecule), distance, atol=1e-2)  # type: ignore


def test_charge_with_arguments_scorer():
    scorer = SCORING_FUNCTIONS["property_scorer"](
        name="charge",
        target=protein_further_ground_truths["charge"],
        parameters={"amide": True, "ph": 5.0},
    )
    assert np.isclose(scorer(protein), distance, atol=1e-2)  # type: ignore


@pytest.mark.parametrize(
    "property_key",
    [(property_key) for property_key in artifact_model_data.keys()],
)
def test_artifact_models_scorer(property_key):
    scorer = SCORING_FUNCTIONS["property_scorer"](
        name=property_key,
        target=artifact_model_data[property_key]["ground_truth"],
        parameters=artifact_model_data[property_key]["parameters"],
    )
    sample = select_sample(property_key)
    assert all(
        np.isclose(
            scorer(sample),
            distance,  # type: ignore
            atol=1e-2,
        )
    )  # type: ignore
