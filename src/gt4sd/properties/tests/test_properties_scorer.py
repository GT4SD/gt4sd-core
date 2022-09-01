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

from gt4sd.properties import (
    MOLECULE_PROPERTY_PREDICTOR_FACTORY,
    PROTEIN_PROPERTY_PREDICTOR_FACTORY,
    SCORING_FACTORY_WITH_PROPERTY_PREDICTORS,
    PropertyPredictorRegistry,
)

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


OPTIMAL_SCORE = 1.0


def select_sample(property_key):
    """select a molecule or protein accordingly to the property."""
    return protein if property_key in PROTEIN_PROPERTY_PREDICTOR_FACTORY else molecule


def select_opposite_sample(property_key):
    """select a molecule or protein not accordingly to the property."""
    return (
        protein if property_key not in PROTEIN_PROPERTY_PREDICTOR_FACTORY else molecule
    )


@pytest.mark.parametrize(
    "property_key", [(property_key) for property_key in ground_truths.keys()]
)
def test_property_scorer(property_key):
    scoring_function = PropertyPredictorRegistry.get_property_predictor(
        name=property_key
    )
    scorer = SCORING_FACTORY_WITH_PROPERTY_PREDICTORS["property_predictor_scorer"](
        name=property_key,
        scoring_function=scoring_function,
        target=ground_truths[property_key],
    )
    sample = select_sample(property_key)
    assert np.isclose(scorer.score(sample), OPTIMAL_SCORE, atol=1e-2)  # type: ignore


def test_similarity_seed_scorer():
    scoring_function = PropertyPredictorRegistry.get_property_predictor(
        name="similarity_seed", parameters={"smiles": seed}
    )
    scorer = SCORING_FACTORY_WITH_PROPERTY_PREDICTORS["property_predictor_scorer"](
        name="similarity_seed",
        scoring_function=scoring_function,
        target=molecule_further_ground_truths["similarity_seed"],
    )
    assert np.isclose(scorer.score(molecule), OPTIMAL_SCORE, atol=1e-2)  # type: ignore


def test_activity_against_target_scorer():
    """target for the scorer is a number (for example the predicted property value). Target for the property is a string (molecule)."""
    scoring_function = PropertyPredictorRegistry.get_property_predictor(
        name="activity_against_target", parameters={"target": _target}
    )
    scorer = SCORING_FACTORY_WITH_PROPERTY_PREDICTORS["property_predictor_scorer"](
        name="activity_against_target",
        scoring_function=scoring_function,
        target=molecule_further_ground_truths["activity_against_target"],
    )
    assert np.isclose(scorer.score(molecule), OPTIMAL_SCORE, atol=1e-2)  # type: ignore


def test_charge_with_arguments_scorer():
    scoring_function = PropertyPredictorRegistry.get_property_predictor(
        name="charge", parameters={"amide": True, "ph": 5.0}
    )
    scorer = SCORING_FACTORY_WITH_PROPERTY_PREDICTORS["property_predictor_scorer"](
        name="charge",
        scoring_function=scoring_function,
        target=protein_further_ground_truths["charge"],
    )
    assert np.isclose(scorer.score(protein), OPTIMAL_SCORE, atol=1e-2)  # type: ignore


@pytest.mark.parametrize(
    "property_key", [(property_key) for property_key in ground_truths.keys()]
)
def test_validation_property_scorer(property_key):
    scoring_function = PropertyPredictorRegistry.get_property_predictor(
        name=property_key
    )
    scorer = SCORING_FACTORY_WITH_PROPERTY_PREDICTORS["property_predictor_scorer"](
        name=property_key,
        scoring_function=scoring_function,
        target=ground_truths[property_key],
    )
    sample = select_opposite_sample(property_key)
    try:
        scorer.score(sample)
        assert False
    except ValueError:
        assert True


@pytest.mark.parametrize(
    "property_key", [(property_key) for property_key in ground_truths.keys()]
)
def test_molecule_property_scorer(property_key):
    scoring_function = PropertyPredictorRegistry.get_property_predictor(
        name=property_key
    )
    if property_key not in MOLECULE_PROPERTY_PREDICTOR_FACTORY:
        try:
            scorer = SCORING_FACTORY_WITH_PROPERTY_PREDICTORS[
                "molecule_property_predictor_scorer"
            ](
                name=property_key,
                scoring_function=scoring_function,
                target=ground_truths[property_key],
            )
            assert False
        except ValueError:
            assert True

    if property_key in MOLECULE_PROPERTY_PREDICTOR_FACTORY:
        scorer = SCORING_FACTORY_WITH_PROPERTY_PREDICTORS[
            "molecule_property_predictor_scorer"
        ](
            name=property_key,
            scoring_function=scoring_function,
            target=ground_truths[property_key],
        )
        sample = select_sample(property_key)
        assert np.isclose(scorer.score(sample), OPTIMAL_SCORE, atol=1e-2)  # type: ignore


@pytest.mark.parametrize(
    "property_key", [(property_key) for property_key in ground_truths.keys()]
)
def test_protein_property_scorer(property_key):
    scoring_function = PropertyPredictorRegistry.get_property_predictor(
        name=property_key
    )
    if property_key not in PROTEIN_PROPERTY_PREDICTOR_FACTORY:
        try:
            scorer = SCORING_FACTORY_WITH_PROPERTY_PREDICTORS[
                "protein_property_predictor_scorer"
            ](
                name=property_key,
                scoring_function=scoring_function,
                target=ground_truths[property_key],
            )
            assert False
        except ValueError:
            assert True

    if property_key in PROTEIN_PROPERTY_PREDICTOR_FACTORY:
        scorer = SCORING_FACTORY_WITH_PROPERTY_PREDICTORS[
            "protein_property_predictor_scorer"
        ](
            name=property_key,
            scoring_function=scoring_function,
            target=ground_truths[property_key],
        )
        sample = select_sample(property_key)
        assert np.isclose(scorer.score(sample), OPTIMAL_SCORE, atol=1e-2)  # type: ignore


def test_property_predictor_scorer_registry():
    scorer = PropertyPredictorRegistry.get_property_predictor_scorer(
        property_name="similarity_seed",
        scorer_name="property_predictor_scorer",
        target=molecule_further_ground_truths["similarity_seed"],
        parameters={"smiles": seed},
    )
    sample = select_sample("similarity_seed")
    assert isinstance(
        scorer, SCORING_FACTORY_WITH_PROPERTY_PREDICTORS["property_predictor_scorer"]
    )
    assert np.isclose(scorer.score(sample), OPTIMAL_SCORE, atol=1e-2)  # type: ignore

    scorer = PropertyPredictorRegistry.get_property_predictor_scorer(
        property_name="charge",
        scorer_name="property_predictor_scorer",
        target=protein_further_ground_truths["charge"],
        parameters={"amide": "True", "ph": 5.0},
    )
    sample = select_sample("charge")
    assert isinstance(
        scorer, SCORING_FACTORY_WITH_PROPERTY_PREDICTORS["property_predictor_scorer"]
    )
    assert np.isclose(scorer.score(sample), OPTIMAL_SCORE, atol=1e-2)  # type: ignore
    assert len(PropertyPredictorRegistry.list_available_scorers()) == 3
