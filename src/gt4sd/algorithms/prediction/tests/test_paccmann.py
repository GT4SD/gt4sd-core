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
"""PaccMann tests."""

from typing import ClassVar, Type

import numpy as np
import pytest

from gt4sd.algorithms.core import AlgorithmConfiguration
from gt4sd.algorithms.prediction.paccmann.core import AffinityPredictor, PaccMann
from gt4sd.algorithms.registry import ApplicationsRegistry


def get_classvar_type(class_var):
    """Extract type from ClassVar type annotation: `ClassVar[T]] -> T`."""
    return class_var.__args__[0]


@pytest.mark.parametrize(
    "config_class, algorithm_type, domain, algorithm_name",
    [
        (
            AffinityPredictor,
            "prediction",
            "materials",
            PaccMann.__name__,
        )
    ],
)
def test_config_class(
    config_class: Type[AlgorithmConfiguration],
    algorithm_type: str,
    domain: str,
    algorithm_name: str,
):
    assert config_class.algorithm_type == algorithm_type
    assert config_class.domain == domain
    assert config_class.algorithm_name == algorithm_name

    for keyword, type_annotation in config_class.__annotations__.items():
        if keyword in ("algorithm_type", "domain", "algorithm_name"):
            assert type_annotation.__origin__ is ClassVar  # type: ignore
            assert str == get_classvar_type(type_annotation)


@pytest.mark.parametrize(
    "config_class",
    [(AffinityPredictor)],
)
def test_config_instance(config_class: Type[AlgorithmConfiguration]):
    config = config_class()  # type:ignore
    assert config.algorithm_application == config_class.__name__


@pytest.mark.parametrize(
    "config, protein_targets, ligands, confidence, algorithm",
    [
        (
            AffinityPredictor,
            [
                "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
                "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
            ],
            [
                "CONN=COc1cc2ccccc2c11Occncc(Cl)c1N(O)O",
                "ClCCC(O1)C(C(N=C1C(=O)Nc1cccc(F)c1F)SO)C",
            ],
            True,
            PaccMann,
        )
    ],
)
def test_generation_via_import(config, protein_targets, ligands, confidence, algorithm):
    configuration = config(
        protein_targets=protein_targets, ligands=ligands, confidence=confidence
    )
    algorithm_instance = algorithm(configuration=configuration)

    predictor = configuration.get_conditional_generator(
        configuration.ensure_artifacts()
    )
    predictions, _ = predictor.predict()
    assert np.isclose(predictions[0, 0].item(), 0.5492, atol=1e-4)
    assert np.isclose(predictions[1, 0].item(), 0.4799, atol=1e-4)

    predictions = list(algorithm_instance.sample(2))
    assert np.isclose(predictions[0], 0.5492, atol=1e-4)
    assert np.isclose(predictions[1], 0.4799, atol=1e-4)


@pytest.mark.parametrize(
    "algorithm_application, protein_targets, ligands, confidence, algorithm_type, domain, algorithm_name",
    [
        (
            AffinityPredictor.__name__,
            [
                "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
                "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
            ],
            [
                "CONN=COc1cc2ccccc2c11Occncc(Cl)c1N(O)O",
                "ClCCC(O1)C(C(N=C1C(=O)Nc1cccc(F)c1F)SO)C",
            ],
            True,
            "prediction",
            "materials",
            PaccMann.__name__,
        ),
    ],
)
def test_generation_via_registry(
    algorithm_type,
    protein_targets,
    ligands,
    confidence,
    domain,
    algorithm_name,
    algorithm_application,
):
    algorithm = ApplicationsRegistry.get_application_instance(
        algorithm_type=algorithm_type,
        domain=domain,
        algorithm_name=algorithm_name,
        algorithm_application=algorithm_application,
        protein_targets=protein_targets,
        ligands=ligands,
        confidence=confidence,
    )
    items = list(algorithm.sample(1))
    assert len(items) == 1
