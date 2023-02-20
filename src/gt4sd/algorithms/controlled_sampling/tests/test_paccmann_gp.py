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
"""CLaSS tests."""

from typing import ClassVar, Type

import pytest

from gt4sd.algorithms.controlled_sampling.paccmann_gp import (
    PaccMannGP,
    PaccMannGPGenerator,
)
from gt4sd.algorithms.core import AlgorithmConfiguration
from gt4sd.algorithms.registry import ApplicationsRegistry

TARGET = {
    "qed": {"weight": 1.0},
    "molwt": {"target": 200},
    "sa": {"weight": 2.0},
    "callable": {"evaluator": lambda x: 1.0},
    "affinity": {"protein": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"},
}
PARAMETERS = {
    "number_of_steps": 8,
    "number_of_initial_points": 4,
    "number_of_optimization_rounds": 1,
    "samples_for_evaluation": 10,
    "maximum_number_of_sampling_steps": 4,
}


def get_classvar_type(class_var):
    """Extract type from ClassVar type annotation: `ClassVar[T]] -> T`."""
    return class_var.__args__[0]


@pytest.mark.parametrize(
    "config_class, algorithm_type, domain, algorithm_name",
    [
        (
            PaccMannGPGenerator,
            "controlled_sampling",
            "materials",
            PaccMannGP.__name__,
        ),
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
    [
        (PaccMannGPGenerator),
    ],
)
def test_config_instance(config_class: Type[AlgorithmConfiguration]):
    config = config_class()  # type:ignore
    assert config.algorithm_application == config_class.__name__


@pytest.mark.parametrize(
    "config_class",
    [
        (PaccMannGPGenerator),
    ],
)
def test_available_versions(config_class: Type[AlgorithmConfiguration]):
    versions = config_class.list_versions()
    assert "v0" in versions


@pytest.mark.parametrize(
    "config, algorithm, algorithm_parameters",
    [
        (PaccMannGPGenerator, PaccMannGP, PARAMETERS),
    ],
)
def test_generation_via_import(config, algorithm, algorithm_parameters):
    parameters = {
        "batch_size": 1,
    }
    for param, value in algorithm_parameters.items():
        parameters[param] = value
    config = config(**parameters)
    algorithm = algorithm(configuration=config, target=TARGET)
    items = list(algorithm.sample(1))
    assert len(items) == 1


@pytest.mark.parametrize(
    "algorithm_application, algorithm_type, domain, algorithm_name, algorithm_parameters",
    [
        (
            PaccMannGPGenerator.__name__,
            "controlled_sampling",
            "materials",
            PaccMannGP.__name__,
            PARAMETERS,
        ),
    ],
)
def test_generation_via_registry(
    algorithm_type,
    domain,
    algorithm_name,
    algorithm_application,
    algorithm_parameters,
):
    parameters = {
        "target": TARGET,
        "algorithm_type": algorithm_type,
        "domain": domain,
        "algorithm_name": algorithm_name,
        "algorithm_application": algorithm_application,
        "batch_size": 1,
    }
    for param, value in algorithm_parameters.items():
        parameters[param] = value
    algorithm = ApplicationsRegistry.get_application_instance(**parameters)
    items = list(algorithm.sample(1))
    assert len(items) == 1
