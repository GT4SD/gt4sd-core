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
"""PaccMannRL tests."""

import pickle
from typing import ClassVar, Type

import numpy as np
import pytest

from gt4sd.algorithms.conditional_generation.paccmann_rl import (
    PaccMannRL,
    PaccMannRLOmicBasedGenerator,
    PaccMannRLProteinBasedGenerator,
)
from gt4sd.algorithms.core import AlgorithmConfiguration
from gt4sd.algorithms.registry import ApplicationsRegistry


def get_classvar_type(class_var):
    """Extract type from ClassVar type annotation: `ClassVar[T]] -> T`."""
    return class_var.__args__[0]


@pytest.mark.parametrize(
    "config_class, algorithm_type, domain, algorithm_name",
    [
        (
            PaccMannRLProteinBasedGenerator,
            "conditional_generation",
            "materials",
            PaccMannRL.__name__,
        ),
        (
            PaccMannRLOmicBasedGenerator,
            "conditional_generation",
            "materials",
            PaccMannRL.__name__,
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
        (PaccMannRLProteinBasedGenerator),
        (PaccMannRLOmicBasedGenerator),
    ],
)
def test_config_instance(config_class: Type[AlgorithmConfiguration]):
    config = config_class()  # type:ignore
    assert config.algorithm_application == config_class.__name__


@pytest.mark.parametrize(
    "config_class",
    [
        (PaccMannRLProteinBasedGenerator),
        (PaccMannRLOmicBasedGenerator),
    ],
)
def test_available_versions(config_class: Type[AlgorithmConfiguration]):
    versions = config_class.list_versions()
    assert "v0" in versions


@pytest.mark.parametrize(
    "config, example_target, algorithm",
    [
        (
            PaccMannRLProteinBasedGenerator,
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
            PaccMannRL,
        ),
        (
            PaccMannRLOmicBasedGenerator,
            np.random.rand(2128),
            PaccMannRL,
        ),
        (
            PaccMannRLOmicBasedGenerator,
            f"[{','.join(map(str, np.random.rand(2128)))}]",
            PaccMannRL,
        ),
    ],
)
def test_generation_via_import(config, example_target, algorithm):
    paccmann_rl = algorithm(configuration=config(), target=example_target)
    items = list(paccmann_rl.sample(5))
    assert len(items) == 5


@pytest.mark.parametrize(
    "algorithm_application, target",
    [
        (
            PaccMannRLProteinBasedGenerator.__name__,
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
        ),
    ],
)
def test_generation_via_registry(target, algorithm_application):
    paccmann_rl = ApplicationsRegistry.get_application_instance(
        target=target,
        algorithm_type="conditional_generation",
        domain="materials",
        algorithm_name=PaccMannRL.__name__,
        algorithm_application=algorithm_application,
        generated_length=5,
    )
    items = list(paccmann_rl.sample(5))
    assert len(items) == 5


@pytest.mark.parametrize(
    "config_class",
    [
        (PaccMannRLProteinBasedGenerator),
        (PaccMannRLOmicBasedGenerator),
    ],
)
def test_configuration_pickable(config_class: Type[AlgorithmConfiguration]):
    # implementation
    obj = config_class(algorithm_version="test")

    # ---
    import inspect

    inspect.getmodule(config_class)
    # ---
    pickled_obj = pickle.dumps(obj)
    restored_obj = pickle.loads(pickled_obj)
    assert restored_obj.algorithm_version == "test"
    assert restored_obj == obj

    # registered
    Config = ApplicationsRegistry.get_application(
        algorithm_type="conditional_generation",
        domain="materials",
        algorithm_name=PaccMannRL.__name__,
        algorithm_application=config_class.__name__,
    ).configuration_class

    obj = Config(algorithm_version="test")
    pickled_obj = pickle.dumps(obj)
    restored_obj = pickle.loads(pickled_obj)

    assert restored_obj.algorithm_version == "test"
    assert restored_obj == obj
