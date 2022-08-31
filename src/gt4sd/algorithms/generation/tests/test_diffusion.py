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
"""Diffusers tests."""

from typing import ClassVar, Type

import pytest

from gt4sd.algorithms.core import AlgorithmConfiguration
from gt4sd.algorithms.generation.diffusion import (
    DDIMGenerator,
    DDPMGenerator,
    DiffusersGenerationAlgorithm,
    LDMGenerator,
    LDMTextToImageGenerator,
    ScoreSdeGenerator,
    StableDiffusionGenerator,
)
from gt4sd.algorithms.registry import ApplicationsRegistry
from gt4sd.tests.utils import GT4SDTestSettings

test_settings = GT4SDTestSettings.get_instance()


def get_classvar_type(class_var):
    """Extract type from ClassVar type annotation: `ClassVar[T]] -> T`."""
    return class_var.__args__[0]


@pytest.mark.parametrize(
    "config_class, algorithm_type, domain, algorithm_name",
    [
        (
            DDPMGenerator,
            "generation",
            "generic",
            DiffusersGenerationAlgorithm.__name__,
        ),
        (
            DDIMGenerator,
            "generation",
            "generic",
            DiffusersGenerationAlgorithm.__name__,
        ),
        (
            ScoreSdeGenerator,
            "generation",
            "generic",
            DiffusersGenerationAlgorithm.__name__,
        ),
        (
            LDMGenerator,
            "generation",
            "generic",
            DiffusersGenerationAlgorithm.__name__,
        ),
        (
            LDMTextToImageGenerator,
            "generation",
            "generic",
            DiffusersGenerationAlgorithm.__name__,
        ),
        (
            StableDiffusionGenerator,
            "generation",
            "generic",
            DiffusersGenerationAlgorithm.__name__,
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
        (DDPMGenerator),
        (DDIMGenerator),
        (ScoreSdeGenerator),
        (LDMGenerator),
        (LDMTextToImageGenerator),
        (StableDiffusionGenerator),
    ],
)
def test_config_instance(config_class: Type[AlgorithmConfiguration]):
    config = config_class()  # type:ignore
    assert config.algorithm_application == config_class.__name__


@pytest.mark.parametrize(
    "config_class",
    [
        (DDPMGenerator),
        (DDIMGenerator),
        (ScoreSdeGenerator),
        (LDMGenerator),
        (LDMTextToImageGenerator),
        (StableDiffusionGenerator),
    ],
)
def test_available_versions(config_class: Type[AlgorithmConfiguration]):
    versions = config_class.list_versions()
    assert len(versions) > 0


@pytest.mark.parametrize(
    "config, algorithm",
    [
        pytest.param(
            DDPMGenerator,
            DiffusersGenerationAlgorithm,
        ),
        pytest.param(
            DDIMGenerator,
            DiffusersGenerationAlgorithm,
        ),
        pytest.param(
            ScoreSdeGenerator,
            DiffusersGenerationAlgorithm,
        ),
        pytest.param(
            LDMGenerator,
            DiffusersGenerationAlgorithm,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            LDMTextToImageGenerator,
            DiffusersGenerationAlgorithm,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            StableDiffusionGenerator,
            DiffusersGenerationAlgorithm,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
    ],
)
def test_generation_via_import(config, algorithm):
    algorithm = algorithm(configuration=config(length=10, number_of_sequences=1))
    items = list(algorithm.sample(1))
    assert len(items) == 1


@pytest.mark.parametrize(
    "algorithm_application, algorithm_type, domain, algorithm_name",
    [
        pytest.param(
            DDPMGenerator.__name__,
            "generation",
            "generic",
            DiffusersGenerationAlgorithm.__name__,
        ),
        pytest.param(
            DDIMGenerator.__name__,
            "generation",
            "generic",
            DiffusersGenerationAlgorithm.__name__,
        ),
        pytest.param(
            ScoreSdeGenerator.__name__,
            "generation",
            "generic",
            DiffusersGenerationAlgorithm.__name__,
        ),
        pytest.param(
            LDMGenerator.__name__,
            "generation",
            "generic",
            DiffusersGenerationAlgorithm.__name__,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            LDMTextToImageGenerator.__name__,
            "generation",
            "generic",
            DiffusersGenerationAlgorithm.__name__,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            StableDiffusionGenerator.__name__,
            "generation",
            "generic",
            DiffusersGenerationAlgorithm.__name__,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
    ],
)
def test_generation_via_registry(
    algorithm_type, domain, algorithm_name, algorithm_application
):
    algorithm = ApplicationsRegistry.get_application_instance(
        target=None,
        algorithm_type=algorithm_type,
        domain=domain,
        algorithm_name=algorithm_name,
        algorithm_application=algorithm_application,
        length=10,
        number_of_sequences=1,
    )
    items = list(algorithm.sample(1))
    assert len(items) == 1
