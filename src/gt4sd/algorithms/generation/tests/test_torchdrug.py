"""TorchDrug tests."""

from typing import ClassVar, Type

import pytest

from gt4sd.algorithms.core import AlgorithmConfiguration
from gt4sd.algorithms.generation.torchdrug import (
    TorchDrugGenerator,
    TorchDrugPlogpGAF,
    TorchDrugPlogpGCPN,
    TorchDrugQedGAF,
    TorchDrugQedGCPN,
    TorchDrugZincGAF,
    TorchDrugZincGCPN,
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
            TorchDrugZincGCPN,
            "generation",
            "materials",
            TorchDrugGenerator.__name__,
        ),
        (
            TorchDrugPlogpGCPN,
            "generation",
            "materials",
            TorchDrugGenerator.__name__,
        ),
        (
            TorchDrugQedGCPN,
            "generation",
            "materials",
            TorchDrugGenerator.__name__,
        ),
        (
            TorchDrugZincGAF,
            "generation",
            "materials",
            TorchDrugGenerator.__name__,
        ),
        (
            TorchDrugPlogpGAF,
            "generation",
            "materials",
            TorchDrugGenerator.__name__,
        ),
        (
            TorchDrugQedGAF,
            "generation",
            "materials",
            TorchDrugGenerator.__name__,
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
        (TorchDrugZincGCPN),
        (TorchDrugPlogpGCPN),
        (TorchDrugQedGCPN),
        (TorchDrugZincGAF),
        (TorchDrugPlogpGAF),
        (TorchDrugQedGAF),
    ],
)
def test_config_instance(config_class: Type[AlgorithmConfiguration]):
    config = config_class()  # type:ignore
    assert config.algorithm_application == config_class.__name__


@pytest.mark.parametrize(
    "config_class",
    [
        (TorchDrugZincGCPN),
        (TorchDrugPlogpGCPN),
        (TorchDrugQedGCPN),
        (TorchDrugZincGAF),
        (TorchDrugPlogpGAF),
        (TorchDrugQedGAF),
    ],
)
def test_available_versions(config_class: Type[AlgorithmConfiguration]):
    versions = config_class.list_versions()
    assert len(versions) > 0


@pytest.mark.parametrize(
    "config, algorithm",
    [
        pytest.param(
            TorchDrugZincGCPN,
            TorchDrugGenerator,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            TorchDrugPlogpGCPN,
            TorchDrugGenerator,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            TorchDrugQedGCPN,
            TorchDrugGenerator,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            TorchDrugZincGAF,
            TorchDrugGenerator,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            TorchDrugPlogpGAF,
            TorchDrugGenerator,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            TorchDrugQedGAF,
            TorchDrugGenerator,
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
            TorchDrugZincGCPN.__name__,
            "generation",
            "materials",
            TorchDrugGenerator.__name__,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            TorchDrugPlogpGCPN.__name__,
            "generation",
            "materials",
            TorchDrugGenerator.__name__,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            TorchDrugQedGCPN.__name__,
            "generation",
            "materials",
            TorchDrugGenerator.__name__,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            TorchDrugZincGAF.__name__,
            "generation",
            "materials",
            TorchDrugGenerator.__name__,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            TorchDrugPlogpGAF.__name__,
            "generation",
            "materials",
            TorchDrugGenerator.__name__,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            TorchDrugQedGAF.__name__,
            "generation",
            "materials",
            TorchDrugGenerator.__name__,
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
