"""MolGX tests."""

from typing import ClassVar, Type

import pytest

from gt4sd.algorithms.core import AlgorithmConfiguration
from gt4sd.algorithms.registry import ApplicationsRegistry
from gt4sd.extras import EXTRAS_ENABLED
from gt4sd.tests.utils import GT4SDTestSettings

if not EXTRAS_ENABLED:
    pytest.skip("Extras from custom PyPI disabled", allow_module_level=True)
else:
    from gt4sd.algorithms.generation.molgx import MolGX, MolGXQM9Generator

test_settings = GT4SDTestSettings.get_instance()


def get_classvar_type(class_var):
    """Extract type from ClassVar type annotation: `ClassVar[T]] -> T`."""
    return class_var.__args__[0]


@pytest.mark.parametrize(
    "config_class, algorithm_type, domain, algorithm_name",
    [(MolGXQM9Generator, "generation", "materials", MolGX.__name__)],
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
    [(MolGXQM9Generator)],
)
def test_config_instance(config_class: Type[AlgorithmConfiguration]):
    config = config_class()  # type:ignore
    assert config.algorithm_application == config_class.__name__


@pytest.mark.parametrize(
    "config_class",
    [(MolGXQM9Generator)],
)
def test_available_versions(config_class: Type[AlgorithmConfiguration]):
    versions = config_class.list_versions()
    assert "v0" in versions


@pytest.mark.parametrize(
    "config, algorithm",
    [
        pytest.param(
            MolGXQM9Generator,
            MolGX,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="slow_runtime"),
        )
    ],
)
def test_generation_via_import(config, algorithm):
    algorithm = algorithm(configuration=config())
    items = list(algorithm.sample(3))
    assert len(items) == 3


@pytest.mark.parametrize(
    "algorithm_application, algorithm_type, domain, algorithm_name",
    [
        pytest.param(
            MolGXQM9Generator.__name__,
            "generation",
            "materials",
            MolGX.__name__,
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="slow_runtime"),
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
    )
    items = list(algorithm.sample(3))
    assert len(items) == 3
