"""Tests for registry that are independent of specific registrations."""

import pickle
from typing import ClassVar

import pytest
from pydantic import ValidationError

from gt4sd.algorithms.core import AlgorithmConfiguration, GeneratorAlgorithm
from gt4sd.algorithms.registry import ApplicationsRegistry
from gt4sd.exceptions import DuplicateApplicationRegistration

# there are at least 2 available versions, 1 per PaccMannRL configuration
AT_LEAST = 2


def assert_pickable(obj):
    pickled_obj = pickle.dumps(obj)
    restored_obj = pickle.loads(pickled_obj)

    assert restored_obj.algorithm_version == "test"
    assert restored_obj == obj

    return restored_obj


def test_list_available_s3():
    len(ApplicationsRegistry.list_available())
    assert len(ApplicationsRegistry.list_available()) >= AT_LEAST


def test_list_available_local_via_S3SyncError(mock_wrong_s3_env):
    assert len(ApplicationsRegistry.list_available()) >= AT_LEAST


def test_inherited_validation():
    Config = next(iter(ApplicationsRegistry.applications.values())).configuration_class
    with pytest.raises(
        ValidationError, match="algorithm_version\n +none is not an allowed value"
    ):
        Config(algorithm_version=None)  # type: ignore

    # NOTE: values convertible to string will not raise!
    Config(algorithm_version=5)  # type: ignore


def test_validation():
    with pytest.raises(
        ValidationError, match="batch_size\n +value is not a valid integer"
    ):
        ApplicationsRegistry.get_configuration_instance(
            algorithm_type="conditional_generation",
            domain="materials",
            algorithm_name="PaccMannRL",
            algorithm_application="PaccMannRLProteinBasedGenerator",
            batch_size="wrong_type",
        )


def test_pickable_wrapped_configurations():
    # https://github.com/samuelcolvin/pydantic/issues/2111
    Config = next(iter(ApplicationsRegistry.applications.values())).configuration_class
    restored_obj = assert_pickable(Config(algorithm_version="test"))

    # wrong type assignment, but we did not configure it to raise here:
    restored_obj.algorithm_version = object
    # ensure the restored dataclass is still a pydantic dataclass (mimic validation)
    _, optional_errors = restored_obj.__pydantic_model__.__fields__.get(
        "algorithm_version"
    ).validate(
        restored_obj.algorithm_version,
        restored_obj.__dict__,
        loc="algorithm_version",
        cls=restored_obj.__class__,
    )
    assert optional_errors is not None


def test_multiple_registration():
    class OtherAlgorithm(GeneratorAlgorithm):
        pass

    @ApplicationsRegistry.register_algorithm_application(
        GeneratorAlgorithm  # type:ignore
    )
    @ApplicationsRegistry.register_algorithm_application(OtherAlgorithm)  # type:ignore
    class Config(AlgorithmConfiguration):
        algorithm_type: ClassVar[str] = "dummy"
        domain: ClassVar[str] = ""
        algorithm_version: str = "development"

    # the class wrapping was applied twice
    config_class = ApplicationsRegistry.get_application(
        algorithm_type="dummy",
        domain="",
        algorithm_name="GeneratorAlgorithm",
        algorithm_application="Config",
    ).configuration_class
    assert config_class is Config
    assert config_class.algorithm_name == "GeneratorAlgorithm"
    assert config_class.algorithm_application == "Config"
    # __wrapped__?

    # retrieve singly wrapped config
    other_config_class = ApplicationsRegistry.get_application(
        algorithm_type="dummy",
        domain="",
        algorithm_name="OtherAlgorithm",
        algorithm_application="Config",
    ).configuration_class
    assert other_config_class is not Config
    assert other_config_class.algorithm_name == "OtherAlgorithm"
    assert other_config_class.algorithm_application == "Config"

    # registering Config directly and with explicit algorithm_application
    ExplicitConfig = ApplicationsRegistry.register_algorithm_application(
        GeneratorAlgorithm,  # type:ignore
        as_algorithm_application="ExplicitApplication",
    )(Config)
    explicit_config_class = ApplicationsRegistry.get_application(
        algorithm_type="dummy",
        domain="",
        algorithm_name="GeneratorAlgorithm",
        algorithm_application="ExplicitApplication",
    ).configuration_class
    assert explicit_config_class is ExplicitConfig
    assert explicit_config_class.algorithm_name == "GeneratorAlgorithm"
    assert explicit_config_class.algorithm_application == "ExplicitApplication"

    # overwriting value in applications is not allowed, applications are unique
    with pytest.raises(DuplicateApplicationRegistration):
        ApplicationsRegistry.register_algorithm_application(
            GeneratorAlgorithm  # type:ignore
        )(Config)
