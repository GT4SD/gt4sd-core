"""CLaSS tests."""

import pickle
from typing import ClassVar, Type

import pytest

from gt4sd.algorithms.core import AlgorithmConfiguration
from gt4sd.algorithms.registry import ApplicationsRegistry
from gt4sd.extras import EXTRAS_ENABLED

if not EXTRAS_ENABLED:
    pytest.skip("Extras from custom PyPI disabled", allow_module_level=True)
else:
    from gt4sd.algorithms.controlled_sampling.class_controlled_sampling import (
        PAG,
        CLaSS,
        CogMol,
    )
    from gt4sd.algorithms.controlled_sampling.class_controlled_sampling.implementation import (
        UnsupportedTargetError,
    )


def get_classvar_type(class_var):
    """Extract type from ClassVar type annotation: `ClassVar[T]] -> T`."""
    return class_var.__args__[0]


MPRO = "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ"


@pytest.mark.parametrize(
    "config_class, algorithm_type, domain, algorithm_name",
    [
        (
            CogMol,
            "controlled_sampling",
            "materials",
            CLaSS.__name__,
        ),
        (
            PAG,
            "controlled_sampling",
            "materials",
            CLaSS.__name__,
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
        (CogMol),
        (PAG),
    ],
)
def test_config_instance(config_class: Type[AlgorithmConfiguration]):
    config = config_class()  # type:ignore
    assert config.algorithm_application == config_class.__name__


@pytest.mark.parametrize(
    "config_class",
    [
        (CogMol),
        (PAG),
    ],
)
def test_available_versions(config_class: Type[AlgorithmConfiguration]):
    versions = config_class.list_versions()
    assert "v0" in versions


@pytest.mark.parametrize(
    "config, example_target, algorithm, kwargs",
    [
        (
            CogMol,
            MPRO,
            CLaSS,
            {
                "samples_per_round": 173,
                "max_length": 40,
                "temperature": 0.8,
                "num_proteins_selectivity": 20,
            },
        ),
        (
            PAG,
            None,
            CLaSS,
            {
                "samples_per_round": 173,
                "max_length": 40,
                "temperature": 0.8,
            },
        ),
    ],
)
def test_generation_via_import(config, example_target, algorithm, kwargs):
    class_sampling = algorithm(
        configuration=config(**kwargs),
        target=example_target,
    )
    items = list(class_sampling.sample(5))
    assert len(items) == 5


@pytest.mark.parametrize(
    "algorithm_application, target",
    [
        (
            CogMol.__name__,
            MPRO,
        ),
        (
            PAG.__name__,
            None,
        ),
    ],
)
def test_generation_via_registry(target, algorithm_application):
    class_sampling = ApplicationsRegistry.get_application_instance(
        target=target,
        algorithm_type="controlled_sampling",
        domain="materials",
        algorithm_name=CLaSS.__name__,
        algorithm_application=algorithm_application,
    )
    items = list(class_sampling.sample(5))
    assert len(items) == 5


def test_unsupported_target(algorithm_application=CogMol.__name__, target=MPRO):
    invalid_target = target[:30]  # assuming this makes it invalid

    # on construction
    with pytest.raises(UnsupportedTargetError):
        ApplicationsRegistry.get_application_instance(
            target=invalid_target,
            algorithm_type="controlled_sampling",
            domain="materials",
            algorithm_name=CLaSS.__name__,
            algorithm_application=algorithm_application,
        )

    # on sampling with changed targed
    config = CogMol()
    implementation = config.get_class_instance(  # type: ignore
        resources_path=config.ensure_artifacts(), target=target
    )
    with pytest.raises(UnsupportedTargetError):
        implementation.sample_accepted(invalid_target)


@pytest.mark.parametrize("config_class", [(CogMol), (PAG)])
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
        algorithm_type="controlled_sampling",
        domain="materials",
        algorithm_name=CLaSS.__name__,
        algorithm_application=config_class.__name__,
    ).configuration_class

    obj = Config(algorithm_version="test")
    pickled_obj = pickle.dumps(obj)
    restored_obj = pickle.loads(pickled_obj)

    assert restored_obj.algorithm_version == "test"
    assert restored_obj == obj
