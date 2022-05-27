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
"""RegressionTransformer tests."""

import pickle
from typing import ClassVar, Type

import pytest

from gt4sd.algorithms.conditional_generation.regression_transformer import (
    RegressionTransformer,
    RegressionTransformerMolecules,
    RegressionTransformerProteins,
)
from gt4sd.algorithms.core import AlgorithmConfiguration
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
            RegressionTransformerMolecules,
            "conditional_generation",
            "materials",
            RegressionTransformer.__name__,
        ),
        (
            RegressionTransformerProteins,
            "conditional_generation",
            "materials",
            RegressionTransformer.__name__,
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
        (RegressionTransformerMolecules),
        (RegressionTransformerProteins),
    ],
)
def test_config_instance(config_class: Type[AlgorithmConfiguration]):
    config = config_class()  # type:ignore
    assert config.algorithm_application == config_class.__name__


@pytest.mark.parametrize(
    "config_class",
    [
        (RegressionTransformerMolecules),
        (RegressionTransformerProteins),
    ],
)
def test_available_versions(config_class: Type[AlgorithmConfiguration]):
    versions = config_class.list_versions()
    assert "solubility" in versions or "stability" in versions


@pytest.mark.parametrize(
    "config, example_target, algorithm, params",
    [
        pytest.param(
            RegressionTransformerMolecules,
            "<esol>[MASK][MASK][MASK][MASK][MASK]|[Cl][C][Branch1_2][Branch1_2][=C][Branch1_1][C][Cl][Cl][Cl]",
            RegressionTransformer,
            {"algorithm_version": "solubility", "search": "greedy", "num_samples": 1},
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            RegressionTransformerMolecules,
            "<esol>-3.49|[C][C][MASK][MASK][MASK][C][Br]",
            RegressionTransformer,
            {"search": "sample", "temperature": 2.0, "num_samples": 5},
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            RegressionTransformerMolecules,
            "<logp>[MASK][MASK][MASK][MASK][MASK]|<scs>[MASK][MASK][MASK][MASK][MASK]|[C][C][O][C][=N][C][=N][C][Branch1_2][Branch1_1][=C][Ring1][Branch1_2][C][N][C][C][O][C][Branch1_1][C][C][C]",
            RegressionTransformer,
            {
                "search": "greedy",
                "num_samples": 1,
                "algorithm_version": "logp_and_synthesizability",
            },
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            RegressionTransformerMolecules,
            "<logp>6.534|<scs>3.835|[C][MASK][MASK][C][=N][C][=N][C][Branch1_2][Branch1_1][=C][Ring1][Branch1_2][C][N][C][C][O][C][Branch1_1][C][C][C]",
            RegressionTransformer,
            {
                "search": "sample",
                "num_samples": 3,
                "temperature": 1.4,
                "algorithm_version": "logp_and_synthesizability",
            },
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            RegressionTransformerProteins,
            "<stab>[MASK][MASK][MASK][MASK][MASK]|GSQEVNSNASPEEAEIARKAGATTWTEKGNKWEIRI",
            RegressionTransformer,
            {"search": "greedy", "num_samples": 1},
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            RegressionTransformerProteins,
            "<stab>1.123|TTIKNG[MASK][MASK][MASK]YTVPLSPEQAAK[MASK][MASK][MASK]KKRWPDYEVQIHGNTVKVT",
            RegressionTransformer,
            {"search": "sample", "temperature": 2.0, "num_samples": 5},
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        # Test the sampling wrapper configurations
        pytest.param(
            RegressionTransformerProteins,
            "TTIKNGABCYTVPLSPEQAAKABCKKRWPDYEVQIHGNTVKVT",
            RegressionTransformer,
            {
                "search": "sample",
                "temperature": 2.0,
                "num_samples": 5,
                "tolerance": 20,
                "sampling_wrapper": {
                    "property_goal": {"<stab>": 1.123},
                    "fraction_to_mask": 0.9,
                    "tokens_to_mask": ["A"],
                },
            },
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            RegressionTransformerMolecules,
            "CCOC1=NC=NC(=C1C)NCCOC(C)C",
            RegressionTransformer,
            {
                "search": "sample",
                "num_samples": 3,
                "temperature": 1.4,
                "algorithm_version": "logp_and_synthesizability",
                "sampling_wrapper": {
                    "property_goal": {"<logp>": 6.534, "<scs>": 3.835},
                    "fraction_to_mask": 0.2,
                },
            },
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
    ],
)
def test_generation_via_import(config, example_target, algorithm, params):
    num_samples = params.pop("num_samples", 1)
    regression_transformer = algorithm(
        configuration=config(**params), target=example_target
    )
    items = list(regression_transformer.sample(num_samples))
    assert len(items) == num_samples


@pytest.mark.parametrize(
    "algorithm_application, target, params",
    [
        pytest.param(
            RegressionTransformerMolecules.__name__,
            "<esol>[MASK][MASK][MASK][MASK][MASK]|[Cl][C][Branch1_2][Branch1_2][=C][Branch1_1][C][Cl][Cl][Cl]",
            {"search": "greedy", "num_samples": 1},
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            RegressionTransformerMolecules.__name__,
            "<esol>-3.49|[C][C][MASK][MASK][MASK][C][Br]",
            {"search": "sample", "temperature": 2.0, "num_samples": 5},
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            RegressionTransformerProteins.__name__,
            "<stab>[MASK][MASK][MASK][MASK][MASK]|GSQEVNSNASPEEAEIARKAGATTWTEKGNKWEIRI",
            {"search": "greedy", "num_samples": 1},
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
        pytest.param(
            RegressionTransformerProteins.__name__,
            "<stab>1.123|TTIKNG[MASK][MASK][MASK]YTVPLSPEQAAK[MASK][MASK][MASK]KKRWPDYEVQIHGNTVKVT",
            {"search": "sample", "temperature": 2.0, "num_samples": 5},
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="high_memory"),
        ),
    ],
)
def test_generation_via_registry(target, algorithm_application, params):
    num_samples = params.pop("num_samples", 1)
    regression_transformer = ApplicationsRegistry.get_application_instance(
        target=target,
        algorithm_type="conditional_generation",
        domain="materials",
        algorithm_name=RegressionTransformer.__name__,
        algorithm_application=algorithm_application,
        **params,
    )
    items = list(regression_transformer.sample(num_samples))
    assert len(items) == num_samples


@pytest.mark.parametrize(
    "config_class",
    [
        (RegressionTransformerMolecules),
        (RegressionTransformerProteins),
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
        algorithm_name=RegressionTransformer.__name__,
        algorithm_application=config_class.__name__,
    ).configuration_class

    obj = Config(algorithm_version="test")
    pickled_obj = pickle.dumps(obj)
    restored_obj = pickle.loads(pickled_obj)

    assert restored_obj.algorithm_version == "test"
    assert restored_obj == obj
