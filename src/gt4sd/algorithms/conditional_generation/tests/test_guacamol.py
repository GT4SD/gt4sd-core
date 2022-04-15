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
"""GuacaMol Baselines tests."""

from typing import ClassVar, Type

import pytest

from gt4sd.algorithms.conditional_generation.guacamol import (
    GraphGAGenerator,
    GraphMCTSGenerator,
    GuacaMolGenerator,
    SMILESGAGenerator,
    SMILESLSTMHCGenerator,
    SMILESLSTMPPOGenerator,
)
from gt4sd.algorithms.core import AlgorithmConfiguration
from gt4sd.algorithms.registry import ApplicationsRegistry
from gt4sd.tests.utils import GT4SDTestSettings

TARGET = {"isomer_scorer": {"target": 5.0, "target_smile": "NCCCCC"}}
algorithm_parameters = {
    "smiles_ga": {"random_start": True},
    "graph_ga": {"random_start": True},
    "graph_mcts": {
        "init_smiles": "CC",
        "population_size": 5,
        "generations": 5,
        "num_sims": 10,
        "max_children": 5,
        "max_atoms": 10,
    },
    "smiles_lstm_hc": {
        "random_start": True,
        "mols_to_sample": 10,
        "keep_top": 5,
        "max_len": 2,
        "optimize_batch_size": 3,
        "n_epochs": 2,
    },
    "smiles_lstm_ppo": {"num_epochs": 2, "episode_size": 10, "optimize_batch_size": 2},
}

test_settings = GT4SDTestSettings.get_instance()


def get_classvar_type(class_var):
    """Extract type from ClassVar type annotation: `ClassVar[T]] -> T`."""
    return class_var.__args__[0]


@pytest.mark.parametrize(
    "config_class, algorithm_type, domain, algorithm_name",
    [
        (
            SMILESGAGenerator,
            "conditional_generation",
            "materials",
            GuacaMolGenerator.__name__,
        ),
        (
            GraphGAGenerator,
            "conditional_generation",
            "materials",
            GuacaMolGenerator.__name__,
        ),
        (
            GraphMCTSGenerator,
            "conditional_generation",
            "materials",
            GuacaMolGenerator.__name__,
        ),
        (
            SMILESLSTMHCGenerator,
            "conditional_generation",
            "materials",
            GuacaMolGenerator.__name__,
        ),
        (
            SMILESLSTMPPOGenerator,
            "conditional_generation",
            "materials",
            GuacaMolGenerator.__name__,
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
        (SMILESGAGenerator),
        (GraphGAGenerator),
        (GraphMCTSGenerator),
        (SMILESLSTMHCGenerator),
        (SMILESLSTMPPOGenerator),
    ],
)
def test_config_instance(config_class: Type[AlgorithmConfiguration]):
    config = config_class()  # type:ignore
    assert config.algorithm_application == config_class.__name__


@pytest.mark.parametrize(
    "config_class",
    [
        (SMILESGAGenerator),
        (GraphGAGenerator),
        (GraphMCTSGenerator),
        (SMILESLSTMHCGenerator),
        (SMILESLSTMPPOGenerator),
    ],
)
def test_available_versions(config_class: Type[AlgorithmConfiguration]):
    versions = config_class.list_versions()
    assert "v0" in versions


@pytest.mark.parametrize(
    "config, algorithm, algorithm_parameters",
    [
        pytest.param(
            SMILESGAGenerator,
            GuacaMolGenerator,
            algorithm_parameters["smiles_ga"],
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="slow_runtime"),
        ),
        (GraphGAGenerator, GuacaMolGenerator, algorithm_parameters["graph_ga"]),
        (GraphMCTSGenerator, GuacaMolGenerator, algorithm_parameters["graph_mcts"]),
        pytest.param(
            SMILESLSTMHCGenerator,
            GuacaMolGenerator,
            algorithm_parameters["smiles_lstm_hc"],
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="slow_runtime"),
        ),
        pytest.param(
            SMILESLSTMPPOGenerator,
            GuacaMolGenerator,
            algorithm_parameters["smiles_lstm_ppo"],
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="slow_runtime"),
        ),
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
        pytest.param(
            SMILESGAGenerator.__name__,
            "conditional_generation",
            "materials",
            GuacaMolGenerator.__name__,
            algorithm_parameters["smiles_ga"],
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="slow_runtime"),
        ),
        (
            GraphGAGenerator.__name__,
            "conditional_generation",
            "materials",
            GuacaMolGenerator.__name__,
            algorithm_parameters["graph_ga"],
        ),
        (
            GraphMCTSGenerator.__name__,
            "conditional_generation",
            "materials",
            GuacaMolGenerator.__name__,
            algorithm_parameters["graph_mcts"],
        ),
        pytest.param(
            SMILESLSTMHCGenerator.__name__,
            "conditional_generation",
            "materials",
            GuacaMolGenerator.__name__,
            algorithm_parameters["smiles_lstm_hc"],
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="slow_runtime"),
        ),
        pytest.param(
            SMILESLSTMPPOGenerator.__name__,
            "conditional_generation",
            "materials",
            GuacaMolGenerator.__name__,
            algorithm_parameters["smiles_lstm_ppo"],
            marks=pytest.mark.skipif(test_settings.gt4sd_ci, reason="slow_runtime"),
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
