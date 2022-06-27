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
"""Regression Transformer trainer unit tests."""

import shutil
import tempfile
from typing import Any, Dict, Iterable, cast

import pkg_resources

from gt4sd.algorithms.conditional_generation.regression_transformer import (
    RegressionTransformerMolecules,
)
from gt4sd.cli.argument_parser import DataClassType
from gt4sd.cli.trainer import TrainerArgumentParser
from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_MAPPING,
    RegressionTransformerTrainingArguments,
    RegressionTransformerTrainingPipeline,
    TRAINING_PIPELINE_ARGUMENTS_MAPPING,
)
from gt4sd.training_pipelines.core import TrainingPipelineArguments

template_config = {
    "model_args": {},
    "dataset_args": {"test_fraction": 0.5},
    "training_args": {
        "training_name": "regression-transformer-test",
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.0005,
        "do_train": True,
        "eval_accumulation_steps": 1,
        'eval_steps': 2,
        'save_steps': 2,
        'overwrite_output_dir': True,
    },
}


def combine_defaults_and_user_args(
    config: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:

    arguments = TRAINING_PIPELINE_ARGUMENTS_MAPPING['regression-transformer-trainer']
    parser = TrainerArgumentParser(
        cast(
            Iterable[DataClassType],
            tuple([*arguments]),
        )
    )
    args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    input_config = {
        arg.__name__: arg.__dict__
        for arg in args
        if isinstance(arg, TrainingPipelineArguments) and isinstance(arg.__name__, str)
    }
    input_config['model_args'].update(**config['model_args'])
    input_config['dataset_args'].update(**config['dataset_args'])
    input_config['training_args'].update(**config['training_args'])

    return input_config


def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("regression-transformer-trainer")
    assert pipeline is not None

    mol_model = RegressionTransformerMolecules(algorithm_version='qed')
    mol_path = mol_model.ensure_artifacts_for_version('qed')

    TEMPORARY_DIRECTORY = tempfile.mkdtemp()
    test_pipeline = cast(RegressionTransformerTrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()
    config["training_args"]["output_dir"] = TEMPORARY_DIRECTORY

    raw_path = pkg_resources.resource_filename(
        "gt4sd",
        "training_pipelines/tests/regression_transformer_raw.csv",
    )
    processed_path = pkg_resources.resource_filename(
        "gt4sd",
        "training_pipelines/tests/regression_transformer_selfies.txt",
    )

    # Test the QED model with csv setup
    config["model_args"]["model_path"] = mol_path
    config['dataset_args']['data_path'] = raw_path
    config['dataset_args']['augment'] = 2
    input_config = combine_defaults_and_user_args(config)
    test_pipeline.train(**input_config)

    # Test the QED model with processed setup
    config["model_args"]["model_path"] = mol_path
    config['dataset_args']['train_data_path'] = processed_path
    config['dataset_args']['test_data_path'] = processed_path
    input_config = combine_defaults_and_user_args(config)
    test_pipeline.train(**input_config)

    # Test training model from scratch with csv setup
    del config["model_args"]["model_path"]
    del config['dataset_args']['train_data_path']
    del config['dataset_args']['test_data_path']
    config["model_args"]["model_type"] = 'xlnet'
    config["model_args"]["tokenizer_name"] = mol_path
    config['dataset_args']['data_path'] = raw_path
    config['dataset_args']['augment'] = 2
    input_config = combine_defaults_and_user_args(config)
    test_pipeline.train(**input_config)

    # Test training model from scratch with processed setup
    config['dataset_args']['test_data_path'] = processed_path
    config['dataset_args']['train_data_path'] = processed_path
    config['dataset_args']['test_data_path'] = processed_path
    input_config = combine_defaults_and_user_args(config)
    test_pipeline.train(**input_config)
