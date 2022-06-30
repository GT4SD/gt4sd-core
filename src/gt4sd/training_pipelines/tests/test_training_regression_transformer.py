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

import json
import os
import tempfile
from typing import Any, Dict, Iterable, cast

import pkg_resources

from gt4sd.algorithms.conditional_generation.regression_transformer import (
    RegressionTransformerMolecules,
)
from gt4sd.cli.argument_parser import DataClassType
from gt4sd.cli.trainer import TrainerArgumentParser
from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_ARGUMENTS_MAPPING,
    TRAINING_PIPELINE_MAPPING,
    RegressionTransformerTrainingPipeline,
)
from gt4sd.training_pipelines.core import TrainingPipelineArguments

template_config = {
    "model_args": {},
    "dataset_args": {"test_fraction": 0.5},
    "training_args": {
        "training_name": "regression-transformer-test",
        "batch_size": 4,
        "learning_rate": 0.0005,
        "do_train": True,
        "eval_accumulation_steps": 1,
        "eval_steps": 100,
        "save_steps": 100,
        "max_steps": 3,
        "epochs": 1,
        "overwrite_output_dir": True,
    },
}

xlnet_config = {
    "architectures": ["XLNetLMHeadModel"],
    "attn_type": "bi",
    "bi_data": False,
    "bos_token_id": 14,
    "clamp_len": -1,
    "d_head": 8,
    "d_inner": 512,
    "d_model": 32,
    "dropout": 0.2,
    "end_n_top": 5,
    "eos_token_id": 14,
    "ff_activation": "gelu",
    "initializer_range": 0.02,
    "language": "selfies",
    "layer_norm_eps": 1e-12,
    "model_type": "xlnet",
    "n_head": 4,
    "n_layer": 8,
    "ne_dim": 16,
    "ne_format": "sum",
    "ne_type": "float",
    "pad_token_id": 0,
    "same_length": False,
    "start_n_top": 5,
    "summary_activation": "tanh",
    "summary_last_dropout": 0.1,
    "summary_type": "last",
    "summary_use_proj": True,
    "task_specific_params": {"text-generation": {"do_sample": True, "max_length": 250}},
    "untie_r": True,
    "use_ne": True,
    "vmax": 1.0,
}


def combine_defaults_and_user_args(
    config: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:

    arguments = TRAINING_PIPELINE_ARGUMENTS_MAPPING["regression-transformer-trainer"]
    """
    We need `conflict_handler='resolve'` because the RT relies on the TrainingArguments
    in HuggingFace. Some arguments, like `output_dir` do not have defaults and thus
    an "empty" parser like the below complains once we call `parse_args_into_dataclasses`.
    Therefore, we manually add the `output_dir` argument with the correct value.
    """
    parser = TrainerArgumentParser(
        cast(
            Iterable[DataClassType],
            tuple([*arguments]),
        ),
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
        default=config["training_args"]["output_dir"],
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
        default=config["dataset_args"]["train_data_path"],
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
        default=config["dataset_args"]["test_data_path"],
    )
    args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    input_config = {
        arg.__name__: arg.__dict__
        for arg in args
        if isinstance(arg, TrainingPipelineArguments) and isinstance(arg.__name__, str)
    }
    input_config["model_args"].update(**config["model_args"])
    input_config["dataset_args"].update(**config["dataset_args"])
    input_config["training_args"].update(**config["training_args"])

    return input_config


def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("regression-transformer-trainer")
    assert pipeline is not None

    mol_model = RegressionTransformerMolecules(algorithm_version="qed")
    mol_path = mol_model.ensure_artifacts_for_version("qed")

    TEMPORARY_DIRECTORY = tempfile.mkdtemp()
    test_pipeline = cast(RegressionTransformerTrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()
    config["training_args"]["output_dir"] = TEMPORARY_DIRECTORY
    raw_path = pkg_resources.resource_filename(
        "gt4sd",
        "training_pipelines/tests/regression_transformer_raw.csv",
    )

    # Test the pretrained QED model
    config["model_args"]["model_path"] = mol_path
    config["dataset_args"]["train_data_path"] = raw_path
    config["dataset_args"]["test_data_path"] = raw_path
    config["dataset_args"]["augment"] = 2
    input_config = combine_defaults_and_user_args(config)
    test_pipeline.train(**input_config)

    # Test training model from scratch
    with tempfile.TemporaryDirectory() as temp:
        f_name = os.path.join(temp, "tmp_xlnet_config.json")
        # Write file
        with open(f_name, "w") as f:
            json.dump(xlnet_config, f, indent=4)

        config["model_args"]["config_name"] = f_name
        del config["model_args"]["model_path"]
        config["model_args"]["tokenizer_name"] = mol_path
        config["dataset_args"]["data_path"] = raw_path
        config["dataset_args"]["augment"] = 2
        input_config = combine_defaults_and_user_args(config)
        test_pipeline.train(**input_config)
