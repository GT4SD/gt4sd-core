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
"""gflownet trainer unit tests."""

import os
import shutil
import tempfile
from typing import Any, Dict, cast

import numpy as np
import pytest

from gt4sd.frameworks.gflownet.envs.graph_building_env import GraphBuildingEnv
from gt4sd.frameworks.gflownet.envs.mol_building_env import MolBuildingEnvContext
from gt4sd.frameworks.gflownet.tests.qm9 import QM9Dataset, QM9GapTask
from gt4sd.training_pipelines import TRAINING_PIPELINE_MAPPING, GFlowNetTrainingPipeline


def _create_training_output_filepaths(directory: str) -> Dict[str, str]:
    """Create output filepath from directory.

    Args:
        directory: output directory.

    Returns:
        a dictionary containing the output files.
    """
    return {
        "log_file": os.path.join(directory, "log.txt"),
        "model_save": os.path.join(directory, "model.pt"),
        "config_save": os.path.join(directory, "config.pt"),
    }


template_config = {
    "model_args": {
        "algorithm": "trajectory_balance",
        "context": None,
        "environment": None,
        "model": "graph_transformer_gfn",
        "sampling_model": "graph_transformer_gfn",
        "task": "qm9",
        "bootstrap_own_reward": False,
        "num_emb": 128,
        "num_layers": 4,
        "tb_epsilon": 1e-10,
        "illegal_action_logreward": -50,
        "reward_loss_multiplier": 1.0,
        "temperature_sample_dist": "uniform",
        "temperature_dist_params": "(.5, 32)",
        "weight_decay": 1e-8,
        "momentum": 0.9,
        "adam_eps": 1e-8,
        "lr_decay": 20000,
        "z_lr_decay": 20000,
        "clip_grad_type": "norm",
        "clip_grad_param": 10.0,
        "random_action_prob": 0.001,
        "sampling_tau": 0.0,
        "max_nodes": 9,
        "num_offline": 10,
    },
    "pl_trainer_args": {
        "strategy": "ddp",
        "basename": "gflownet",
        "check_val_every_n_epoch": 5,
        "trainer_log_every_n_steps": 50,
        "auto_lr_find": True,
        "profiler": "simple",
        "learning_rate": 0.0001,
        "test_output_path": "./test",
        "num_workers": 0,
        "log_dir": "./log/",
        "save_dir": "./log/",
        "num_training_steps": 1000,
        "validate_every": 1000,
        "seed": 142857,
        "device": "cpu",
        "development_mode": True,
        "resume_from_checkpoint": None,
        "save_top_k": 1,
        "epochs": 3,
    },
    "dataset_args": {
        "dataset": "qm9",
        "dataset_path": "./data/qm9.h5",
        "epoch": 100,
        "batch_size": 64,
        "global_batch_size": 16,
        "validation_split": None,
        "sampling_iterator": True,
        "ratio": 0.9,
    },
}


@pytest.mark.skip(reason="we need to add support for dataset buckets")
def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("gflownet-trainer")

    assert pipeline is not None

    TEMPORARY_DIRECTORY = tempfile.mkdtemp()

    test_pipeline = cast(GFlowNetTrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()
    for key, value in _create_training_output_filepaths(TEMPORARY_DIRECTORY).items():
        config["pl_trainer_args"][key] = value
    config["pl_trainer_args"]["rng"] = np.random.default_rng(
        config["pl_trainer_args"]["seed"]
    )

    dataset = QM9Dataset(config["dataset_args"]["dataset_path"], target="gap")
    environment = GraphBuildingEnv()
    context = MolBuildingEnvContext()

    task = QM9GapTask(
        configuration={
            **config["pl_trainer_args"],
            **config["model_args"],
            **config["dataset_args"],
        },
        dataset=dataset,
    )

    config["dataset"] = dataset
    config["environment"] = environment
    config["context"] = context
    config["task"] = task

    test_pipeline.train(**config)

    shutil.rmtree(TEMPORARY_DIRECTORY)
