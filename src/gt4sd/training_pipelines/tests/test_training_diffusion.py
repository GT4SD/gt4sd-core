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
"""Diffusion trainer unit tests."""

import os
import shutil
import tempfile
from typing import Any, Dict, cast

from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_MAPPING,
    DiffusionForVisionTrainingPipeline,
)


def _create_training_output_filepaths(directory: str) -> Dict[str, str]:
    """Create output filepath from directory.

    Args:
        directory: output directory.

    Returns:
        a dictionary containing the output files.
    """
    return {
        "config_save": os.path.join(directory, "config.pt"),
    }


template_config = {
    "model_args": {
        "model_path": "",
        "training_name": "ddpm",
        "num_train_timesteps": 1000,
        "learning_rate": 1e-4,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 500,
        "adam_beta1": 0.95,
        "adam_beta2": 0.999,
        "adam_weight_decay": 1e-6,
        "adam_epsilon": 1e-8,
        "gradient_accumulation_steps": 1,
        "in_channels": 3,
        "out_channels": 3,
        "layers_per_block": 2,
    },
    "training_args": {
        "local_rank": -1,
        "output_dir": "./outputs/",
        "save_images_epochs": 10,
        "save_model_epochs": 10,
        "mixed_precision": "no",
        "logging_dir": "./logs/",
        "cache_dir": "./cifar10/",
        "use_auth_token": False,
        "ema_inv_gamma": 1.0,
        "ema_power": 0.75,
        "ema_max_decay": 0.9999,
        "use_ema": True,
        "dummy_training": True,
        "is_sampling": False,
    },
    "dataset_args": {
        "dataset_name": "cifar10",
        "dataset_config_name": None,
        "resolution": 32,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "num_epochs": 1,
    },
}


def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("diffusion-trainer")

    assert pipeline is not None

    TEMPORARY_DIRECTORY = tempfile.mkdtemp()

    test_pipeline = cast(DiffusionForVisionTrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()
    for key, value in _create_training_output_filepaths(TEMPORARY_DIRECTORY).items():
        config["training_args"][key] = value

    test_pipeline.train(**config)

    shutil.rmtree(TEMPORARY_DIRECTORY)
