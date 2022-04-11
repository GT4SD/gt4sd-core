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
"""GuacaMol LSTM trainer unit tests."""

import os
import shutil
import tempfile
from typing import Any, Dict, cast

import pkg_resources

from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_MAPPING,
    GuacaMolLSTMTrainingPipeline,
)

TEST_DATA_DIRECTORY = pkg_resources.resource_filename(
    "gt4sd",
    "training_pipelines/tests/",
)

template_config = {
    "training_args": {
        "batch_size": 512,
        "valid_every": 1000,
        "n_epochs": 10,
        "lr": 1e-3,
    },
    "model_args": {
        "max_len": 100,
        "hidden_size": 512,
        "n_layers": 2,
        "rnn_dropout": 0.2,
    },
    "dataset_args": {},
}


def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("guacamol-lstm-trainer")

    assert pipeline is not None

    TEMPORARY_DIRECTORY = tempfile.mkdtemp()

    test_pipeline = cast(GuacaMolLSTMTrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()
    file_path = os.path.join(TEST_DATA_DIRECTORY, "molecules.smiles")
    config["training_args"]["output_dir"] = TEMPORARY_DIRECTORY
    config["dataset_args"]["train_smiles_filepath"] = file_path
    config["dataset_args"]["test_smiles_filepath"] = file_path

    test_pipeline.train(**config)

    shutil.rmtree(TEMPORARY_DIRECTORY)
