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
"""TorchDrug GCPN trainer unit tests."""

import shutil
import tempfile
from typing import Any, Dict, cast

import importlib_resources

from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_MAPPING,
    TorchDrugGCPNTrainingPipeline,
)

template_config = {
    "model_args": {
        "hidden_dims": "[128, 128]",
        "batch_norm": True,
        "short_cut": True,
        "concat_hidden": True,
        "readout": "mean",
        "hidden_dim_mlp": 128,
        "agent_update_interval": 16,
        "gamma": 0.95,
        "reward_temperature": 1.2,
        "criterion": "{'nll': 1}",
    },
    "dataset_args": {
        "dataset_name": "freesolv",
        "lazy": True,
        "no_kekulization": False,
    },
    "training_args": {
        "model_path": "/tmp/torchdrug-gcpn",
        "training_name": "torchdrug-gcpn-test",
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.0005,
        "log_interval": 2,
        "gradient_interval": 2,
    },
}


def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("torchdrug-gcpn-trainer")

    assert pipeline is not None

    TEMPORARY_DIRECTORY = tempfile.mkdtemp()

    test_pipeline = cast(TorchDrugGCPNTrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()
    config["training_args"]["model_path"] = TEMPORARY_DIRECTORY
    config["training_args"]["dataset_path"] = TEMPORARY_DIRECTORY
    test_pipeline.train(**config)

    # Now test with a custom dataset
    with importlib_resources.as_file(
        importlib_resources.files("gt4sd") / "training_pipelines/tests/molecules.csv"
    ) as file_path:

        config["dataset_args"]["dataset_name"] = "custom"
        config["dataset_args"]["file_path"] = file_path
        config["dataset_args"]["smiles_field"] = "smiles"
        config["dataset_args"]["target_field"] = "qed"
        # This should filter out the QED again from the batches
        config["dataset_args"]["transform"] = "lambda x: {'graph': x['graph']}"
        test_pipeline.train(**config)

        # Test the property optimization
        config["training_args"]["task"] = "qed"
        config["dataset_args"]["node_feature"] = "symbol"
        config["model_args"]["criterion"] = "{'ppo': 1}"
        test_pipeline.train(**config)

    shutil.rmtree(TEMPORARY_DIRECTORY)
