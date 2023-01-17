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
"""CGCNN trainer unit tests."""

import shutil
import tempfile
from typing import Any, Dict, cast

import pytest

from gt4sd.training_pipelines import TRAINING_PIPELINE_MAPPING, CGCNNTrainingPipeline

template_config = {
    "model_args": {
        "atom_fea_len": 64,
        "h_fea_len": 128,
        "n_conv": 3,
        "n_h": 1,
    },
    "training_args": {
        "task": "classification",
        "disable_cuda": True,
        "epochs": 5,
        "batch_size": 256,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "optim": "SGD",
    },
    "dataset_args": {
        "datapath": "./data/cgcnn_sample_classification",
    },
}


@pytest.mark.skip(reason="we need to add support for dataset buckets")
def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("cgcnn")

    assert pipeline is not None

    TEMPORARY_DIRECTORY = tempfile.mkdtemp()

    test_pipeline = cast(CGCNNTrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()
    config["training_args"]["output_path"] = TEMPORARY_DIRECTORY

    test_pipeline.train(**config)

    shutil.rmtree(TEMPORARY_DIRECTORY)
