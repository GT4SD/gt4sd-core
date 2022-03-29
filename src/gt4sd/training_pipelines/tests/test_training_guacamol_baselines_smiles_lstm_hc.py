"""Language modeling trainer unit tests."""

import os
from typing import Any, Dict, cast

import pkg_resources
import pytest

from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_MAPPING,
    GuacaMolLSTMHCTrainingPipeline,
)

OUTPUT_DIR = "/tmp/guacamol_lstm_hc/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

MODEL_ARTIFACTS_LOAD = VALID_FILE_PATH = pkg_resources.resource_filename(
    "gt4sd",
    "training_pipelines/tests/guacamol_test_data/",
)

template_config = {
    "model_args": {
        "batch_size": 512,
        "valid_every": 1000,
        "n_epochs": 10,
        "hidden_size": 512,
        "n_layers": 2,
        "rnn_dropout": 0.2,
        "lr": 1e-3,
    },
    "training_args": {
        "max_len": 100,
        "output_dir": OUTPUT_DIR,
    },
    "dataset_args": {},
}


@pytest.mark.skip(reason="not ready for testing")
def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("guacamol-lstm-hc-trainer")

    assert pipeline is not None

    test_pipeline = cast(GuacaMolLSTMHCTrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()
    file_path = os.path.join(MODEL_ARTIFACTS_LOAD, "guacamol_v1_test.smiles")

    config["dataset_args"]["train_smiles_filepath"] = file_path
    config["dataset_args"]["test_smiles_filepath"] = file_path
    test_pipeline.train(**config)
