"""Language modeling trainer unit tests."""

from typing import Any, Dict, cast

import pkg_resources

from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_MAPPING,
    GuacamolLSTMHCTrainingPipeline,
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
        "output_dir": "/Users/ashishdave/Desktop/GT4SD/gt4sd-core/",
    },
    "dataset_args": {},
}


def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("guacamol-lstm-hc-trainer")

    assert pipeline is not None

    test_pipeline = cast(GuacamolLSTMHCTrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()
    file_path = pkg_resources.resource_filename(
        "gt4sd",
        "training_pipelines/tests/molecules.smi",
    )

    config["dataset_args"]["train_smiles_filepath"] = file_path
    config["dataset_args"]["test_smiles_filepath"] = file_path
    test_pipeline.train(**config)
