"""Language modeling trainer unit tests."""

from typing import Any, Dict, cast


from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_MAPPING,
    SMILESLSTMHCTrainingPipeline,
)

template_config = {
    "model_args": {
        "batch_size": 512,
        "valid_every": 1000,
        "n_epochs": 10,
        "max_len": 100,
        "hidden_size": 512,
        "n_layers": 2,
        "rnn_dropout": 0.2,
        "lr": 1e-3,
        "output_dir": "/Users/ashishdave/Desktop/GT4SD/gt4sd-core/",
    },
    "dataset_args": {},
}


def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("smiles-lstm-hc-trainer")

    assert pipeline is not None

    test_pipeline = cast(SMILESLSTMHCTrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()

    config["dataset_args"][
        "train_smiles_filepath"
    ] = "/Users/ashishdave/Desktop/GT4SD/guacamol_baselines/guacamol_baselines/data/guacamol_v1_train.smiles"
    config["dataset_args"][
        "test_smiles_filepath"
    ] = "/Users/ashishdave/Desktop/GT4SD/guacamol_baselines/guacamol_baselines/data/guacamol_v1_train.smiles"
    test_pipeline.train(**config)
