"""Language modeling trainer unit tests."""

from typing import Any, Dict, cast

import pkg_resources

from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_MAPPING,
    PaccMannVAETrainingPipeline,
)

template_config = {
    "model_args": {
        "n_layers": 1,
        "bidirectional": False,
        "rnn_cell_size": 64,
        "latent_dim": 32,
        "stack_width": 8,
        "stack_depth": 8,
        "decoder_search": "sampling",
        "dropout": 0.2,
        "generate_len": 50,
        "kl_growth": 0.003,
        "input_keep": 0.85,
        "test_input_keep": 1.0,
        "temperature": 0.8,
        "embedding": "one_hot",
        "batch_mode": "packed",
        "vocab_size": 380,
        "pad_index": 0,
        "embedding_size": 380,
    },
    "dataset_args": {
        "add_start_stop_token": True,
        "selfies": True,
        "num_workers": 1,
        "pin_memory": False,
    },
    "training_args": {
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.0005,
        "optimizer": "adam",
        "log_interval": 2,
        "save_interval": 2,
        "eval_interval": 2,
        "model_path": "/tmp/paccmann_vae",
        "training_name": "paccmann-vae-test",
    },
}


def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("paccmann-vae-trainer")

    assert pipeline is not None

    test_pipeline = cast(PaccMannVAETrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()

    file_path = pkg_resources.resource_filename(
        "gt4sd",
        "training_pipelines/tests/molecules.smi",
    )

    config["dataset_args"]["train_smiles_filepath"] = file_path
    config["dataset_args"]["test_smiles_filepath"] = file_path
    test_pipeline.train(**config)
