"""Moses AAE Trainer unit tests."""

from typing import Any, Dict, cast
from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_MAPPING,
    MosesAAETrainingPipeline,
)

template_config = {
    "model_args": {
        "embedding_size": 32,
        "encoder_hidden_size": 512,
        "encoder_num_layers": 1,
        "encoder_bidirectional": True,
        "encoder_dropout": 0,
        "decoder_hidden_size": 512,
        "decoder_num_layers": 2,
        "decoder_dropout": 0,
        "latent_size": 128,
        "discriminator_layers": [640, 256],
    },
    "training_args": {
        "pretrain_epochs": 0,
        "train_epochs": 120,
        "n_batch": 512,
        "lr": 1e-3,
        "step_size": 20,
        "gamma": 0.5,
        "n_jobs": 1,
        "n_workers": 1,
        "discriminator_steps": 1,
        "weight_decay": 0,
    },
    "common_args": {
        "train_load": "/Users/ashishdave/Desktop/GT4SD/guacamol_baselines/guacamol_baselines/data/guacamol_v1_train.smiles",
        "val_load": "/Users/ashishdave/Desktop/GT4SD/guacamol_baselines/guacamol_baselines/data/guacamol_v1_train.smiles",
        "model_save": "/Users/ashishdave/desktop/GT4SD/gt4sd-core/",
        "log_file": "/Users/ashishdave/desktop/GT4SD/gt4sd-core/",
        "config_save": "/Users/ashishdave/desktop/GT4SD/gt4sd-core/config.pt",
        "vocab_save": "/Users/ashishdave/desktop/GT4SD/gt4sd-core/vocab.pt",
        "vocab_load": "/Users/ashishdave/.gt4sd/algorithms/conditional_generation/MosesGenerator/AaeGenerator/v0/vocab.pt",
        "seed": 0,
        "device": "cpu",
    },
}


def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("moses-aae-trainer")

    assert pipeline is not None

    test_pipeline = cast(MosesAAETrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()

    test_pipeline.train(**config)


test_train()
