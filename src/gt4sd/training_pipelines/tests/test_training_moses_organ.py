"""Moses Organ Trainer unit tests."""

from typing import Any, Dict, cast

import pkg_resources
import os

from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_MAPPING,
    MosesOrganTrainingPipeline,
)

MODEL_ARTIFACTS_LOAD = VALID_FILE_PATH = pkg_resources.resource_filename(
    "gt4sd",
    "training_pipelines/tests/guacamol_test_data/",
)

OUTPUT_DIR = "/tmp/moses_organ"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

template_config = {
    "model_args": {
        "embedding_size": 32,
        "hidden_size": 512,
        "num_layers": 2,
        "dropout": 0,
        "discriminator_layers": [
            (100, 1),
            (200, 2),
            (200, 3),
            (200, 4),
            (200, 5),
            (100, 6),
            (100, 7),
            (100, 8),
            (100, 9),
            (100, 10),
            (160, 15),
            (160, 20),
        ],
        "discriminator_dropout": 0,
        "reward_weight": 0.7,
        "n_jobs": 8,
        "generator_pretrain_epochs": 1,
        "discriminator_pretrain_epochs": 1,
        "pg_iters": 1,
        "n_batch": 4,
        "lr": 1e-4,
        "n_workers": 8,
        "max_length": 50,
        "clip_grad": 5,
        "rollouts": 4,
        "generator_updates": 1,
        "discriminator_updates": 1,
        "discriminator_epochs": 1,
        "pg_smooth_const": 0.1,
        "n_ref_subsample": 1,
        "additional_rewards": [],
        "addition_rewards": [],
    },
    "common_args": {
        "train_load": os.path.join(MODEL_ARTIFACTS_LOAD, "guacamol_v1_train.smiles"),
        "val_load": os.path.join(MODEL_ARTIFACTS_LOAD, "guacamol_v1_test.smiles"),
        "vocab_load": os.path.join(MODEL_ARTIFACTS_LOAD, "vocab.pt"),
        "log_file": os.path.join(OUTPUT_DIR, "log.txt"),
        "model_save": os.path.join(OUTPUT_DIR, "model.pt"),
        "config_save": os.path.join(OUTPUT_DIR, "config.pt"),
        "vocab_save": os.path.join(OUTPUT_DIR, "vocab.pt"),
        "seed": 0,
        "device": "cpu",
        "save_frequency": 1,
    },
}


def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("moses-organ-trainer")

    assert pipeline is not None

    test_pipeline = cast(MosesOrganTrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()

    test_pipeline.train(**config)
