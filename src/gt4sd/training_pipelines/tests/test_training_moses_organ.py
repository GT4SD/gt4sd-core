"""Moses Organ Trainer unit tests."""

from typing import Any, Dict, cast
from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_MAPPING,
    MosesOrganTrainingPipeline,
)
from moses.script_utils import MetricsReward
import pkg_resources

MODEL_ARTIFACTS_LOAD = VALID_FILE_PATH = pkg_resources.resource_filename(
    "gt4sd",
    "training_pipelines/tests/guacamol_test_data/",
)
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
        "generator_pretrain_epochs": 50,
        "discriminator_pretrain_epochs": 50,
        "pg_iters": 1000,
        "n_batch": 64,
        "lr": 1e-4,
        "n_workers": 8,
        "max_length": 1,
        "clip_grad": 5,
        "rollouts": 16,
        "generator_updates": 1,
        "discriminator_updates": 1,
        "discriminator_epochs": 10,
        "pg_smooth_const": 0.1,
        "n_ref_subsample": 500,
        "additional_rewards": [],
        "addition_rewards": MetricsReward.supported_metrics,
    },
    "common_args": {
        "train_load": MODEL_ARTIFACTS_LOAD + "/guacamol_v1_train.smiles",
        "val_load": MODEL_ARTIFACTS_LOAD + "/guacamol_v1_test.smiles",
        "model_save": MODEL_ARTIFACTS_LOAD + "/model_artifacts",
        "log_file": MODEL_ARTIFACTS_LOAD + "/model_artifacts",
        "config_save": MODEL_ARTIFACTS_LOAD + "config.pt",
        "vocab_save": MODEL_ARTIFACTS_LOAD + "/vocab.pt",
        "vocab_load": MODEL_ARTIFACTS_LOAD + "/vocab.pt",
        "seed": 0,
        "device": "cpu",
    },
}


def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("moses-organ-trainer")

    assert pipeline is not None

    test_pipeline = cast(MosesOrganTrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()

    test_pipeline.train(**config)
