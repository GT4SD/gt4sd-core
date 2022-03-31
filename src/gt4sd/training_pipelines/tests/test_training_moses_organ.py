"""Moses Organ trainer unit tests."""

import os
import shutil
import tempfile
from typing import Any, Dict, cast

import pkg_resources

from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_MAPPING,
    MosesOrganTrainingPipeline,
)

TEST_DATA_DIRECTORY = pkg_resources.resource_filename(
    "gt4sd",
    "training_pipelines/tests/",
)


def _create_training_output_filepaths(directory: str) -> Dict[str, str]:
    """Create output filepath from directory.

    Args:
        directory: output directory.

    Returns:
        a dictionary containing the output files.
    """
    return {
        "log_file": os.path.join(directory, "log.txt"),
        "model_save": os.path.join(directory, "model.pt"),
        "config_save": os.path.join(directory, "config.pt"),
        "vocab_save": os.path.join(directory, "vocab.pt"),
    }


template_config = {
    "model_args": {
        "embedding_size": 32,
        "hidden_size": 512,
        "num_layers": 2,
        "dropout": 0.0,
        "discriminator_layers": "[(100, 1), (200, 2), (200, 3), (200, 4), (200, 5), (100, 6), (100, 7), (100, 8), (100, 9), (100, 10), (160, 15), (160, 20)]",
        "discriminator_dropout": 0.0,
    },
    "training_args": {
        "reward_weight": 0.7,
        "n_jobs": 1,
        "generator_pretrain_epochs": 1,
        "discriminator_pretrain_epochs": 1,
        "pg_iters": 1,
        "n_batch": 4,
        "lr": 1e-4,
        "n_workers": 1,
        "max_length": 50,
        "clip_grad": 5,
        "rollouts": 16,
        "generator_updates": 1,
        "discriminator_updates": 1,
        "discriminator_epochs": 1,
        "n_ref_subsample": 1,
        "addition_rewards": "sa,qed",
        "seed": 0,
        "device": "cpu",
        "save_frequency": 1,
    },
    "dataset_args": {
        "train_load": os.path.join(TEST_DATA_DIRECTORY, "molecules.smiles"),
        "val_load": os.path.join(TEST_DATA_DIRECTORY, "molecules.smiles"),
    },
}


def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("moses-organ-trainer")

    assert pipeline is not None

    TEMPORARY_DIRECTORY = tempfile.mkdtemp()

    test_pipeline = cast(MosesOrganTrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()
    for key, value in _create_training_output_filepaths(TEMPORARY_DIRECTORY).items():
        config["training_args"][key] = value

    test_pipeline.train(**config)

    shutil.rmtree(TEMPORARY_DIRECTORY)
