"""Moses VAE trainer unit tests."""

import os
import shutil
import tempfile
from typing import Any, Dict, cast

import pkg_resources

from gt4sd.training_pipelines import TRAINING_PIPELINE_MAPPING, MosesVAETrainingPipeline

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
        "q_cell": "gru",
        "q_bidir": False,
        "q_d_h": 256,
        "q_n_layers": 1,
        "q_dropout": 0.5,
        "d_cell": "gru",
        "d_n_layers": 3,
        "d_dropout": 0,
        "d_d_h": 512,
        "d_z": 128,
        "freeze_embeddings": False,
    },
    "training_args": {
        "n_batch": 512,
        "grad_clipping": 50,
        "kl_start": 0,
        "kl_w_start": 0,
        "kl_w_end": 0.05,
        "lr_start": 3 * 1e-4,
        "lr_n_period": 1,
        "lr_n_restarts": 1,
        "lr_n_mult": 1,
        "lr_end": 3 * 1e-4,
        "n_last": 1000,
        "n_jobs": 1,
        "n_workers": 1,
        "save_frequency": 1,
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

    pipeline = TRAINING_PIPELINE_MAPPING.get("moses-vae-trainer")

    assert pipeline is not None

    TEMPORARY_DIRECTORY = tempfile.mkdtemp()

    test_pipeline = cast(MosesVAETrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()
    for key, value in _create_training_output_filepaths(TEMPORARY_DIRECTORY).items():
        config["training_args"][key] = value

    test_pipeline.train(**config)

    shutil.rmtree(TEMPORARY_DIRECTORY)
