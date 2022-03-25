"""Moses VAE Trainer unit tests."""

import os
from typing import Any, Dict, cast

import pkg_resources

from gt4sd.training_pipelines import TRAINING_PIPELINE_MAPPING, MosesVAETrainingPipeline

MODEL_ARTIFACTS_LOAD = VALID_FILE_PATH = pkg_resources.resource_filename(
    "gt4sd",
    "training_pipelines/tests/",
)
OUTPUT_DIR = "/tmp/moses_vae"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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
    },
    "common_args": {
        "train_load": os.path.join(MODEL_ARTIFACTS_LOAD, "molecules.smi"),
        "val_load": os.path.join(MODEL_ARTIFACTS_LOAD, "molecules.smi"),
        "vocab_load": os.path.join(
            MODEL_ARTIFACTS_LOAD, "guacamol_test_data", "vocab.pt"
        ),
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

    pipeline = TRAINING_PIPELINE_MAPPING.get("moses-vae-trainer")

    assert pipeline is not None

    test_pipeline = cast(MosesVAETrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()

    test_pipeline.train(**config)
