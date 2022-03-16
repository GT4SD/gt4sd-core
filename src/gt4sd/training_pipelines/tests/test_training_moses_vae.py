"""Moses VAE Trainer unit tests."""

from typing import Any, Dict, cast
from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_MAPPING,
    MosesVAETrainingPipeline,
)

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
        "lr_n_period": 10,
        "lr_n_restarts": 10,
        "lr_n_mult": 1,
        "lr_end": 3 * 1e-4,
        "n_last": 1000,
        "n_jobs": 1,
        "n_workers": 1,
    },
    "common_args": {
        "train_load": "/Users/ashishdave/Desktop/GT4SD/guacamol_baselines/guacamol_baselines/data/guacamol_v1_train.smiles",
        "val_load": "/Users/ashishdave/Desktop/GT4SD/guacamol_baselines/guacamol_baselines/data/guacamol_v1_train.smiles",
        "model_save": "/Users/ashishdave/desktop/GT4SD/gt4sd-core/",
        "log_file": "/Users/ashishdave/desktop/GT4SD/gt4sd-core/log.txt",
        "config_save": "/Users/ashishdave/desktop/GT4SD/gt4sd-core/config.pt",
        "vocab_save": "/Users/ashishdave/desktop/GT4SD/gt4sd-core/vocab.pt",
        "vocab_load": "/Users/ashishdave/.gt4sd/algorithms/conditional_generation/MosesGenerator/AaeGenerator/v0/vocab.pt",
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


test_train()
