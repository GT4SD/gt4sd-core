"""Language modeling trainer unit tests."""
import os
from typing import Any, Dict, cast

import pytest

from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_MAPPING,
    GuacaMolLSTMPPOTrainingPipeline,
)

OUTPUT_DIR = "/tmp/guacamol_lstm_ppo/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

template_config = {
    "model_args": {
        "optimization_objective": {
            "isomer_scorer": {"target": 5.0, "target_smile": "NCCCCC"}
        },
        "max_seq_length": 100,
        "device": "cpu",
        "output_dir": OUTPUT_DIR,
    },
    "train_args": {
        "num_epochs": 1,
        "clip_param": 0.2,
        "batch_size": 10,
        "episode_size": 10,
        "entropy_weight": 1.0,
        "kl_div_weight": 5.0,
        "input_size": 47,
        "hidden_size": 1024,
        "output_size": 47,
        "n_layers": 3,
        "rnn_dropout": 0.2,
    },
}


@pytest.mark.skip(reason="not ready for testing")
def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("guacamol-lstm-ppo-trainer")

    assert pipeline is not None

    test_pipeline = cast(GuacaMolLSTMPPOTrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()

    test_pipeline.train(**config)
