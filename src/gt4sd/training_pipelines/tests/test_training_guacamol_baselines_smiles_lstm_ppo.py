"""Language modeling trainer unit tests."""

from typing import Any, Dict, cast
from guacamol.scoring_function import MoleculewiseScoringFunction
from guacamol.score_modifier import GaussianModifier
from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_MAPPING,
    SMILESLSTMPPOTrainingPipeline,
)

template_config = {
    "model_args": {
        "model": "/Users/ashishdave/.gt4sd/algorithms/conditional_generation/GuacaMolGenerator/SMILESLSTMPPOGenerator/v0/model_final_0.473.pt",
        "optimization_objective": MoleculewiseScoringFunction(
            GaussianModifier(mu=2, sigma=0.5)
        ),
        "max_seq_length": 100,
        "device": "cpu",
        "num_epochs": 1,
        "clip_param": 0.2,
        "batch_size": 10,
        "episode_size": 10,
        "entropy_weight": 1.0,
        "kl_div_weight": 5.0,
    }
}


def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("smiles-lstm-ppo-trainer")

    assert pipeline is not None

    test_pipeline = cast(SMILESLSTMPPOTrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()

    test_pipeline.train(**config)


test_train()
