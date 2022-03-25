import logging
from dataclasses import dataclass, field
from typing import Any, Dict

from guacamol_baselines.smiles_lstm_hc.rnn_model import SmilesRnn
from guacamol_baselines.smiles_lstm_ppo.ppo_trainer import PPOTrainer
from guacamol_baselines.smiles_lstm_ppo.rnn_model import SmilesRnnActorCritic

from gt4sd.algorithms.conditional_generation.guacamol.implementation import (
    CombinedScorer,
    get_target_parameters,
)

from ...core import TrainingPipelineArguments
from ..core import GuacamolBaselinesTrainingPipeline

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GuacamolLSTMPPOTrainingPipeline(GuacamolBaselinesTrainingPipeline):
    """SMILES LSTM PPO training pipelines."""

    def train(self, model_args: Dict[str, Any], train_args: Dict[str, Any]) -> None:  # type: ignore
        params = {**model_args, **train_args}
        score_list, weights = get_target_parameters(params["optimization_objective"])
        params["optimization_objective"] = CombinedScorer(
            scorer_list=score_list,
            weights=weights,
        )
        model_params = [
            params.pop("input_size"),
            params.pop("hidden_size"),
            params.pop("output_size"),
            params.pop("n_layers"),
            params.pop("rnn_dropout"),
        ]
        model = SmilesRnn(*model_params)
        model.to(params["device"])
        model = SmilesRnnActorCritic(smiles_rnn=model).to(params["device"])
        params["model"] = model
        trainer = PPOTrainer(**params)
        trainer.train()


@dataclass
class GuacamolLSTMPPOModelArguments(TrainingPipelineArguments):
    """Arguments related to SMILES LSTM PPO trainer."""

    __name__ = "training_args"

    model: str = field(metadata={"help": "Model Path."})
    optimization_objective: Dict[str, Any] = field(
        metadata={"help": "Scoring Function used for training."}
    )
    max_seq_length: int = field(
        default=1000, metadata={"help": "Maximum length for sequence."}
    )
    device: str = field(
        default="cpu",
        metadata={"help": 'Device to run: "cpu" or "cuda:<device number>".'},
    )


@dataclass
class GuacamolLSTMPPOTrainingArguments(TrainingPipelineArguments):
    """Arguments related to SMILES LSTM PPO trainer."""

    __name__ = "training_args"

    num_epochs: int = field(
        default=10, metadata={"help": "Number of epochs to sample."}
    )
    clip_param: float = field(
        default=0.2,
        metadata={
            "help": "Used for determining how far the new policy is from the old one."
        },
    )
    batch_size: int = field(
        default=3, metadata={"help": "Batch size for the optimization."}
    )
    episode_size: int = field(
        default=8192,
        metadata={
            "help": "Number of molecules sampled by the policy at the start of a series of ppo updates."
        },
    )
    entropy_weight: float = field(
        default=1.0, metadata={"help": "Used for calculating entropy loss."}
    )
    kl_div_weight: float = field(
        default=5.0,
        metadata={"help": "Used for calculating Kullback-Leibler divergence loss."},
    )
    input_size: int = field(
        default=47,
        metadata={"help": "Number of input symbols."},
    )
    hidden_size: int = field(
        default=1024,
        metadata={"help": "Number of hidden units."},
    )
    output_size: int = field(
        default=47,
        metadata={"help": "Number of output symbols."},
    )
    n_layers: int = field(
        default=3,
        metadata={"help": "Number of hidden layers."},
    )
    rnn_dropout: float = field(
        default=0.2,
        metadata={"help": "Recurrent dropout."},
    )


@dataclass
class GuacamolLSTMPPOSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to PaccMann trainer."""

    __name__ = "saving_args"

    model_path: str = field(
        metadata={"help": "Path where the model artifacts are stored."}
    )
