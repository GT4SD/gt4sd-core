import logging
from dataclasses import dataclass, field
from typing import Any, Dict
from pathlib import Path
from guacamol_baselines.smiles_lstm_ppo.ppo_trainer import PPOTrainer
from guacamol_baselines.smiles_lstm_hc.rnn_utils import load_rnn_model
from guacamol_baselines.smiles_lstm_ppo.rnn_model import SmilesRnnActorCritic
from ...core import TrainingPipelineArguments
from ..core import GuacamolBaselinesTrainingPipeline

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SMILESLSTMPPOTrainingPipeline(GuacamolBaselinesTrainingPipeline):
    """SMILES LSTM PPO training pipelines."""

    def train(self, model_args: Dict[str, Any]) -> None:  # type: ignore
        model_def = Path(model_args["model_path"]).with_suffix(".json")
        smiles_rnn = load_rnn_model(
            model_def,
            model_args["model_path"],
            model_args["device"],
            copy_to_cpu=True,
        )
        model = SmilesRnnActorCritic(smiles_rnn=smiles_rnn).to(model_args["device"])
        model_args["model"] = model
        trainer = PPOTrainer(**model_args)
        trainer.train()


@dataclass
class SMILESLSTMPPOModelArguments(TrainingPipelineArguments):
    """Arguments related to SMILES LSTM PPO trainer."""

    __name__ = "training_args"

    model: str = field(metadata={"help": "Model Path."})
    optimization_objective: str = field(
        metadata={"help": "Scoring Function used for training."}
    )
    max_seq_length: int = field(
        default=1000, metadata={"help": "Maximum length for sequence."}
    )
    device: str = field(
        default="cpu",
        metadata={"help": 'Device to run: "cpu" or "cuda:<device number>".'},
    )
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
