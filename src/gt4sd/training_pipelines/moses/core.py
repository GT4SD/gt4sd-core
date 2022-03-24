"""Guacamol Baselines training utilities."""

from typing import Any, Dict
from dataclasses import dataclass, field
from ..core import TrainingPipeline, TrainingPipelineArguments


class MosesTrainingPipeline(TrainingPipeline):
    """PyTorch lightining training pipelines."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        common_args: Dict[str, Any],
    ) -> None:
        """Generic training function for Guacamol Baselines training.

        Args:
            training_args: training arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            common_args: common arguments passed to the configuration.

        Raises:
            NotImplementedError: the generic trainer does not implement the pipeline.
        """
        raise NotImplementedError


@dataclass
class MosesCommonArguments(TrainingPipelineArguments):
    """Arguments related to PaccMann trainer."""

    __name__ = "common_args"
    train_load: str = field(metadata={"help": "Input data in csv format to train."})
    val_load: str = field(metadata={"help": "Input data in csv format to validation."})
    model_save: str = field(metadata={"help": "Path to where save the trained model."})
    log_file: str = field(metadata={"help": "Log file path to where save the logs."})
    config_save: str = field(metadata={"help": "Path for the config."})
    vocab_save: str = field(metadata={"help": "Path to save the model vocabulary"})
    vocab_load: str = field(metadata={"help": "Path to retrieve the model vocabulary"})
    save_frequency: int = field(
        default=1, metadata={"help": "How often to save the model"}
    )
    seed: int = field(default=0, metadata={"help": "Seed"})
    device: str = field(
        default="cpu",
        metadata={"help": 'Device to run: "cpu" or "cuda:<device number>"'},
    )


@dataclass
class MosesSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to PaccMann trainer."""

    __name__ = "saving_args"

    model_path: str = field(
        metadata={"help": "Path where the model artifacts are stored."}
    )
    config_path: str = field(
        metadata={"help": "Path where the config artifacts are stored."}
    )
    vocab_path: str = field(
        metadata={"help": "Path where the vocab artifacts are stored."}
    )
