"""Moses baselines training utilities."""

from dataclasses import dataclass, field
from typing import Any, Dict

from ..core import TrainingPipeline, TrainingPipelineArguments


class MosesTrainingPipeline(TrainingPipeline):
    """PyTorch lightining training pipelines."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        common_args: Dict[str, Any],
    ) -> None:
        """Generic training function for GuacaMol Baselines training.

        Args:
            training_args: training arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            common_args: common arguments passed to the configuration.

        Raises:
            NotImplementedError: the generic trainer does not implement the pipeline.
        """
        raise NotImplementedError


@dataclass
class MosesDataArguments(TrainingPipelineArguments):
    """Arguments related to Moses data loading."""

    __name__ = "dataset_args"
    train_load: str = field(
        metadata={"help": "Input data in csv format used for training."}
    )
    val_load: str = field(
        metadata={"help": "Input data in csv format used for validation."}
    )


@dataclass
class MosesTrainingArguments(TrainingPipelineArguments):
    """Arguments related to Moses trainer."""

    __name__ = "training_args"
    model_save: str = field(metadata={"help": "Path where the trained model is saved."})
    log_file: str = field(metadata={"help": "Path where to save the the logs."})
    config_save: str = field(metadata={"help": "Path for the config."})
    vocab_save: str = field(metadata={"help": "Path to save the model vocabulary."})
    save_frequency: int = field(
        default=1, metadata={"help": "How often to save the model."}
    )
    seed: int = field(
        default=0, metadata={"help": "Seed used for random number generation."}
    )
    device: str = field(
        default="cpu",
        metadata={"help": "Device to run: 'cpu' or 'cuda:<device number>'"},
    )


@dataclass
class MosesSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to PaccMann trainer."""

    __name__ = "saving_args"

    model_path: str = field(metadata={"help": "Path where the model is stored."})
    config_path: str = field(metadata={"help": "Path where the config is stored."})
    vocab_path: str = field(metadata={"help": "Path where the vocab is stored."})
