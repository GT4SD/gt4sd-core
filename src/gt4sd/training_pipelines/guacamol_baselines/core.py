"""GuacaMol baselines training utilities."""

from dataclasses import dataclass, field
from typing import Any, Dict

from ..core import TrainingPipeline, TrainingPipelineArguments


class GuacaMolBaselinesTrainingPipeline(TrainingPipeline):
    """GuacaMol Baselines training pipelines."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
    ) -> None:
        """Generic training function for GuacaMol Baselines training.

        Args:
            training_args: training arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.

        Raises:
            NotImplementedError: the generic trainer does not implement the pipeline.
        """
        raise NotImplementedError


@dataclass
class GuacaMolDataArguments(TrainingPipelineArguments):
    """Arguments related to data loading."""

    __name__ = "dataset_args"

    train_smiles_filepath: str = field(
        metadata={"help": "Path of SMILES file for Training."}
    )
    test_smiles_filepath: str = field(
        metadata={"help": "Path of SMILES file for Validation."}
    )


@dataclass
class GuacaMolSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to GuacaMol trainer."""

    __name__ = "saving_args"

    model_filepath: str = field(metadata={"help": "Path to the model file."})
    model_config_filepath: str = field(
        metadata={"help": "Path to the model config file."}
    )
