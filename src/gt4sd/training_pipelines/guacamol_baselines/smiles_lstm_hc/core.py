import logging
from dataclasses import dataclass, field
from typing import Any, Dict
from guacamol_baselines.smiles_lstm_hc.smiles_rnn_distribution_learner import (
    SmilesRnnDistributionLearner,
)
from ...core import TrainingPipelineArguments
from ..core import GuacamolBaselinesTrainingPipeline

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GuacamolLSTMHCTrainingPipeline(GuacamolBaselinesTrainingPipeline):
    """SMILES LSTM HC training pipelines."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
    ) -> None:
        params = {**training_args, **model_args, **dataset_args}
        with open(params.pop("train_smiles_filepath")) as f:
            train_list = f.readlines()

        with open(params.pop("test_smiles_filepath")) as f:
            valid_list = f.readlines()
        trainer = SmilesRnnDistributionLearner(**params)

        trainer.train(training_set=train_list, validation_set=valid_list)


@dataclass
class GuacamolLSTMHCTrainingArguments(TrainingPipelineArguments):
    """Training Arguments related to SMILES LSTM HC trainer."""

    __name__ = "training_args"

    batch_size: int = field(
        default=512, metadata={"help": "Size of a mini-batch for gradient descent"}
    )
    valid_every: int = field(
        default=1000, metadata={"help": "Validate every so many batches"}
    )
    n_epochs: int = field(default=10, metadata={"help": "Number of training epochs"})
    hidden_size: int = field(default=512, metadata={"help": "Size of hidden layer"})
    n_layers: int = field(default=3, metadata={"help": "Number of layers for training"})
    rnn_dropout: float = field(default=0.2, metadata={"help": "Dropout value for RNN"})
    lr: float = field(default=1e-3, metadata={"help": "RNN learning rate"})


@dataclass
class GuacamolLSTMHCModelArguments(TrainingPipelineArguments):
    """Arguments related to SMILES LSTM HC trainer."""

    __name__ = "model_args"

    max_len: int = field(
        default=100, metadata={"help": "Max length of a SMILES string"}
    )
    output_dir: str = field(default="", metadata={"help": "Output directory"})


@dataclass
class GuacamolLSTMHCDataArguments(TrainingPipelineArguments):
    """Arguments related to SMILES LSTM HC data loading."""

    __name__ = "dataset_args"

    train_smiles_filepath: str = field(
        metadata={"help": "Path of SMILES file for Training."}
    )
    test_smiles_filepath: str = field(
        metadata={"help": "Path of SMILES file for Validation."}
    )


@dataclass
class GuacamolLSTMHCSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to PaccMann trainer."""

    __name__ = "saving_args"

    model_path: str = field(
        metadata={"help": "Path where the model artifacts are stored."}
    )
