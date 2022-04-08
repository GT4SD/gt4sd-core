"""SMILES LSTM training pipeline from GuacaMol."""
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict

from guacamol_baselines.smiles_lstm_hc.smiles_rnn_distribution_learner import (
    SmilesRnnDistributionLearner,
)

from ...core import TrainingPipelineArguments
from ..core import GuacaMolBaselinesTrainingPipeline

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GuacaMolLSTMTrainingPipeline(GuacaMolBaselinesTrainingPipeline):
    """GuacaMol SMILES LSTM training pipeline."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
    ) -> None:
        params = {**training_args, **model_args, **dataset_args}

        os.makedirs(params["output_dir"], exist_ok=True)

        with open(params.pop("train_smiles_filepath")) as f:
            train_list = f.readlines()

        with open(params.pop("test_smiles_filepath")) as f:
            valid_list = f.readlines()
        trainer = SmilesRnnDistributionLearner(**params)

        trainer.train(training_set=train_list, validation_set=valid_list)


@dataclass
class GuacaMolLSTMTrainingArguments(TrainingPipelineArguments):
    """Training Arguments related to SMILES LSTM trainer."""

    __name__ = "training_args"

    output_dir: str = field(metadata={"help": "Output directory."})
    batch_size: int = field(
        default=512, metadata={"help": "Size of a mini-batch for gradient descent."}
    )
    valid_every: int = field(
        default=1000, metadata={"help": "Validate every so many batches."}
    )
    n_epochs: int = field(default=10, metadata={"help": "Number of training epochs."})
    lr: float = field(default=1e-3, metadata={"help": "RNN learning rate."})


@dataclass
class GuacaMolLSTMModelArguments(TrainingPipelineArguments):
    """Arguments related to SMILES LSTM trainer."""

    __name__ = "model_args"

    hidden_size: int = field(default=512, metadata={"help": "Size of hidden layer."})
    n_layers: int = field(
        default=3, metadata={"help": "Number of layers for training."}
    )
    rnn_dropout: float = field(default=0.2, metadata={"help": "Dropout value for RNN."})
    max_len: int = field(
        default=100, metadata={"help": "Max length of a SMILES string."}
    )
