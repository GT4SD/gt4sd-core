import argparse
import logging
from dataclasses import dataclass, field
from typing import Any, Dict

from guacamol_baselines.moses_baselines.organ_train import main
from moses.script_utils import MetricsReward

from ...core import TrainingPipelineArguments
from ..core import MosesDataArguments, MosesTrainingArguments, MosesTrainingPipeline

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MosesOrganTrainingPipeline(MosesTrainingPipeline):
    """Moses ORGAN training pipelines."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
    ) -> None:
        """Generic training function for MOSES ORGAN training.

        Args:
            training_args: training arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.
        """
        params = {**training_args, **model_args, **dataset_args}
        parser = argparse.ArgumentParser()
        for k, v in params.items():
            parser.add_argument("--" + k, default=v)
        args = parser.parse_known_args()[0]
        main(args)


@dataclass
class MosesOrganDataArguments(MosesDataArguments):
    """Arguments related to MOSES ORGAN data loading."""

    __name__ = "dataset_args"

    n_ref_subsample: int = field(
        default=500,
        metadata={
            "help": "Number of reference molecules (sampling from training data)."
        },
    )


@dataclass
class MosesOrganTrainingArguments(MosesTrainingArguments):
    """Arguments related to MOSES ORGAN training."""

    generator_pretrain_epochs: int = field(
        default=50, metadata={"help": "Number of epochs for generator pretraining."}
    )
    discriminator_pretrain_epochs: float = field(
        default=50, metadata={"help": "Number of epochs for discriminator pretraining."}
    )
    pg_iters: int = field(
        default=1000,
        metadata={"help": "Number of inerations for policy gradient training."},
    )
    n_batch: int = field(default=64, metadata={"help": "Size of batch."})
    lr: float = field(default=1e-4, metadata={"help": "Learning rate."})
    n_jobs: int = field(default=8, metadata={"help": "Number of threads."})
    n_workers: int = field(default=8, metadata={"help": "Number of workers."})
    clip_grad: int = field(
        default=5, metadata={"help": "Clip PG generator gradients to this value."}
    )
    rollouts: int = field(default=16, metadata={"help": "Number of rollouts."})
    generator_updates: int = field(
        default=1, metadata={"help": "Number of updates of generator per iteration."}
    )
    discriminator_updates: int = field(
        default=1,
        metadata={"help": "Number of updates of discriminator per iteration."},
    )
    discriminator_epochs: int = field(
        default=10,
        metadata={"help": "Number of epochs of discriminator per iteration."},
    )
    pg_smooth_const: float = field(
        default=0.1, metadata={"help": "Smoothing factor for Policy Gradient logs."}
    )
    reward_weight: float = field(
        default=0.7, metadata={"help": "Reward weight for policy gradient training."}
    )
    additional_rewards: list = field(
        default_factory=lambda: [], metadata={"help": "Adding of addition rewards."}
    )
    addition_rewards: list = field(
        default_factory=lambda: MetricsReward.supported_metrics,
        metadata={"help": "Adding of addition rewards."},
    )
    max_length: int = field(
        default=1, metadata={"help": "Maximum length for sequence."}
    )


@dataclass
class MosesOrganModelArguments(TrainingPipelineArguments):
    """Arguments related to MOSES ORGAN model."""

    __name__ = "model_args"

    embedding_size: int = field(
        default=32, metadata={"help": "Embedding size in generator and discriminator."}
    )
    hidden_size: int = field(
        default=512,
        metadata={"help": "Size of hidden state for lstm layers in generator."},
    )
    num_layers: int = field(
        default=2, metadata={"help": "Number of lstm layers in generator."}
    )
    dropout: int = field(
        default=0,
        metadata={"help": "Dropout probability for lstm layers in generator."},
    )
    discriminator_layers: list = field(
        default_factory=lambda: [
            (100, 1),
            (200, 2),
            (200, 3),
            (200, 4),
            (200, 5),
            (100, 6),
            (100, 7),
            (100, 8),
            (100, 9),
            (100, 10),
            (160, 15),
            (160, 20),
        ],
        metadata={
            "help": "Numbers of features for convalution layers in discriminator."
        },
    )
    discriminator_dropout: int = field(
        default=0, metadata={"help": "Dropout probability for discriminator."}
    )
