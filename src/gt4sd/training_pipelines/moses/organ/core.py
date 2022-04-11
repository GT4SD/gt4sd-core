#
# MIT License
#
# Copyright (c) 2022 GT4SD team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""Moses Organ training pipeline."""
import argparse
import ast
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict

from guacamol_baselines.moses_baselines.organ_train import main
from moses.script_utils import MetricsReward

from ...core import TrainingPipelineArguments
from ..core import MosesTrainingArguments, MosesTrainingPipeline

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MosesOrganTrainingPipeline(MosesTrainingPipeline):
    """Moses Organ training pipelines."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
    ) -> None:
        """Generic training function for Moses Organ training.

        Args:
            training_args: training arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.
        """
        params = {**training_args, **model_args, **dataset_args}

        os.makedirs(os.path.dirname(params["model_save"]), exist_ok=True)
        os.makedirs(os.path.dirname(params["log_file"]), exist_ok=True)
        os.makedirs(os.path.dirname(params["config_save"]), exist_ok=True)
        os.makedirs(os.path.dirname(params["vocab_save"]), exist_ok=True)
        params["addition_rewards"] = list(
            map(str.strip, params["addition_rewards"].split(","))
        )
        params["discriminator_layers"] = ast.literal_eval(
            params["discriminator_layers"]
        )

        args = argparse.Namespace(**params)
        main(args)


@dataclass
class MosesOrganTrainingArguments(MosesTrainingArguments):
    """Arguments related to Moses Organ training."""

    generator_pretrain_epochs: int = field(
        default=50, metadata={"help": "Number of epochs for generator pretraining."}
    )
    discriminator_pretrain_epochs: int = field(
        default=50, metadata={"help": "Number of epochs for discriminator pretraining."}
    )
    pg_iters: int = field(
        default=1000,
        metadata={"help": "Number of iterations for policy gradient training."},
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
    reward_weight: float = field(
        default=0.7, metadata={"help": "Reward weight for policy gradient training."}
    )
    addition_rewards: str = field(
        default="sa",
        metadata={
            "help": f"Comma separated list of rewards. Feasible values from: {','.join(MetricsReward.supported_metrics)}. Defaults to optimization of SA."
        },
    )
    max_length: int = field(
        default=100, metadata={"help": "Maximum length for sequence."}
    )
    n_ref_subsample: int = field(
        default=500,
        metadata={
            "help": "Number of reference molecules (sampling from training data)."
        },
    )


@dataclass
class MosesOrganModelArguments(TrainingPipelineArguments):
    """Arguments related to Moses Organ model."""

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
    dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout probability for lstm layers in generator."},
    )
    discriminator_layers: str = field(
        default="[(100, 1), (200, 2), (200, 3), (200, 4), (200, 5), (100, 6), (100, 7), (100, 8), (100, 9), (100, 10), (160, 15), (160, 20)]",
        metadata={
            "help": "String representation of numbers of features for convolutional layers in discriminator."
        },
    )
    discriminator_dropout: float = field(
        default=0.0, metadata={"help": "Dropout probability for discriminator."}
    )
