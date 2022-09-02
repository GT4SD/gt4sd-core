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
"""GFlowNet training utilities."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import sentencepiece as _sentencepiece
from pytorch_lightning import LightningDataModule, LightningModule

from ....frameworks.gflownet.dataloader.data_module import (
    GFlowNetDataModule,
    GFlowNetTask,
)
from ....frameworks.gflownet.dataloader.dataset import GFlowNetDataset
from ....frameworks.gflownet.envs.graph_building_env import (
    GraphBuildingEnv,
    GraphBuildingEnvContext,
)
from ....frameworks.gflownet.loss import ALGORITHM_FACTORY
from ....frameworks.gflownet.ml.models import MODEL_FACTORY
from ....frameworks.gflownet.ml.module import GFlowNetModule
from ...core import TrainingPipelineArguments
from ..core import PytorchLightningTrainingArguments, PyTorchLightningTrainingPipeline

# sentencepiece has to be loaded before lightning to avoid segfaults
_sentencepiece

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GFlowNetTrainingPipeline(PyTorchLightningTrainingPipeline):
    """gflownet training pipelines."""

    # TODO: compatible signature with Pytorch Lightning training pipelines
    def get_data_and_model_modules(  # type: ignore
        self,
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
        pl_training_args: Dict[str, Any],
        dataset: GFlowNetDataset,
        environment: GraphBuildingEnv,
        context: GraphBuildingEnvContext,
        task: GFlowNetTask,
    ) -> Tuple[LightningDataModule, LightningModule]:
        """Get data and model modules for training.

        Args:
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.

        Returns:
            the data and model modules.
        """

        configuration = {**model_args, **dataset_args, **pl_training_args}

        if configuration["algorithm"] in ALGORITHM_FACTORY:
            algorithm = ALGORITHM_FACTORY[getattr(configuration, "algorithm")](
                configuration,
                environment,
                context,
            )
        else:
            raise ValueError(
                "Algorithm configuration is not given in the specified config file."
            )

        if configuration["model"] in MODEL_FACTORY:
            model = MODEL_FACTORY[getattr(configuration, "model")](
                configuration,
                context,
            )
        else:
            raise ValueError(
                "Models configuration is not given in the specified config file."
            )

        dm = GFlowNetDataModule(
            configuration=configuration,
            dataset=dataset,
            environment=environment,
            context=context,
            task=task,
            algorithm=algorithm,
            model=model,
        )
        dm.prepare_data()

        module = GFlowNetModule(
            configuration=configuration,
            dataset=dataset,
            environment=environment,
            context=context,
            task=task,
            algorithm=algorithm,
            model=model,
        )

        return dm, module


@dataclass
class GFlowNetPytorchLightningTrainingArguments(PytorchLightningTrainingArguments):
    """
    Arguments related to pytorch lightning trainer.
    """

    __name__ = "pl_trainer_args"

    basename: str = field(
        default="gflownet",
        metadata={"help": "The basename as the name for the run."},
    )

    every_n_val_epochs: Optional[int] = field(
        default=5,
        metadata={"help": "Number of training epochs between checkpoints."},
    )
    auto_lr_find: bool = field(
        default=True,
        metadata={
            "help": "Select whether to run a learning rate finder to try to optimize initial learning for faster convergence."
        },
    )
    profiler: Optional[str] = field(
        default="simple",
        metadata={
            "help": "To profile individual steps during training and assist in identifying bottlenecks."
        },
    )
    learning_rate: float = field(
        default=0.0001,
        metadata={"help": "The learning rate."},
    )

    test_output_path: Optional[str] = field(
        default="./test",
        metadata={
            "help": "Path where to save latent encodings and predictions for the test set when an epoch ends."
        },
    )
    num_workers: int = field(
        default=0,
        metadata={"help": "number of workers. Defaults to 1."},
    )

    log_dir: str = field(
        default="./log/",
        metadata={"help": "The directory to save logs."},
    )

    num_training_steps: int = field(
        default=1000,
        metadata={"help": "The number of training steps."},
    )

    validate_every: int = field(
        default=1000,
        metadata={"help": "The number of training steps between validation."},
    )

    seed: int = field(
        default=142857,
        metadata={"help": "The random seed."},
    )

    device: str = field(
        default="cpu",
        metadata={"help": "The device to use."},
    )
    distributed_training_strategy: str = field(
        default="ddp",
        metadata={"help": "The distributed training strategy. "},
    )

    development_mode: bool = field(
        default=False,
        metadata={"help": "Whether to run in development mode. "},
    )


@dataclass
class GFlowNetModelArguments(TrainingPipelineArguments):
    """
    Arguments related to model.
    """

    __name__ = "model_args"

    algorithm: str = field(
        default="trajectory_balance",
        metadata={"help": "The algorithm to use for training the model. "},
    )
    context: str = field(  # type: ignore
        default=None,
        metadata={"help": "The environment context to use for training the model. "},
    )
    environment: str = field(  # type: ignore
        default=None,
        metadata={"help": "The environment to use for training the model. "},
    )
    model: str = field(
        default="graph_transformer_gfn",
        metadata={"help": "The model to use for training the model. "},
    )
    sampling_model: str = field(
        default="graph_transformer_gfn",
        metadata={"help": "The model used to generate samples. "},
    )
    task: str = field(
        default="qm9",
        metadata={"help": "The task to use for training the model. "},
    )

    bootstrap_own_reward: bool = field(
        default=False,
        metadata={"help": "Whether to bootstrap the own reward. "},
    )

    num_emb: int = field(
        default=128,
        metadata={"help": "The number of embeddings. "},
    )

    num_layers: int = field(
        default=4,
        metadata={"help": "The number of layers. "},
    )

    tb_epsilon: float = field(
        default=1e-10,
        metadata={"help": "The epsilon. "},
    )

    illegal_action_logreward: float = field(
        default=-50.0,
        metadata={"help": "The illegal action log reward. "},
    )

    reward_loss_multiplier: float = field(
        default=1.0,
        metadata={"help": "The reward loss multiplier. "},
    )

    temperature_sample_dist: str = field(
        default="uniform",
        metadata={"help": "The temperature sample distribution. "},
    )

    temperature_dist_params: str = field(
        default="(.5, 32)",
        metadata={"help": "The temperature distribution parameters. "},
    )

    weight_decay: float = field(
        default=1e-8,
        metadata={"help": "The weight decay. "},
    )
    momentum: float = field(
        default=0.9,
        metadata={"help": "The momentum. "},
    )

    adam_eps: float = field(
        default=1e-8,
        metadata={"help": "The adam epsilon. "},
    )

    lr_decay: float = field(
        default=20000,
        metadata={"help": "The learning rate decay steps. "},
    )

    z_lr_decay: float = field(
        default=20000,
        metadata={"help": "The learning rate decay steps for z."},
    )

    clip_grad_type: str = field(
        default="norm",
        metadata={"help": "The clip grad type. "},
    )

    clip_grad_param: float = field(
        default=10.0,
        metadata={"help": "The clip grad param. "},
    )

    random_action_prob: float = field(
        default=0.001,
        metadata={"help": "The random action probability. "},
    )

    sampling_tau: float = field(
        default=0.0,
        metadata={"help": "The sampling temperature. "},
    )

    max_nodes: int = field(
        default=9,
        metadata={"help": "The maximum number of nodes. "},
    )

    num_offline: int = field(
        default=10,
        metadata={"help": "The number of offline samples. "},
    )


@dataclass
class GFlowNetDataArguments(TrainingPipelineArguments):
    """
    Arguments related to data.
    """

    __name__ = "dataset_args"

    dataset: str = field(
        default="qm9",
        metadata={"help": "The dataset to use for training the model. "},
    )
    dataset_path: str = field(
        default="./data/qm9",
        metadata={"help": "The path to the dataset to use for training the model. "},
    )
    epoch: int = field(
        default=100,
        metadata={"help": "The number of epochs. "},
    )

    batch_size: int = field(
        default=64,
        metadata={"help": "Batch size of the training. Defaults to 64."},
    )
    global_batch_size: int = field(
        default=16,
        metadata={"help": "Global batch size of the training. Defaults to 16."},
    )

    validation_split: Optional[float] = field(
        default=None,
        metadata={
            "help": "Proportion used for validation. Defaults to None, a.k.a., use indices file if provided otherwise uses half of the data for validation."
        },
    )
    validation_indices_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Indices to use for validation. Defaults to None, a.k.a., use validation split proportion, if not provided uses half of the data for validation."
        },
    )
    stratified_batch_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Stratified batch file for sampling. Defaults to None, a.k.a., no stratified sampling."
        },
    )
    stratified_value_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Stratified value name. Defaults to None, a.k.a., no stratified sampling. Needed in case a stratified batch file is provided."
        },
    )
    num_data_loader_workers: int = field(
        default=8,
        metadata={"help": "The number of data loader workers. "},
    )
    sampling_iterator: bool = field(
        default=True,
        metadata={"help": "Whether to use a sampling iterator. "},
    )

    ratio: float = field(
        default=0.9,
        metadata={"help": "The ratio. "},
    )


@dataclass
class GFlowNetSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to Granular trainer."""

    __name__ = "saving_args"

    model_path: str = field(
        metadata={"help": "Path to the checkpoint file to be used."}
    )
