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
"""Train module implementation."""

import logging
from argparse import Namespace
from typing import Any, Dict

import sentencepiece as _sentencepiece
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ..arg_parser.parser import parse_arguments_from_config

# from ..dataloader import build_dataset
from ..dataloader.data_module import GFlowNetDataModule
from ..dataloader.dataset import GFlowNetDataset, GFlowNetTask

# from ..envs import build_env_context
from ..envs.graph_building_env import GraphBuildingEnv, GraphBuildingEnvContext
from ..loss import ALGORITHM_FACTORY
from ..ml.models import MODEL_FACTORY
from ..ml.module import GFlowNetModule

# from ..train import build_task

# sentencepiece has to be loaded before lightning to avoid segfaults
_sentencepiece

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def train_gflownet(
    configuration: Dict[str, Any],
    dataset: GFlowNetDataset,
    environment: GraphBuildingEnv,
    context: GraphBuildingEnvContext,
    task: GFlowNetTask,
) -> None:
    """Train a gflownet given a configuration, a dataset and a task.
    The default enviroment and context are compatible with small molecules.

    Args:
        configuration: a configuration dictionary.
        dataset: a dataset compatible with lightning.
        environment: an environment specifying the state space.
        context: an environment context specifying how to combine states.
        task: a task specifying the reward structure.
    """

    arguments = Namespace(**configuration)

    if arguments.algorithm in ALGORITHM_FACTORY:
        algorithm = ALGORITHM_FACTORY[getattr(arguments, "algorithm")](
            configuration=configuration,
            environment=environment,
            context=context,
        )
    else:
        raise ValueError(f"Algorithm {arguments.algorithm} not supported.")

    if arguments.model in MODEL_FACTORY:
        model = MODEL_FACTORY[getattr(arguments, "model")](
            configuration=configuration,
            context=context,
        )
    else:
        raise ValueError(f"Model {arguments.model} not supported.")

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

    tensorboard_logger = TensorBoardLogger(
        "logs", name=getattr(arguments, "basename", "default")
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
    )
    trainer = pl.Trainer.from_argparse_args(
        arguments,
        profiler="simple",
        logger=tensorboard_logger,
        auto_lr_find=True,
        log_every_n_steps=getattr(arguments, "trainer_log_every_n_steps", 50),
        callbacks=[checkpoint_callback],
        max_epochs=getattr(arguments, "epoch", 10),
        check_val_every_n_epoch=getattr(arguments, "checkpoint_every_n_val_epochs", 5),
        fast_dev_run=getattr(arguments, "development_mode", False),
        strategy=getattr(arguments, "strategy", "ddp"),
    )
    trainer.fit(module, dm)


def train_gflownet_main(
    configuration: Dict[str, Any],
    dataset: GFlowNetDataset,
    environment: GraphBuildingEnv,
    context: GraphBuildingEnvContext,
    task: GFlowNetTask,
) -> None:
    """Train a gflownet module parsing arguments from config and standard input."""

    # add user configuration
    configuration.update(vars(parse_arguments_from_config()))
    # train gflownet
    train_gflownet(
        configuration=configuration,
        dataset=dataset,
        environment=environment,
        context=context,
        task=task,
    )
