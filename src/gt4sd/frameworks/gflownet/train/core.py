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
from typing import Any, Callable, Dict

import sentencepiece as _sentencepiece
import numpy as np
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
    _task: Callable,
) -> None:
    """Train a gflownet given a configuration and lightning modules.
    dataset, enviroment, context and task are optional. The defaults are small molecules compatible.

    Args:
        configuration: a configuration dictionary.
        dataset: a dataset compatible with lightning.
        environment: an environment compatible with lightning.
        context: an environment context compatible with lightning.
        task: a task compatible with lightning.
    """

    arguments = Namespace(**configuration)

    # if not dataset:
    #     dataset = build_dataset(
    #         dataset=getattr(arguments, "dataset"),
    #         configuration=configuration,
    #     )
    # if not (environment or context):
    #     environment, context = build_env_context(
    #         environment_name=getattr(arguments, "environment"),
    #         context_name=getattr(arguments, "context"),
    #     )
    # if not task:
    #     task = build_task(task=getattr(arguments, "task"))

    algorithm = ALGORITHM_FACTORY[getattr(arguments, "algorithm")](
        configuration,
        environment,
        context,
    )
    model = MODEL_FACTORY[getattr(arguments, "model")](
        configuration,
        context,
    )

    task = _task(
        configuration=configuration,
        dataset=dataset,
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

    tensorboard_logger = TensorBoardLogger(
        "logs", name=getattr(arguments, "basename", "default")
    )
    checkpoint_callback = ModelCheckpoint(
        every_n_val_epochs=getattr(arguments, "checkpoint_every_n_val_epochs", 5),
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
        flush_logs_every_n_steps=getattr(
            arguments, "trainer_flush_logs_every_n_steps", 100
        ),
        fast_dev_run=getattr(arguments, "development", False),
        accelerator=getattr(arguments, "distributed_training_strategy", "ddp"),
    )
    trainer.fit(module, dm)


def train_gflownet_main(
    configuration: Dict[str, Any],
    dataset: GFlowNetDataset,
    environment: GraphBuildingEnv,
    context: GraphBuildingEnvContext,
    _task: Callable[[], GFlowNetTask],
) -> None:
    """Train a gflownet module parsing arguments from config and standard input."""

    def default_hps() -> Dict[str, Any]:
        return {
            "bootstrap_own_reward": False,
            "learning_rate": 1e-4,
            "global_batch_size": 16,
            "num_emb": 128,
            "num_layers": 4,
            "tb_epsilon": None,
            "illegal_action_logreward": -50,
            "reward_loss_multiplier": 1,
            "temperature_sample_dist": "uniform",
            "temperature_dist_params": "(.5, 32)",
            "weight_decay": 1e-8,
            "num_data_loader_workers": 8,
            "momentum": 0.9,
            "adam_eps": 1e-8,
            "lr_decay": 20000,
            "Z_lr_decay": 20000,
            "clip_grad_type": "norm",
            "clip_grad_param": 10,
            "random_action_prob": 0.001,
            "sampling_tau": 0.0,
            "max_nodes": 9,
            "num_offline": 10,
            "sampling_iterator": True,
            "ratio": 0.9,
            "distributed_training_strategy": "ddp",
            "development": False,
        }

    configuration["rng"] = np.random.default_rng(142857)

    # add default configuration
    configuration.update(default_hps())
    # add user configuration
    configuration.update(vars(parse_arguments_from_config()))
    # train gflownet
    train_gflownet(
        configuration=configuration,
        dataset=dataset,
        environment=environment,
        context=context,
        _task=_task,
    )
