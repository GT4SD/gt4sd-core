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

from gt4sd.frameworks.gflownet.arg_parser.parser import parse_arguments_from_config
from gt4sd.frameworks.gflownet.dataloader import build_dataset_and_architecture
from gt4sd.frameworks.gflownet.dataloader.data_module import GFlowNetDataModule
from gt4sd.frameworks.gflownet.envs import build_env_context
from gt4sd.frameworks.gflownet.ml.module import GFlowNetModule
from gt4sd.frameworks.gflownet.train import build_task

# sentencepiece has to be loaded before lightning to avoid segfaults
_sentencepiece

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def train_gflownet(configuration: Dict[str, Any]) -> None:
    """Train a gflownet given a configuration.
    Args:
        configuration: a configuration dictionary.
    """
    arguments = Namespace(**configuration)

    hparams = configuration
    model_type = hparams["type"].lower()
    dataset, architecture = build_dataset_and_architecture(
        hparams["name"],
        hparams["data_path"],
        hparams["data_file"],
        hparams["dataset_type"],
        hparams["type"],
        hparams,
    )

    env, ctx = build_env_context(hparams["env"], hparams["context"])
    task = build_task(hparams["task"])

    dm = GFlowNetDataModule(
        dataset,
        env,
        ctx,
        task,
        algo=getattr(arguments, "algorithm", "trajectory_balance"),
        model=getattr(arguments, "model_name", "graph_transformer"),
        batch_size=getattr(arguments, "batch_size", 64),
        validation_split=getattr(arguments, "validation_split", None),
        validation_indices_file=getattr(arguments, "validation_indices_file", None),
        stratified_batch_file=getattr(arguments, "stratified_batch_file", None),
        stratified_value_name=getattr(arguments, "stratified_value_name", None),
        num_workers=getattr(arguments, "num_workers", 1),
    )
    dm.prepare_data()

    module = GFlowNetModule(
        architecture=model_type,
        lr=getattr(arguments, "lr", 0.0001),
        test_output_path=getattr(arguments, "test_output_path", "./test"),
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
        max_epochs=getattr(arguments, "epoch", 1),
        flush_logs_every_n_steps=getattr(
            arguments, "trainer_flush_logs_every_n_steps", 100
        ),
    )
    trainer.fit(module, dm)


def train_gflownet_main() -> None:
    """Train a gflownet module parsing arguments from config and standard input."""
    train_gflownet(configuration=vars(parse_arguments_from_config()))
