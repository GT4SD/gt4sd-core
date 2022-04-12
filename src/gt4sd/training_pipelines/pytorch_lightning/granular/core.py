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
"""Granular training utilities."""

import json
import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from pytorch_lightning import LightningDataModule, LightningModule

from ....frameworks.granular.dataloader.data_module import GranularDataModule
from ....frameworks.granular.dataloader.dataset import build_dataset_and_architecture
from ....frameworks.granular.ml.models import AUTOENCODER_ARCHITECTURES
from ....frameworks.granular.ml.module import GranularModule
from ...core import TrainingPipelineArguments
from ..core import PyTorchLightningTrainingPipeline

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GranularTrainingPipeline(PyTorchLightningTrainingPipeline):
    """Granular training pipelines."""

    def get_data_and_model_modules(
        self,
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
        **kwargs,
    ) -> Tuple[LightningDataModule, LightningModule]:
        """Get data and model modules for training.

        Args:
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.

        Returns:
            the data and model modules.
        """

        configuration = {**model_args, **dataset_args}

        with open(model_args["model_list_path"], "r") as fp:  # type:ignore
            configuration["model_list"] = json.load(fp)["models"]

        arguments = Namespace(**configuration)
        datasets = []
        architecture_autoencoders = []
        architecture_latent_models = []
        for model in arguments.model_list:
            logger.info(f"dataset preparation for model={model}")
            hparams = configuration["model_list"][model]
            hparams["name"] = model
            model_type = hparams["type"].lower()
            dataset, architecture = build_dataset_and_architecture(
                hparams["name"],
                hparams["data_path"],
                hparams["data_file"],
                hparams["dataset_type"],
                hparams["type"],
                hparams,
            )
            datasets.append(dataset)
            if model_type in AUTOENCODER_ARCHITECTURES:
                architecture_autoencoders.append(architecture)
            else:
                architecture_latent_models.append(architecture)
        dm = GranularDataModule(
            datasets,
            batch_size=arguments.batch_size,
            validation_split=arguments.validation_split,
            validation_indices_file=arguments.validation_indices_file,
            stratified_batch_file=arguments.stratified_batch_file,
            stratified_value_name=arguments.stratified_value_name,
            num_workers=arguments.num_workers,
        )
        dm.prepare_data()
        module = GranularModule(
            architecture_autoencoders=architecture_autoencoders,
            architecture_latent_models=architecture_latent_models,
            lr=arguments.lr,
            test_output_path=arguments.test_output_path,
        )

        return dm, module


@dataclass
class GranularModelArguments(TrainingPipelineArguments):
    """
    Arguments related to model.
    """

    __name__ = "model_args"

    model_list_path: str = field(
        metadata={
            "help": "Path to a json file that contains a dictionary with models and their parameters."
        },
    )
    lr: float = field(
        default=0.0001,
        metadata={"help": "The learning rate."},
    )

    test_output_path: Optional[str] = field(
        default="./test",
        metadata={
            "help": "Path where to save latent encodings and predictions for the test set when an epoch ends. Defaults to a a folder called 'test' in the current working directory."
        },
    )


@dataclass
class GranularDataArguments(TrainingPipelineArguments):
    """
    Arguments related to data.
    """

    __name__ = "dataset_args"

    batch_size: int = field(
        default=64,
        metadata={"help": "Batch size of the training. Defaults to 64."},
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
    num_workers: int = field(
        default=1,
        metadata={"help": "number of workers. Defaults to 1."},
    )


@dataclass
class GranularSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to Granular trainer."""

    __name__ = "saving_args"

    model_path: str = field(
        metadata={"help": "Path to the checkpoint file to be used."}
    )
