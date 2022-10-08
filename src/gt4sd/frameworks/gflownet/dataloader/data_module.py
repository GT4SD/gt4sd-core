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
"""Data module for gflownet."""

import logging
from typing import Any, Dict, Optional

import sentencepiece as _sentencepiece
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader  # , Subset, random_split

from ..dataloader.dataset import GFlowNetDataset, GFlowNetTask
from ..envs.graph_building_env import GraphBuildingEnv, GraphBuildingEnvContext
from ..loss.trajectory_balance import TrajectoryBalance
from ..ml.models import MODEL_FACTORY
from .sampler import SamplingIterator

# sentencepiece has to be loaded before lightning to avoid segfaults
_sentencepiece

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GFlowNetDataModule(pl.LightningDataModule):
    """Data module from gflownet."""

    def __init__(
        self,
        configuration: Dict[str, Any],
        dataset: GFlowNetDataset,
        environment: GraphBuildingEnv,
        context: GraphBuildingEnvContext,
        task: GFlowNetTask,
        algorithm: TrajectoryBalance,
        model: Optional[nn.Module] = None,
    ) -> None:
        """Construct GFlowNetDataModule.

        The module assumes a model and algorithm factory/registry.
        The user should provide a dataset, environment, context for the environment, and task.

        Args:
            configuration: configuration dictionary.
            dataset: dataset.
            environment: environment for graph building.
            context: context environment.
            task: generic task.
            algorithm: loss function.
            model: model used to generate data with the sampling iterator.
                It can be a custom model or the same as the one used in the algorithm.
        """
        super().__init__()
        self.hps = configuration

        # if model is given
        if model:
            self.sampling_model = model
        else:
            self.sampling_model = MODEL_FACTORY[self.hps["sampling_model"]](
                self.hps, context
            )
        self.algo = algorithm
        self.env = environment
        self.ctx = context
        self.dataset = dataset
        self.task = task

        self.sampling_iterator = self.hps["sampling_iterator"]
        self.batch_size = self.hps["batch_size"]
        self.num_workers = self.hps["num_workers"]
        self.device = self.hps["device"]
        self.rng = self.hps["rng"]
        self.ratio = self.hps["ratio"]
        self.mb_size = self.hps["global_batch_size"]

    def prepare_data(self) -> None:
        """Prepare training and test dataset."""
        self.train_dataset = self.dataset
        self.val_dataset = self.dataset
        self.test_dataset = self.dataset

    def setup(self, stage: Optional[str]) -> None:  # type:ignore
        """Setup the data module.

        Args:
            stage: stage considered. Defaults to None.
        """

        ll = self.dataset.get_len()
        ixs = np.arange(ll)
        self.rng.shuffle(ixs)
        thresh = int(np.floor(self.ratio * ll))

        self.ix_train = ixs[: int(0.9 * thresh)]
        self.ix_val = ixs[int(0.9 * thresh) : thresh]
        self.ix_test = ixs[thresh:]

        if stage == "fit" or stage is None:
            self.train_dataset.set_indexes(self.ix_train)  # type: ignore
            self.val_dataset.set_indexes(self.ix_val)  # type: ignore
        if stage == "test" or stage is None:
            self.test_dataset.set_indexes(self.ix_test)  # type: ignore
        if stage == "predict" or stage is None:
            self.test_dataset.set_indexes(self.ix_test)  # type: ignore

        logger.info(
            f"number of data points used for training: {len(self.train_dataset)}"
        )
        logger.info(f"number of data points used for testing: {len(self.test_dataset)}")
        logger.info(
            f"testing proportion: {len(self.test_dataset) / (len(self.test_dataset) + len(self.train_dataset))}"
        )

    def train_dataloader(self) -> DataLoader:
        """Get a data loader for training.

        Returns:
            a training data loader.
        """
        if self.sampling_iterator:
            iterator = SamplingIterator(
                self.train_dataset,
                self.sampling_model,
                self.mb_size * 2,
                self.ctx,
                self.algo,
                self.task,
                device=self.device,
            )
            batch_size = None
        else:
            iterator = self.train_dataset  # type: ignore
            batch_size = self.batch_size

        return DataLoader(
            iterator,
            batch_size=batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Get a data loader for validation.

        Returns:
            a validation data loader.
        """
        if self.sampling_iterator:
            iterator = SamplingIterator(
                self.val_dataset,
                self.sampling_model,
                self.mb_size,
                self.ctx,
                self.algo,
                self.task,
                ratio=1,
                stream=False,
                device=self.device,
            )
            batch_size = None
        else:
            iterator = self.val_dataset  # type: ignore
            batch_size = self.batch_size

        return DataLoader(
            iterator,
            batch_size=batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Get a data loader for testing.

        Returns:
            a testing data loader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """Get a data loader for prediction.

        Returns:
            a prediction data loader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )
