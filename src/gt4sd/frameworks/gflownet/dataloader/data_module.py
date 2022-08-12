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
"""Data module for granular."""

import logging
from typing import Any, Dict, List, NewType, Optional, Tuple

import sentencepiece as _sentencepiece
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch.utils.data import DataLoader  # , Subset, random_split

from ..envs.graph_building_env import (
    GraphActionCategorical,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
)
from ..ml.models import MODEL_FACTORY
from ..util import wrap_model_mp
from .dataset import GFlowNetDataset
from .sampling_iterator import SamplingIterator

# sentencepiece has to be loaded before lightning to avoid segfaults
_sentencepiece

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# This type represents an unprocessed list of reward signals/conditioning information
FlatRewards = NewType("FlatRewards", torch.tensor)  # type: ignore

# This type represents the outcome for a multi-objective task of
# converting FlatRewards to a scalar, e.g. (sum R_i omega_i) ** beta
RewardScalar = NewType("RewardScalar", torch.tensor)  # type: ignore


class GFlowNetTask:
    """Abstract class for a generic task."""

    def __init__():
        pass

    """We consider the task as part of the dataset (environment)"""

    def cond_info_to_reward(
        self, cond_info: Dict[str, torch.Tensor], flat_reward: FlatRewards
    ) -> RewardScalar:
        """Combines a minibatch of reward signal vectors and conditional information into a scalar reward.

        Args:
            cond_info: a dictionary with various conditional informations (e.g. temperature).
            flat_reward: a 2d tensor where each row represents a series of flat rewards.

        Returns:
            reward: a 1d tensor, a scalar reward for each minibatch entry.
        """
        raise NotImplementedError()

    def compute_flat_rewards(self, x: List[Any]) -> Tuple[RewardScalar, torch.Tensor]:
        """Compute the flat rewards of mols according the the tasks' proxies.

        Args:
            mols: a list of RDKit molecules.
        Returns:
            reward: a 1d tensor, a scalar reward for each molecule.
            is_valid: a 1d tensor, a boolean indicating whether the molecule is valid.
        """
        raise NotImplementedError()

    def _wrap_model_mp(self, model):
        """Wraps a nn.Module instance so that it can be shared to `DataLoader` workers."""
        if self.num_workers > 0:
            placeholder = wrap_model_mp(
                model, self.num_workers, cast_types=(gd.Batch, GraphActionCategorical)
            )
            return placeholder
        return model


class GFlowNetDataModule(pl.LightningDataModule):
    """Data module from gflownet.
    We assume to have a model and algorithm factory/registry. The user should provide
    a dataset, environment, context for the environment, and task.
    """

    def __init__(
        self,
        configuration: Dict[str, Any],
        dataset: GFlowNetDataset,
        environment: GraphBuildingEnv,
        context: GraphBuildingEnvContext,
        task: GFlowNetTask,
        algorithm: nn.Module,
        model: nn.Module,
        sampling_model: str,
        sampling_iterator: Optional[bool] = True,
    ) -> None:
        """Construct GFlowNetDataModule.

        Args:
            dataset: dataset.
            environment: environment for graph building.
            context: context env.
            task: generic task.
            algorithm: loss function.
            model: model type.
            sampling_model:
            sampling_iterator: sampling iterator to use.
        """
        super().__init__()
        self.hps = configuration

        self.model = model
        self.sampling_model = MODEL_FACTORY[sampling_model](
            context,
            num_emb=self.hps["num_emb"],
            num_layers=self.hps["num_layers"],
        )
        self.algo = algorithm
        self.env = environment
        self.ctx = context
        self.dataset = dataset
        self.task = task

        self.sampling_iterator = sampling_iterator

        self.batch_size = self.hps["batch_size"]
        self.num_workers = self.hps["num_workers"]

        # self.validation_split = self.hps["validation_split"]
        # self.validation_indices_file = self.hps["validation_indices_file"]
        # self.stratified_batch_file = self.hps["stratified_batch_file"]
        # self.stratified_value_name = self.hps["stratified_value_name"]

        self.device = self.hps["device"]
        self.rng = self.hps["rng"]
        self.ratio = self.hps["ratio"]
        self.mb_size = self.hps["global_batch_size"]

    def prepare_data(self) -> None:
        """Prepare training and test dataset."""
        self.train_dataset = self.dataset
        self.test_dataset = self.dataset

    def setup(self, stage: Optional[str]) -> None:
        """Setup the data module.
        Args:
            stage: stage considered, unused. Defaults to None.
        """

        ll = self.dataset.get_len()
        ixs = np.arange(ll)
        self.rng.shuffle(ixs)

        # TODO: use Subset?
        self.ix_train = ixs[: int(np.floor(self.ratio * ll))]
        self.ix_test = ixs[int(np.floor(self.ratio * ll)) :]

        if stage == "fit" or stage is None:
            self.train_dataset.set_indexes(self.ix_train)
        if stage == "test" or stage is None:
            self.test_dataset.set_indexes(self.ix_test)

        logger.info(
            f"number of data points used for training: {len(self.train_dataset)}"
        )
        logger.info(f"number of data points used for testing: {len(self.test_dataset)}")
        logger.info(
            f"testing proportion: {len(self.test_dataset) / (len(self.test_dataset) + len(self.train_dataset))}"
        )

    # @staticmethod
    # def get_stratified_batch_sampler(
    #     stratified_batch_file: str,
    #     stratified_value_name: str,
    #     batch_size: int,
    #     selector_fn: Callable[[pd.DataFrame], pd.DataFrame],
    # ) -> StratifiedSampler:
    #     """Get stratified batch sampler.

    #     Args:
    #         stratified_batch_file: stratified batch file for sampling.
    #         stratified_value_name: stratified value name.
    #         batch_size: batch size.
    #         selector_fn: selector function for stratified sampling.
    #     Returns:
    #         a stratified batch sampler.
    #     """
    #     stratified_batch_dataframe = pd.read_csv(stratified_batch_file)
    #     stratified_data = stratified_batch_dataframe[
    #         selector_fn(stratified_batch_dataframe)
    #     ][stratified_value_name].values
    #     stratified_data_tensor = torch.from_numpy(stratified_data)
    #     return StratifiedSampler(targets=stratified_data_tensor, batch_size=batch_size)

    def train_dataloader(self) -> DataLoader:
        """Get a training data loader.

        Returns:
            a training data loader.
        """
        if self.sampling_iterator:
            # model, dev = self._wrap_model_mp(self.model)
            iterator = SamplingIterator(
                self.train_dataset,
                self.model,
                self.mb_size * 2,
                self.ctx,
                self.algo,
                self.task,
                self.device,
            )
        else:
            iterator = self.train_dataset
        return DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Get a validation data loader.

        Returns:
            a validation data loader.
        """
        if self.sampling_iterator:
            # model, dev = self._wrap_model_mp(self.model)
            iterator = SamplingIterator(
                self.test_dataset,  # TODO: add validation set
                self.model,
                self.mb_size,
                self.ctx,
                self.algo,
                self.task,
                self.device,
                ratio=1,
                stream=False,
            )
        else:
            iterator = self.test_dataset
        return DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Get a testing data loader.

        Returns:
            a testing data loader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )
