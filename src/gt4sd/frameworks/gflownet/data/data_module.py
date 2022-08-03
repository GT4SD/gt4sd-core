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
from typing import Callable, List, Optional, cast

import sentencepiece as _sentencepiece
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, Subset, random_split

from gt4sd.frameworks.gflownet.data.dataset import GFlowNetDataset
#from gt4sd.frameworks.gflownet.data.sampling_iterator import SamplingIterator
from gt4sd.frameworks.gflownet.envs.graph_building_env import (
    GraphBuildingEnv,
    GraphBuildingEnvContext,
)

# sentencepiece has to be loaded before lightning to avoid segfaults
_sentencepiece

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# This type represents an unprocessed list of reward signals/conditioning information
FlatRewards = NewType("FlatRewards", Tensor)  # type: ignore

# This type represents the outcome for a multi-objective task of
# converting FlatRewards to a scalar, e.g. (sum R_i omega_i) ** beta
RewardScalar = NewType("RewardScalar", Tensor)  # type: ignore


class GFlowNetTask:
    def cond_info_to_reward(
        self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards
    ) -> RewardScalar:
        """Combines a minibatch of reward signal vectors and conditional information into a scalar reward.

        Args:
            cond_info: a dictionary with various conditional informations (e.g. temperature).
            flat_reward: a 2d tensor where each row represents a series of flat rewards.

        Returns:
            reward: a 1d tensor, a scalar reward for each minibatch entry.
        """
        raise NotImplementedError()

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[RewardScalar, Tensor]:
        """Compute the flat rewards of mols according the the tasks' proxies.

        Args:
            mols: a list of RDKit molecules.
        Returns:
            reward: a 1d tensor, a scalar reward for each molecule.
            is_valid: a 1d tensor, a boolean indicating whether the molecule is valid.
        """
        raise NotImplementedError()


class GFlowNetDataModule(pl.LightningDataModule):
    """Data module from granular."""

    def __init__(
        self,
        dataset_list: List[GFlowNetDataset],
        validation_split: Optional[float] = None,
        validation_indices_file: Optional[str] = None,
        stratified_batch_file: Optional[str] = None,
        stratified_value_name: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 1,
    ) -> None:
        """Construct GFlowNetDataModule.

        Args:
            dataset_list: a list of granular datasets.
            validation_split: proportion used for validation. Defaults to None,
                a.k.a., use indices file if provided otherwise uses half of the data for validation.
            validation_indices_file: indices to use for validation. Defaults to None, a.k.a.,
                use validation split proportion, if not provided uses half of the data for validation.
            stratified_batch_file: stratified batch file for sampling. Defaults to None, a.k.a.,
                no stratified sampling.
            stratified_value_name: stratified value name. Defaults to None, a.k.a.,
                no stratified sampling. Needed in case a stratified batch file is provided.
            batch_size: batch size. Defaults to 64.
            num_workers: number of workers. Defaults to 1.
        """
        super().__init__()
        self.model: nn.Module
        self.sampling_model: nn.Module
        self.mb_size: int
        self.env: GraphBuildingEnv
        self.ctx: GraphBuildingEnvContext
        self.task: GFlowNetTask

        self.batch_size = batch_size
        self.validation_split = validation_split
        self.validation_indices_file = validation_indices_file
        self.dataset_list = dataset_list
        self.num_workers = num_workers
        self.stratified_batch_file = stratified_batch_file
        self.stratified_value_name = stratified_value_name
        self.prepare_train_data()

    @staticmethod  # TODO: fix this
    def combine_datasets(
        dataset_list: List[GFlowNetDataset],
    ) -> GFlowNetDataset:
        """Combine granular datasets.

        Args:
            dataset_list: a list of granular datasets.

        Returns:
            a combined granular dataset.
        """
        return GFlowNetDataset([a_dataset.dataset for a_dataset in dataset_list])

    def prepare_train_data(self) -> None:
        """Prepare training dataset."""
        self.train_dataset = GFlowNetDataModule.combine_datasets(self.dataset_list)

    def prepare_test_data(self, dataset_list: List[GFlowNetDataset]) -> None:
        """Prepare testing dataset.

        Args:
            dataset_list: a list of granular datasets.
        """
        self.test_dataset = GFlowNetDataModule.combine_datasets(dataset_list)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup the data module.

        Args:
            stage: stage considered, unused. Defaults to None.
        """
        if (
            self.stratified_batch_file is not None
            and self.stratified_value_name is None
        ):
            raise ValueError(
                f"stratified_batch_file={self.stratified_batch_file}, need to set stratified_value_name as well"
            )
        if self.validation_indices_file is None and self.validation_split is None:
            self.validation_split = 0.5
        if self.validation_indices_file:
            val_indices = (
                pd.read_csv(self.validation_indices_file).values.flatten().tolist()
            )
            train_indices = [
                i for i in range(len(self.train_dataset)) if i not in val_indices
            ]
            self.train_data = Subset(self.train_dataset, train_indices)
            self.val_data = Subset(self.train_dataset, val_indices)

        else:
            val = int(len(self.train_dataset) * cast(float, (self.validation_split)))
            train = len(self.train_dataset) - val
            self.train_data, self.val_data = random_split(
                self.train_dataset, [train, val]
            )
        logger.info(f"number of data points used for training: {len(self.train_data)}")
        logger.info(f"number of data points used for validation: {len(self.val_data)}")
        logger.info(
            f"validation proportion: {len(self.val_data) / (len(self.val_data) + len(self.train_data))}"
        )

    @staticmethod
    def get_stratified_batch_sampler(
        stratified_batch_file: str,
        stratified_value_name: str,
        batch_size: int,
        selector_fn: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> StratifiedSampler:
        """Get stratified batch sampler.

        Args:
            stratified_batch_file: stratified batch file for sampling.
            stratified_value_name: stratified value name.
            batch_size: batch size.
            selector_fn: selector function for stratified sampling.
        Returns:
            a stratified batch sampler.
        """
        stratified_batch_dataframe = pd.read_csv(stratified_batch_file)
        stratified_data = stratified_batch_dataframe[
            selector_fn(stratified_batch_dataframe)
        ][stratified_value_name].values
        stratified_data_tensor = torch.from_numpy(stratified_data)
        return StratifiedSampler(targets=stratified_data_tensor, batch_size=batch_size)

    def train_dataloader(self) -> DataLoader:
        """Get a training data loader.

        Returns:
            a training data loader.
        """
        sampler: Optional[Sampler] = None
        if self.stratified_batch_file:
            sampler = GFlowNetDataModule.get_stratified_batch_sampler(
                stratified_batch_file=self.stratified_batch_file,
                stratified_value_name=str(self.stratified_value_name),
                batch_size=self.batch_size,
                selector_fn=lambda dataframe: ~dataframe["validation"],
            )
        return DataLoader(
            self.train_data,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        """Get a validation data loader.

        Returns:
            a validation data loader.
        """
        sampler: Optional[Sampler] = None
        if self.stratified_batch_file:
            sampler = GFlowNetDataModule.get_stratified_batch_sampler(
                stratified_batch_file=self.stratified_batch_file,
                stratified_value_name=str(self.stratified_value_name),
                batch_size=self.batch_size,
                selector_fn=lambda dataframe: dataframe["validation"],
            )
        return DataLoader(
            self.val_data,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            sampler=sampler,
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
