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
from typing import Dict, List, NewType, Optional, Tensor, Tuple

import sentencepiece as _sentencepiece
import numpy as np
import pytorch_lightning as pl
import torch
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol
from torch.utils.data import DataLoader  # , Subset, random_split

from gt4sd.frameworks.gflownet.data.dataset import GFlowNetDataset
from gt4sd.frameworks.gflownet.data.sampling_iterator import SamplingIterator
from gt4sd.frameworks.gflownet.envs.graph_building_env import (
    GraphActionCategorical,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
)
from gt4sd.frameworks.gflownet.loss import ALGORITHM_FACTORY
from gt4sd.frameworks.gflownet.model import MODEL_FACTORY
from gt4sd.frameworks.gflownet.util import wrap_model_mp

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
    """Data module from gflownet."""

    def __init__(
        self,
        dataset: GFlowNetDataset,
        env: GraphBuildingEnv,
        ctx: GraphBuildingEnvContext,
        task: GFlowNetTask,
        algo: str,
        model: str,
        sampling_model: str,
        validation_split: Optional[float] = None,
        validation_indices_file: Optional[str] = None,
        stratified_batch_file: Optional[str] = None,
        stratified_value_name: Optional[str] = None,
        sampling_iterator: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 1,
        device: str = "cuda",
        seed: int = 142857,
        ratio: float = 0.9,
    ) -> None:
        """Construct GFlowNetDataModule.

        Args:
            dataset: dataset.
            env: environment for graph building.
            ctx: context env.
            task: generic task.
            algo: loss function.
            model: model type.
            sampling_model:
            validation_split: proportion used for validation. Defaults to None,
                a.k.a., use indices file if provided otherwise uses half of the data for validation.
            validation_indices_file: indices to use for validation. Defaults to None, a.k.a.,
                use validation split proportion, if not provided uses half of the data for validation.
            stratified_batch_file: stratified batch file for sampling. Defaults to None, a.k.a.,
                no stratified sampling.
            stratified_value_name: stratified value name. Defaults to None, a.k.a.,
                no stratified sampling. Needed in case a stratified batch file is provided.
            sampling_iterator: sampling iterator to use. Defaults to None.
            batch_size: batch size.
            num_workers: number of workers.
            device: device.
        """
        super().__init__()
        self.model = MODEL_FACTORY[model]
        self.sampling_model = MODEL_FACTORY[sampling_model]
        self.algo = ALGORITHM_FACTORY[algo]
        self.env = env
        self.ctx = ctx
        self.task = task
        self.dataset = dataset

        self.mb_size: int
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.validation_indices_file = validation_indices_file
        self.num_workers = num_workers
        self.device = device
        self.stratified_batch_file = stratified_batch_file
        self.stratified_value_name = stratified_value_name
        self.sampling_iterator = sampling_iterator

        rng = np.random.default_rng(seed)
        ll = self.dataset.get_len_df()
        ixs = np.arange(ll)
        rng.shuffle(ixs)
        # TODO: use Subset
        self.ix_train = ixs[: int(np.floor(ratio * ll))]
        self.ix_test = ixs[int(np.floor(ratio * ll)) :]

        self.prepare_train_data()

    def prepare_train_data(self, dataset: GFlowNetDataset) -> None:
        """Prepare training dataset."""
        self.train_dataset = dataset
        self.train_dataset.set_indexes(self.ixs_train)

    def prepare_test_data(self, dataset: GFlowNetDataset) -> None:
        """Prepare testing dataset."""
        self.test_dataset = dataset
        self.test_dataset.set_indexes(self.ixs_test)

    # def setup(self, stage: Optional[str] = None) -> None:
    #     """Setup the data module.

    #     Args:
    #         stage: stage considered, unused. Defaults to None.
    #     """

    #     if self.validation_indices_file is None and self.validation_split is None:
    #         self.validation_split = 0.5
    #     if self.validation_indices_file:
    #         val_indices = (
    #             pd.read_csv(self.validation_indices_file).values.flatten().tolist()
    #         )
    #         train_indices = [
    #             i for i in range(len(self.train_dataset)) if i not in val_indices
    #         ]
    #         self.train_data = Subset(self.train_dataset, train_indices)
    #         self.val_data = Subset(self.train_dataset, val_indices)

    #     else:
    #         val = int(len(self.train_dataset) * cast(float, (self.validation_split)))
    #         train = len(self.train_dataset) - val
    #         self.train_data, self.val_data = random_split(
    #             self.train_dataset, [train, val]
    #         )
    #     logger.info(f"number of data points used for training: {len(self.train_data)}")
    #     logger.info(f"number of data points used for validation: {len(self.val_data)}")
    #     logger.info(
    #         f"validation proportion: {len(self.val_data) / (len(self.val_data) + len(self.train_data))}"
    #     )

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

    def _wrap_model_mp(self, model):
        """Wraps a nn.Module instance so that it can be shared to `DataLoader` workers."""
        if self.num_workers > 0:
            placeholder = wrap_model_mp(
                model, self.num_workers, cast_types=(gd.Batch, GraphActionCategorical)
            )
            return placeholder, torch.device("cpu")
        return model, self.device

    def train_dataloader(self) -> DataLoader:
        """Get a training data loader.

        Returns:
            a training data loader.
        """
        if self.sampling_iterator:
            model, dev = self._wrap_model_mp(self.model)
            iterator = SamplingIterator(
                self.train_dataset,
                model,
                self.mb_size * 2,
                self.ctx,
                self.algo,
                self.task,
                dev,
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
            model, dev = self._wrap_model_mp(self.model)
            iterator = SamplingIterator(
                self.test_dataset,
                model,
                self.mb_size,
                self.ctx,
                self.algo,
                self.task,
                dev,
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
