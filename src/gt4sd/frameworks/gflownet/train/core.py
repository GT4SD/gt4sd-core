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
import pathlib
from typing import Any, Dict, List, NewType, Optional, Tuple

import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from gt4sd.frameworks.gflownet.data.sampling_iterator import SamplingIterator
from gt4sd.frameworks.gflownet.envs.graph_building_env import (
    GraphActionCategorical,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
)
from gt4sd.frameworks.gflownet.util import wrap_model_mp

# This type represents an unprocessed list of reward signals/conditioning information
FlatRewards = NewType("FlatRewards", Tensor)  # type: ignore

# This type represents the outcome for a multi-objective task of
# converting FlatRewards to a scalar, e.g. (sum R_i omega_i) ** beta
RewardScalar = NewType("RewardScalar", Tensor)  # type: ignore


class GFNAlgorithm:
    def compute_batch_losses(
        self, model: nn.Module, batch: gd.Batch, num_bootstrap: Optional[int] = 0
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Computes the loss for a batch of data, and proves logging informations.

        Args:
            model: the model being trained or evaluated.
            batch: a batch of graphs.
            num_bootstrap: the number of trajectories with reward targets in the batch (if applicable).

        Returns:
            loss: the loss for that batch.
            info: logged information about model predictions.
        """
        raise NotImplementedError()


class GFNTask:
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
        """Compute the flat rewards of mols according the the tasks' proxies

        Parameters
        ----------
        mols: List[RDMol]
            A list of RDKit molecules.
        Returns
        -------
        reward: RewardScalar
            A 1d tensor, a scalar reward for each molecule.
        is_valid: Tensor
            A 1d tensor, a boolean indicating whether the molecule is valid.
        """
        raise NotImplementedError()


class GFNTrainer:
    def __init__(self, hps: Dict[str, Any], device: torch.device):
        # self.setup should at least set these up:
        self.training_data: Dataset
        self.test_data: Dataset
        self.model: nn.Module
        self.sampling_model: nn.Module
        self.mb_size: int
        self.env: GraphBuildingEnv
        self.ctx: GraphBuildingEnvContext
        self.task: GFNTask
        self.algo: GFNAlgorithm
        self.device: str = device

        self.hps = {**self.default_hps(), **hps}
        self.num_workers: int = self.hps.get("num_data_loader_workers", 0)
        self.setup()

    def default_hps(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def setup(self):
        raise NotImplementedError()

    def step(self, loss: Tensor):
        raise NotImplementedError()

    def _wrap_model_mp(self, model):
        """Wraps a nn.Module instance so that it can be shared to `DataLoader` workers."""
        if self.num_workers > 0:
            placeholder = wrap_model_mp(
                model, self.num_workers, cast_types=(gd.Batch, GraphActionCategorical)
            )
            return placeholder, torch.device("cpu")
        return model, self.device

    def build_training_data_loader(self) -> DataLoader:
        model, dev = self._wrap_model_mp(self.sampling_model)
        iterator = SamplingIterator(
            self.training_data,
            model,
            self.mb_size * 2,
            self.ctx,
            self.algo,
            self.task,
            dev,
        )
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def build_validation_data_loader(self) -> DataLoader:
        model, dev = self._wrap_model_mp(self.model)
        iterator = SamplingIterator(
            self.test_data,
            model,
            self.mb_size,
            self.ctx,
            self.algo,
            self.task,
            dev,
            ratio=1,
            stream=False,
        )
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_batch(
        self, batch: gd.Batch, epoch_idx: int, batch_idx: int
    ) -> Dict[str, Any]:
        loss, info = self.algo.compute_batch_losses(
            self.model, batch, num_bootstrap=self.mb_size
        )
        self.step(loss)
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def evaluate_batch(
        self, batch: gd.Batch, epoch_idx: int = 0, batch_idx: int = 0
    ) -> Dict[str, Any]:
        loss, info = self.algo.compute_batch_losses(
            self.model, batch, num_bootstrap=batch.num_offline
        )
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def run(self):
        """Trains the GFN for num_training_steps minibatches, performing
        validation every validate_every minibatches.
        """

        self.model.to(self.device)
        self.sampling_model.to(self.device)

        epoch_length = len(self.training_data)

        train_dl = self.build_training_data_loader()
        valid_dl = self.build_validation_data_loader()

        for it, batch in zip(range(1, 1 + self.hps["num_training_steps"]), train_dl):
            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            batch = batch.to(self.device)
            info = self.train_batch(batch, epoch_idx, batch_idx)
            self.log(info, it, "train")

            if it % self.hps["validate_every"] == 0:
                for batch in valid_dl:
                    batch = batch.to(self.device)
                    info = self.evaluate_batch(batch, epoch_idx, batch_idx)
                    self.log(info, it, "valid")

                torch.save(
                    {
                        "models_state_dict": [self.model.state_dict()],
                        "hps": self.hps,
                    },
                    open(pathlib.Path(self.hps["log_dir"]) / "model_state.pt", "wb"),
                )

    def log(self, info, index, key):
        if not hasattr(self, "_summary_writer"):
            self._summary_writer = SummaryWriter(self.hps["log_dir"])
        for k, v in info.items():
            self._summary_writer.add_scalar(f"{key}_{k}", v, index)
