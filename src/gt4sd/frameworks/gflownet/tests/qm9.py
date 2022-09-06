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
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol

from gt4sd.frameworks.gflownet.dataloader.dataset import (
    FlatRewards,
    GFlowNetDataset,
    GFlowNetTask,
    RewardScalar,
)
from gt4sd.frameworks.gflownet.ml.models.mxmnet import (
    HAR2EV,
    MXMNet,
    MXMNetConfig,
    mol2graph,
)

PROPERTIES: List[str] = [
    "rA",
    "rB",
    "rC",
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
]


class QM9Dataset(GFlowNetDataset):
    """QM9 dataset compatible with gflownet."""

    def __init__(
        self,
        h5_file: str,
        target: str = "gap",
        properties: List[str] = PROPERTIES,
    ) -> None:
        """Initialize QM9 dataset.

        Args:
            h5_file: path to the h5 file containing the dataset.
            target: target property to optimize and build the reward.
            properties: list of properties to use as features.
        """
        super().__init__(
            h5_file=h5_file,
            target=target,
            properties=properties,
        )


def thermometer(
    v: torch.Tensor, n_bins: int = 50, vmin: int = 0, vmax: int = 1
) -> torch.Tensor:
    """Compute a thermometer reward using gap.

    Args:
        v: tensor of values to compute the reward.
        n_bins: number of bins to use.
        vmin: minimum value of the range.
        vmax: maximum value of the range.

    Returns:
        tensor of the reward.
    """
    bins = torch.linspace(vmin, vmax, n_bins)
    gap = bins[1] - bins[0]
    return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(
        0, gap.item()
    ) / gap


# define task
class QM9GapTask(GFlowNetTask):
    """QM9 task compatible with gflownet."""

    def __init__(
        self,
        configuration: Dict[str, Any],
        dataset: GFlowNetDataset,
        reward_model: nn.Module = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        """Initialize QM9 task.

        Code adapted from: https://github.com/recursionpharma/gflownet/blob/trunk/src/gflownet/tasks/qm9/qm9.py.

        Args:
            configuration: configuration of the task.
            dataset: dataset to use for the task.
            reward_model: model to use for the reward.
            wrap_model: function to wrap the model.
        """
        super().__init__(
            configuration=configuration,
            dataset=dataset,
            reward_model=reward_model,
            wrap_model=wrap_model,
        )

    def flat_reward_transform(self, _y: Union[float, torch.Tensor]) -> FlatRewards:
        """Transforms a target quantity y (e.g. the LUMO energy in QM9) to a positive reward scalar.

        Args:
            _y: target quantity to transform.

        Returns:
            reward scalar.
        """
        y = np.array(_y)
        y = y.astype("float")
        if self._rtrans == "exp":
            flat_r = np.exp(-(y - self._min) / self._width)
        elif self._rtrans == "unit":
            flat_r = 1 - (y - self._min) / self._width
        elif self._rtrans == "unit+95p":
            # Add constant such that 5% of rewards are > 1
            flat_r = 1 - (y - self._percentile_95) / self._width
        else:
            raise ValueError(self._rtrans)
        return FlatRewards(flat_r)

    def inverse_flat_reward_transform(self, rp):
        """Inverse transform a reward scalar to a target quantity y (e.g. the LUMO energy in QM9).

        Args:
            rp: reward scalar to transform.

        Returns:
            target quantity.
        """
        if self._rtrans == "exp":
            return -np.log(rp) * self._width + self._min
        elif self._rtrans == "unit":
            return (1 - rp) * self._width + self._min
        elif self._rtrans == "unit+95p":
            return (1 - rp + (1 - self._percentile_95)) * self._width + self._min

    def load_task_models(self) -> Dict[str, nn.Module]:
        """Loads the models for the task.

        Returns:
            dictionary of models.
        """
        gap_model = MXMNet(MXMNetConfig(128, 6, 5.0))
        try:
            state_dict = torch.load("/ckpt/mxmnet_gap_model.pt")
            gap_model.load_state_dict(state_dict)
        except FileNotFoundError:
            pass
        gap_model.to(self.device)
        # gap_model = self._wrap_model(gap_model)
        return {"model_task": gap_model}

    def sample_conditional_information(self, n):
        """Sample conditional information for the task.

        Args:
            n: number of samples to sample.

        Returns:
            dictionary of conditional information.
        """
        beta = None
        if self.temperature_sample_dist == "gamma":
            beta = self.rng.gamma(*self.temperature_dist_params, n).astype(np.float32)
        elif self.temperature_sample_dist == "uniform":
            beta = self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
        elif self.temperature_sample_dist == "beta":
            beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
        beta_enc = thermometer(torch.tensor(beta), 32, 0, 32)
        return {"beta": torch.tensor(beta), "encoding": beta_enc}

    def cond_info_to_reward(
        self, cond_info: Dict[str, torch.Tensor], _flat_reward: FlatRewards
    ) -> RewardScalar:

        """Compute the reward for a given conditional information.

        Args:
            cond_info: dictionary of conditional information.
            _flat_reward: flat reward.
        Returns:
            reward scalar.
        """
        if isinstance(_flat_reward, list):
            flat_reward = torch.tensor(_flat_reward)
        return RewardScalar(flat_reward ** cond_info["beta"])

    def compute_flat_rewards(
        self, mols: List[RDMol]
    ) -> Tuple[RewardScalar, torch.Tensor]:
        """Computes the flat rewards for a list of molecules.

        Args:
            mols: list of molecules.

        Returns:
            reward scalar and validity.
        """
        graphs = [mol2graph(i) for i in mols]  # type: ignore[attr-defined]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return RewardScalar(torch.zeros((0,))), is_valid

        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.device)

        # sample model
        preds = self.model["model_task"](batch)
        preds = preds.reshape((-1,)).data.cpu() / HAR2EV  # type: ignore[attr-defined]
        preds[preds.isnan()] = 1
        preds = self.flat_reward_transform(preds).clip(1e-4, 2)

        return RewardScalar(preds), is_valid
