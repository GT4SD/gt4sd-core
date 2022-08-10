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

from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset

import gt4sd.frameworks.gflownet.ml.models.mxmnet as mxmnet
from gt4sd.frameworks.gflownet.dataloader.data_module import (
    FlatRewards,
    GFlowNetTask,
    RewardScalar,
)


def thermometer(v: Tensor, n_bins=50, vmin=0, vmax=1) -> Tensor:
    bins = torch.linspace(vmin, vmax, n_bins)
    gap = bins[1] - bins[0]
    return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(
        0, gap.item()
    ) / gap


# define task
class QM9GapTask(GFlowNetTask):
    """Define task for QM9 dataset."""

    def __init__(
        self,
        reward_model: nn.Module,  # this is set to self.model
        dataset: Dataset,
        temperature_distribution: str,
        temperature_parameters: Tuple[float],
        wrap_model: Callable[[nn.Module], nn.Module] = None,
        device: str = "cuda",
    ):
        """This class captures conditional information generation and reward transforms.

        Args:
            reward_model: The model that is used to generate the conditional reward.
            dataset:
            temperature_distribution:
            temperature_parameters:
            wrap_model: a wrapper function that is applied to the model. #TODO: do we need it with lightning?
            device: cpu or cuda
        """
        
        self._wrap_model = wrap_model
        self.device = device
        # fix this
        if reward_model:
            self.models = reward_model
        else:
            self.models = self.load_task_models()
        self.dataset = dataset
        self.temperature_sample_dist = temperature_distribution
        self.temperature_dist_params = temperature_parameters

        self._min, self._max, self._percentile_95 = self.dataset.get_stats(percentile=0.05)  # type: ignore
        self._width = self._max - self._min
        self._rtrans = "unit+95p"

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        """Transforms a target quantity y (e.g. the LUMO energy in QM9) to a positive reward scalar."""
        y = np.array(y)
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
        if self._rtrans == "exp":
            return -np.log(rp) * self._width + self._min
        elif self._rtrans == "unit":
            return (1 - rp) * self._width + self._min
        elif self._rtrans == "unit+95p":
            return (1 - rp + (1 - self._percentile_95)) * self._width + self._min

    def load_task_models(self) -> Dict[str, nn.Module]:
        """Loads the models for the task."""
        gap_model = mxmnet.MXMNet(mxmnet.Config(128, 6, 5.0))
        try:
            state_dict = torch.load("/ckpt/mxmnet_gap_model.pt")
            gap_model.load_state_dict(state_dict)
        except FileNotFoundError:
            pass
        gap_model.to(self.device)
        #gap_model = self._wrap_model(gap_model)
        return {"mxmnet_gap": gap_model}

    def sample_conditional_information(self, n):
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
        self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards
    ) -> RewardScalar:
        if isinstance(flat_reward, list):
            flat_reward = torch.tensor(flat_reward)
        return RewardScalar(flat_reward ** cond_info["beta"])

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[RewardScalar, Tensor]:
        graphs = [mxmnet.mol2graph(i) for i in mols]  # type: ignore[attr-defined]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return RewardScalar(torch.zeros((0,))), is_valid
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.device)

        # sample model
        preds = self.models["mxmnet_gap"](batch).reshape((-1,)).data.cpu() / mxmnet.HAR2EV  # type: ignore[attr-defined]
        preds[preds.isnan()] = 1
        preds = self.flat_reward_transform(preds).clip(1e-4, 2)
        
        return RewardScalar(preds), is_valid
