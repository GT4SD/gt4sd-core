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
import ast
import copy
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit import RDLogger
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset

import gt4sd.frameworks.gflownet.model.mxmnet as mxmnet
from gt4sd.frameworks.gflownet.data.qm9 import QM9Dataset
from gt4sd.frameworks.gflownet.envs.graph_building_env import GraphBuildingEnv
from gt4sd.frameworks.gflownet.envs.mol_building_env import MolBuildingEnvContext
from gt4sd.frameworks.gflownet.loss.trajectory_balance import TrajectoryBalance
from gt4sd.frameworks.gflownet.model.graph_transformer import GraphTransformerGFN
from gt4sd.frameworks.gflownet.train.train_gfn import (
    FlatRewards,
    GFlowNetTask,
    GFlowNetTrainer,
    RewardScalar,
)


def thermometer(v: Tensor, n_bins=50, vmin=0, vmax=1) -> Tensor:
    bins = torch.linspace(vmin, vmax, n_bins)
    gap = bins[1] - bins[0]
    return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(
        0, gap.item()
    ) / gap


# TODO: training_pipeline for tasks (QM9))
class QM9GapTask(GFlowNetTask):
    def __init__(
        self,
        dataset: Dataset,
        temperature_distribution: str,
        temperature_parameters: Tuple[float],
        wrap_model: Callable[[nn.Module], nn.Module] = None,
        device: str = "cuda",
    ):
        """This class captures conditional information generation and reward transforms.

        Args:
            dataset:
            temperature_distribution:
            temperature_parameters:
            wrap_model:
            device:
        """
        self._wrap_model = wrap_model
        self.device = device
        self.models = self.load_task_models()
        self.dataset = dataset
        self.temperature_sample_dist = temperature_distribution
        self.temperature_dist_params = temperature_parameters
        # TODO: fix interface
        self._min, self._max, self._percentile_95 = self.dataset.get_stats(percentile=0.05)  # type: ignore
        self._width = self._max - self._min
        self._rtrans = "unit+95p"  # TODO: hyperparameter

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

    def load_task_models(self):
        gap_model = mxmnet.MXMNet(mxmnet.Config(128, 6, 5.0))
        # TODO: this path should be part of the config?
        try:
            state_dict = torch.load("/data/chem/qm9/mxmnet_gap_model.pt")
            gap_model.load_state_dict(state_dict)
        except FileNotFoundError:
            pass
        gap_model.to(self.device)
        gap_model, self.device = self._wrap_model(gap_model)
        return {"mxmnet_gap": gap_model}

    def sample_conditional_information(self, n):
        beta = None
        if self.temperature_sample_dist == "gamma":
            beta = self.rng.gamma(*self.temperature_dist_params, n).astype(np.float32)
        elif self.temperature_sample_dist == "uniform":
            beta = self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
        elif self.temperature_sample_dist == "beta":
            beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
        beta_enc = thermometer(torch.tensor(beta), 32, 0, 32)  # TODO: hyperparameters
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
        preds = self.models["mxmnet_gap"](batch).reshape((-1,)).data.cpu() / mxmnet.HAR2EV  # type: ignore[attr-defined]
        preds[preds.isnan()] = 1
        preds = self.flat_reward_transform(preds).clip(1e-4, 2)
        return RewardScalar(preds), is_valid


class QM9GapTrainer(GFlowNetTrainer):
    def default_hps(self) -> Dict[str, Any]:
        return {
            "bootstrap_own_reward": False,
            "learning_rate": 1e-4,
            "global_batch_size": 64,
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
        }

    def setup(self):
        hps = self.hps
        RDLogger.DisableLog("rdApp.*")
        self.rng = np.random.default_rng(142857)
        self.env = GraphBuildingEnv()
        # TODO: support to infer environment objects from dataset
        self.ctx = MolBuildingEnvContext(["H", "C", "N", "F", "O"], num_cond_dim=32)

        self.training_data = QM9Dataset(hps["qm9_h5_path"], train=True, target="gap")
        self.test_data = QM9Dataset(hps["qm9_h5_path"], train=False, target="gap")

        model = GraphTransformerGFN(
            self.ctx, num_emb=hps["num_emb"], num_layers=hps["num_layers"]
        )
        self.model = model

        # Separate Z parameters from non-Z to allow for LR decay on the former
        Z_params = list(model.logZ.parameters())
        non_Z_params = [
            i for i in self.model.parameters() if all(id(i) != id(j) for j in Z_params)
        ]

        self.opt = torch.optim.Adam(
            non_Z_params,
            hps["learning_rate"],
            (hps["momentum"], 0.999),
            weight_decay=hps["weight_decay"],
            eps=hps["adam_eps"],
        )
        self.opt_Z = torch.optim.Adam(Z_params, hps["learning_rate"], (0.9, 0.999))

        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda steps: 2 ** (-steps / hps["lr_decay"])
        )
        self.lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(
            self.opt_Z, lambda steps: 2 ** (-steps / hps["Z_lr_decay"])
        )

        self.sampling_tau = hps["sampling_tau"]
        if self.sampling_tau > 0:
            self.sampling_model = copy.deepcopy(model)
        else:
            self.sampling_model = self.model
        eps = hps["tb_epsilon"]
        hps["tb_epsilon"] = ast.literal_eval(eps) if isinstance(eps, str) else eps

        self.algo = TrajectoryBalance(self.env, self.ctx, self.rng, hps, max_nodes=9)

        self.task = QM9GapTask(
            self.training_data,
            hps["temperature_sample_dist"],
            ast.literal_eval(hps["temperature_dist_params"]),
            wrap_model=self._wrap_model_mp,
            device=self.device,
        )

        self.mb_size = hps["global_batch_size"]
        self.clip_grad_param = hps["clip_grad_param"]
        self.clip_grad_callback = {
            "value": (
                lambda params: torch.nn.utils.clip_grad_value_(
                    params, self.clip_grad_param
                )
            ),
            "norm": (
                lambda params: torch.nn.utils.clip_grad_norm_(
                    params, self.clip_grad_param
                )
            ),
            "none": (lambda x: None),
        }[hps["clip_grad_type"]]

    def step(self, loss: Tensor):
        # check nans, check loss smaller than threshold
        loss.backward()
        for i in self.model.parameters():
            self.clip_grad_callback(i)

        self.opt.step()
        self.opt.zero_grad()

        self.opt_Z.step()
        self.opt_Z.zero_grad()

        self.lr_sched.step()
        self.lr_sched_Z.step()

        if self.sampling_tau > 0:
            for a, b in zip(self.model.parameters(), self.sampling_model.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))
