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
"""Model module."""

import ast
import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as _sentencepiece
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_geometric.data as gd

from ..dataloader.dataset import GFlowNetDataset, GFlowNetTask
from ..envs.graph_building_env import GraphBuildingEnv, GraphBuildingEnvContext

# sentencepiece has to be loaded before lightning to avoid segfaults
_sentencepiece

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GFlowNetAlgorithm:
    """We consider the algorithm (objective structure) as part of the model."""

    def compute_batch_losses(
        self, model: nn.Module, batch: gd.Batch, num_bootstrap: Optional[int] = 0
    ) -> Tuple[float, Dict[str, float]]:
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


class GFlowNetModule(pl.LightningModule):
    """Module from generative flow networks."""

    def __init__(
        self,
        configuration: Dict[str, Any],
        dataset: GFlowNetDataset,
        environment: GraphBuildingEnv,
        context: GraphBuildingEnvContext,
        task: GFlowNetTask,
        algorithm: GFlowNetAlgorithm,
        model: nn.Module,
    ) -> None:
        """Construct GFNModule.

        Args:
            dataset: the dataset to use.
            environment: the environment to use.
            context: the context to use.
            task: the task to solve.
            model: architecture (graph_transformer or mxmnet).
            algorithm: algorithm (trajectory_balance or td_loss).
            lr: learning rate for Adam optimizer. Defaults to 1e-4.
            test_output_path: path where to save latent encodings and predictions for the test set
                when an epoch ends. Defaults to a a folder called "test" in the current working directory.
        """

        super().__init__()
        self.hps = configuration
        # self.save_hyperparameters()

        self.env = environment
        self.ctx = context
        self.dataset = dataset

        self.model = model
        self.algo = algorithm

        self.task = task

        self.lr = self.hps["lr"]
        self.test_output_path = self.hps["test_output_path"]

        self.rng = self.hps["rng"]
        self.mb_size = self.hps["global_batch_size"]
        self.clip_grad_param = self.hps["clip_grad_param"]
        self.sampling_tau = self.hps["sampling_tau"]

    def training_step(
        self, batch: gd.Batch, optimizer_idx: int, batch_idx: int
    ) -> Dict[str, Any]:
        """Training step implementation.

        Args:
            batch: batch representation.
            epoch_idx: epoch index.
            batch_idx: batch index.

        Returns:
            loss and logs.
        """
        logs = dict()
        loss, info = self.algo.compute_batch_losses(
            self.model, batch, num_bootstrap=self.mb_size
        )
        logs.update(
            {
                self.model.name + f"/{k}": v if hasattr(v, "item") else v  # type: ignore
                for k, v in info.items()
            }
        )
        logs.update({"total_loss": loss.item()})  # type: ignore

        # logs for step
        _logs = {f"train/{k}": v for k, v in logs.items()}
        self.log_dict(_logs, on_step=True, on_epoch=True, prog_bar=True)

        # logs per epoch
        logs_epoch = {f"train_epoch/{k}": v for k, v in logs.items()}
        logs_epoch["step"] = self.current_epoch
        self.log_dict(logs_epoch, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "logs": logs}

    def training_step_end(self, batch_parts):
        for i in self.model.parameters():
            self.clip_grad_callback(i)

        if self.sampling_tau > 0:
            for a, b in zip(self.model.parameters(), self.sampling_model.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))

    def validation_step(self, batch: gd.Batch, batch_idx: int) -> Dict[str, Any]:
        """Validation step implementation.

        Args:
            batch: batch representation.

        Returns:
            loss and logs.
        """
        loss = 0.0
        logs = dict()
        loss, info = self.algo.compute_batch_losses(
            self.model, batch, num_bootstrap=batch.num_offline
        )
        logs.update({k: v if hasattr(v, "item") else v for k, v in info.items()})
        logs.update({"total_loss": loss})

        self.log_dict(
            {f"val/{k}": v for k, v in logs.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        return {"loss": loss, "logs": logs}

    def test_step(  # type:ignore
        self, batch: Any, batch_idx: int, *args, **kwargs
    ) -> Dict[str, Any]:
        """Testing step implementation.

        Args:
            batch: batch representation.
            batch_idx: batch index, unused.

        Returns:
            loss, logs, and latent encodings.
        """
        loss = 0.0
        logs = dict()
        loss, info = self.algo.compute_batch_losses(
            self.model, batch, num_bootstrap=batch.num_offline
        )
        logs.update({k: v if hasattr(v, "item") else v for k, v in info.items()})
        logs.update({"total_loss": loss})

        self.log_dict(
            {f"test/{k}": v for k, v in logs.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "logs": logs}

    def prediction_step(self, batch):
        """Inference step implementation."""
        pass

    def train_epoch_end(self, outputs: List[Dict[str, Any]]):
        pass

    # change the following to new implementation
    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:  # type:ignore
        """Callback called at the end of an epoch on test outputs.

        Dump encodings and targets for the test set.

        Args:
            outputs: outputs for test batches.
        """
        z = {}
        targets = {}
        z_keys = [key for key in outputs[0]["z"]]
        targets_keys = [key for key in outputs[0]["targets"]]

        for key in z_keys:
            z[key] = (
                torch.cat(
                    [torch.squeeze(an_output["z"][key]) for an_output in outputs], dim=0
                )
                .detach()
                .cpu()
                .numpy()
            )

        for key in targets_keys:
            targets[key] = (
                torch.cat(
                    [torch.squeeze(an_output["targets"][key]) for an_output in outputs],
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )

        pd.to_pickle(z, f"{self.test_output_path}{os.path.sep}z_build.pkl")
        pd.to_pickle(targets, f"{self.test_output_path}{os.path.sep}targets.pkl")

    def configure_optimizers(self):
        """Configure optimizers.

        Returns:
            an optimizer, currently only Adam is supported.
        """
        # Separate Z parameters from non-Z to allow for LR decay on the former
        Z_params = list(self.model.logZ.parameters())  # type: ignore
        non_Z_params = [
            i for i in self.model.parameters() if all(id(i) != id(j) for j in Z_params)
        ]

        self.opt = torch.optim.Adam(
            non_Z_params,
            self.hps["learning_rate"],
            (self.hps["momentum"], 0.999),
            weight_decay=self.hps["weight_decay"],
            eps=self.hps["adam_eps"],
        )
        self.opt_Z = torch.optim.Adam(Z_params, self.hps["learning_rate"], (0.9, 0.999))

        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda steps: 2 ** (-steps / self.hps["lr_decay"])
        )
        self.lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(
            self.opt_Z, lambda steps: 2 ** (-steps / self.hps["Z_lr_decay"])
        )

        if self.sampling_tau > 0:
            self.sampling_model = copy.deepcopy(self.model)
        else:
            self.sampling_model = self.model

        eps = self.hps["tb_epsilon"]
        self.hps["tb_epsilon"] = ast.literal_eval(eps) if isinstance(eps, str) else eps

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
        }[self.hps["clip_grad_type"]]

        return [self.opt, self.opt_Z], [self.lr_sched, self.lr_sched_Z]
