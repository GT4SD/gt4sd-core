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
from argparse import Namespace
from typing import Any, Dict

import pytorch_lightning as pl

from examples.gflownet.dateset_qm9 import QM9Dataset
from examples.gflownet.task_qm9 import QM9GapTask
from gt4sd.frameworks.gflownet.envs.graph_building_env import GraphBuildingEnv
from gt4sd.frameworks.gflownet.envs.mol_building_env import MolBuildingEnvContext

from ..arg_parser.parser import parse_arguments_from_config
from ..dataloader.data_module import GFlowNetDataModule
from ..loss import ALGORITHM_FACTORY
from ..ml.models import MODEL_FACTORY
from ..ml.module import GFlowNetModule


def test_gfn(configuration):
    """test basic GFN training on QM9."""

    dataset = QM9Dataset(configuration["dataset_path"], train=True, target="gap")
    environment = GraphBuildingEnv()
    context = MolBuildingEnvContext(["H", "C", "N", "F", "O"], num_cond_dim=32)

    algorithm = ALGORITHM_FACTORY[configuration["algorithm"]](
        environment,
        context,
        configuration,
    )
    model = MODEL_FACTORY[configuration["model"]](
        context,
        num_emb=configuration["num_emb"],
        num_layers=configuration["num_layers"],
    )

    task = QM9GapTask(
        dataset,
        configuration["temperature_sample_dist"],
        ast.literal_eval(configuration["temperature_dist_params"]),
        device=configuration["device"],
    )

    dm = GFlowNetDataModule(
        configuration=configuration,
        dataset=dataset,
        environment=environment,
        context=context,
        task=task,
        algorithm=algorithm,
        model=model,
        sampling_model=configuration["sampling_model"],
        sampling_iterator=configuration["sampling_iterator"],
    )
    dm.prepare_data()

    module = GFlowNetModule(
        configuration=configuration,
        dataset=dataset,
        environment=environment,
        context=context,
        task=task,
        algorithm=algorithm,
        model=model,
    )

    trainer = pl.Trainer.from_argparse_args(
        Namespace(**configuration),
        profiler="simple",
        auto_lr_find=True,
        log_every_n_steps=50,
        max_epochs=10,
        flush_logs_every_n_steps=100,
        fast_dev_run=True,
    )
    trainer.fit(module, dm)


if __name__ == "__main__":

    def default_hps() -> Dict[str, Any]:
        return {
            "bootstrap_own_reward": False,
            "learning_rate": 1e-4,
            "global_batch_size": 16,
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
            "max_nodes": 9,
            "num_offline": 10,
            "sampling_iterator": True,
            "ratio": 0.9,
            "dev": False,
        }

    hps = {"dataset": "qm9", "dataset_path": "/GFN/qm9.h5", "device": "cpu"}
    # add default configuration
    hps.update(default_hps())
    # add user configuration
    hps.update(vars(parse_arguments_from_config()))

    test_gfn(hps)
