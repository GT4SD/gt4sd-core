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
"""TorchDrug GraphAF training utilities."""
import ast
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from torchdrug.core import Engine
from torchdrug.layers import distribution
from torchdrug.models import RGCN, GraphAF
from torchdrug.tasks import AutoregressiveGeneration

# isort: off
import torch
from torch import optim
from torch import nn

# isort: on
from ....cli.argument_parser import eval_lambda
from ...core import TrainingPipelineArguments
from .. import DATASET_FACTORY
from ..core import TorchDrugTrainingPipeline

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

"""
Necessary because torchdrug silently overwrites the default nn.Module. This is quite
invasive and causes significant side-effects in the rest of the code.
See: https://github.com/DeepGraphLearning/torchdrug/issues/77
"""
nn.Module = nn._Module  # type: ignore


class TorchDrugGraphAFTrainingPipeline(TorchDrugTrainingPipeline):
    """TorchDrug GraphAF training pipelines."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
    ) -> None:
        """Generic training function for training a
            (GraphAF) model. For details see:
                Shi, Chence, et al. "GraphAF: a Flow-based Autoregressive Model for
                Molecular Graph Generation".
                International Conference on Learning Representations (ICLR), 2020.

        Args:
            training_args: training arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.
        """
        try:

            params = {**training_args, **dataset_args, **model_args}

            model_path = params["model_path"]
            training_name = params["training_name"]
            dataset_name = params["dataset_name"]

            logger.info(f"Model with name {training_name} starts.")

            model_dir = Path(model_path).joinpath(training_name)
            os.makedirs(model_dir, exist_ok=True)

            # Set up the dataset here
            joint_dataset_args = {
                "verbose": params.get("verbose", 1),
                "lazy": params.get("lazy", False),
                "transform": eval_lambda(params.get("transform", "lambda x: x")),
                "node_feature": params.get("node_feature", "default"),
                "edge_feature": params.get("edge_feature", "default"),
                "graph_feature": params.get("graph_feature", None),
                "with_hydrogen": params.get("with_hydrogen", False),
                "kekulize": not params.get("no_kekulization", False),
            }
            if dataset_name not in DATASET_FACTORY.keys():
                raise ValueError(
                    f"Dataset {dataset_name} is not supported. Choose from "
                    f"{DATASET_FACTORY.keys()}"
                )
            if dataset_name != "custom":
                # This is a native TorchDrug dataset
                dataset = DATASET_FACTORY[dataset_name](
                    path=params["dataset_path"], **joint_dataset_args
                )
            else:
                # User brought their own dataset
                dataset = DATASET_FACTORY["custom"](
                    file_path=params["file_path"],
                    target_fields=[params["target_field"]],
                    smiles_field=params.get("smiles_field", "smiles"),
                    **joint_dataset_args,
                )

            hidden_dims = ast.literal_eval(params["hidden_dims"])
            num_atom_type = dataset.num_atom_type
            num_bond_type = dataset.num_bond_type

            model = RGCN(
                input_dim=params.get("input_dim", num_atom_type),
                num_relation=params.get("num_relation", num_bond_type),
                hidden_dims=hidden_dims,
                batch_norm=params.get("batch_norm", True),
                edge_input_dim=params.get("edge_input_dim", None),
                short_cut=params.get("short_cut", False),
                activation=params.get("activation", "relu"),
                concat_hidden=params.get("concat_hidden", False),
                readout=params.get("readout", "sum"),
            )
            task = params.get("task")
            if dataset_name == "custom" and task and params["target_field"] != task:
                raise ValueError(
                    "If custom dataset is used & task is specified, then target_field "
                    "has to be set s.t. it extracts the task/property of interest. "
                    f"Not task={task} and target_field={params['target_field']}"
                )
            criterion = ast.literal_eval(params["criterion"])
            if "ppo" in criterion.keys() and (
                params["no_kekulization"] or params["node_feature"] != "symbol"
            ):
                # See torchdrug issue: https://github.com/DeepGraphLearning/torchdrug/issues/77
                raise ValueError(
                    "For property optimiz. leave `no_kekulization` at the default ("
                    "False) & set `node_feature` to `symbol` and not: "
                    f"{params['no_kekulization']} and `{params['node_feature']}`."
                )

            # Model prior/flow initialization
            node_prior = distribution.IndependentGaussian(
                torch.zeros(num_atom_type), torch.ones(num_atom_type)
            )
            edge_prior = distribution.IndependentGaussian(
                torch.zeros(num_bond_type + 1), torch.ones(num_bond_type + 1)
            )
            node_flow = GraphAF(
                model, node_prior, num_layer=params.get("num_node_flow_layers", 12)
            )
            edge_flow = GraphAF(
                model,
                edge_prior,
                num_layer=params.get("num_edge_flow_layers", 12),
                use_edge=not params.get("no_edge", False),
            )

            task = AutoregressiveGeneration(
                node_flow,
                edge_flow,
                task=params.get("task", None),
                max_edge_unroll=params.get("max_edge_unroll", 12),
                max_node=params.get("max_node", 38),
                criterion=ast.literal_eval(params["criterion"]),
                num_node_sample=params.get("num_node_sample", -1),
                num_edge_sample=params.get("num_edge_sample", -1),
                agent_update_interval=params.get("agent_update_interval", 10),
                gamma=params.get("gamma", 0.9),
                reward_temperature=params.get("reward_temperature", 1.0),
                baseline_momentum=params.get("baseline_momentum", 0.9),
            )

            optimizer = optim.Adam(
                task.parameters(), lr=params.get("learning_rate", 1e-5)
            )
            device = (0,) if torch.cuda.is_available() else None
            solver = Engine(
                task,
                dataset,
                None,  # validation data
                None,  # test data
                optimizer,
                batch_size=params.get("batch_size", 16),
                log_interval=params.get("log_interval", 100),
                scheduler=params.get("scheduler", None),
                gpus=params.get("gpus", device),
                gradient_interval=params.get("gradient_interval", 1),
                num_worker=params.get("num_worker", 0),
            )
            # Necessary since we have re-assigned nn.Module to the native torch.nn.Module
            # rather than the torchdrug-overwritten version.
            solver.model.device = solver.device

            weight_paths = sorted(list(model_dir.glob("*.pkl")), key=os.path.getmtime)
            if len(weight_paths) > 0:
                solver.load(
                    weight_paths[-1], load_optimizer=params.get("load_optimizer", False)
                )
                logger.info(f"Restored existing model from {weight_paths[-1]}")
                logger.info(
                    "To avoid this, set `training_name` & `model_path` to a new folder."
                )

            epochs = params.get("epochs", 10)
            solver.train(num_epoch=epochs)

            # Save model
            task_name = f"task={params.get('task')}_" if params.get("task") else ""
            data_name = "data=" + (
                dataset_name
                + "_"
                + str(params["file_path"]).split(os.sep)[-1].split(".")[0]
                if dataset_name == "custom"
                else dataset_name
            )

            solver.save(
                model_dir.joinpath(
                    f"graphaf_data={data_name}_{task_name}epoch={epochs}.pkl"
                )
            )

        except Exception:
            logger.exception(
                "Exception occurred while running TorchDrugGraphAFTrainingPipeline"
            )


@dataclass
class TorchDrugGraphAFModelArguments(TrainingPipelineArguments):
    """Arguments pertaining to model instantiation."""

    __name__ = "model_args"

    hidden_dims: str = field(
        default="[128, 128]",
        metadata={"help": "Dimensionality of each hidden layer"},
    )
    batch_norm: bool = field(
        default=False, metadata={"help": "Whether the RGCN uses batch normalization"}
    )
    edge_input_dim: Optional[int] = field(
        default=None, metadata={"help": "Dimension of edge features"}
    )
    short_cut: bool = field(
        default=False, metadata={"help": "Whether the RGCN uses a short cut"}
    )
    activation: str = field(
        default="relu", metadata={"help": "Activation function for RGCN"}
    )
    concat_hidden: bool = field(
        default=False,
        metadata={
            "help": "Whether hidden representations from all layers are concatenated"
        },
    )
    num_node_flow_layers: int = field(
        default=12,
        metadata={"help": "Number of layers in the node flow GraphAF model"},
    )
    num_edge_flow_layers: int = field(
        default=12,
        metadata={"help": "Number of layers in the edge flow GraphAF model"},
    )
    no_edge: bool = field(
        default=False,
        metadata={
            "help": "Whether to use edge features in the edge GraphAF model. Per "
            "default, edges are used."
        },
    )

    readout: str = field(
        default="sum",
        metadata={"help": "RGCN Readout function. Either `sum` or `mean`"},
    )
    max_edge_unroll: int = field(
        default=12,
        metadata={
            "help": "max node id difference. Inferred from training data if not provided"
        },
    )
    max_node: int = field(
        default=38,
        metadata={
            "help": "max number of node. Inferred from training data if not provided."
        },
    )
    criterion: str = field(
        default="{'nll': 1.0}",
        metadata={
            "help": "training criterion. Available criteria are `nll` and `ppo` for"
            " regular training and property optimization respectively. If dict, the "
            "keys are criterions and values are the corresponding weights. If list, "
            "both criteria are used with equal weights."
        },
    )
    num_node_sample: int = field(
        default=-1,
        metadata={"help": "Number of node samples per graph."},
    )
    num_edge_sample: int = field(
        default=-1,
        metadata={"help": "Number of edge samples per graph."},
    )
    agent_update_interval: int = field(
        default=10,
        metadata={
            "help": "Update the agent every n batches (similar to gradient accumulation)"
        },
    )
    gamma: float = field(
        default=0.9,
        metadata={"help": "Reward discount rate"},
    )
    reward_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for the reward (larger -> higher mean reward)"
            "lower -> higher maximal reward."
        },
    )
    baseline_momentum: float = field(
        default=0.9,
        metadata={"help": "Momentum for value function baseline"},
    )
