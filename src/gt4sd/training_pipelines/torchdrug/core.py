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
"""TorchDrug training utilities."""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ...configuration import gt4sd_configuration_instance
from ..core import TrainingPipeline, TrainingPipelineArguments
from . import DATASET_FACTORY

DATA_ROOT = os.path.join(
    gt4sd_configuration_instance.gt4sd_local_cache_path, "data", "torchdrug"
)
os.makedirs(DATA_ROOT, exist_ok=True)


class TorchDrugTrainingPipeline(TrainingPipeline):
    """TorchDrug training pipelines."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
    ) -> None:
        """Generic training function for launching a TorchDrug training.

        Args:
            training_args: training arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.

        Raises:
            NotImplementedError: the generic trainer does not implement the pipeline.
        """
        raise NotImplementedError


@dataclass
class TorchDrugTrainingArguments(TrainingPipelineArguments):
    """Arguments related to torchDrug trainer."""

    __name__ = "training_args"

    model_path: str = field(
        metadata={"help": "Path where the model artifacts are stored."}
    )
    training_name: str = field(metadata={"help": "Name used to identify the training."})
    epochs: int = field(default=10, metadata={"help": "Number of epochs."})
    batch_size: int = field(default=16, metadata={"help": "Size of the batch."})
    learning_rate: float = field(
        default=1e-5, metadata={"help": "Learning rate used in training."}
    )
    log_interval: int = field(
        default=100, metadata={"help": "Number of steps between log intervals."}
    )
    gradient_interval: int = field(
        default=1, metadata={"help": "Gradient accumulation steps"}
    )
    num_worker: int = field(
        default=0, metadata={"help": "Number of CPU workers per GPU."}
    )
    task: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optimization task for goal-driven generation."
            "Currently, TorchDrug only supports `plogp` and `qed`."
        },
    )


@dataclass
class TorchDrugDataArguments(TrainingPipelineArguments):
    """Arguments related to TorchDrug data loading."""

    __name__ = "dataset_args"

    dataset_name: str = field(
        metadata={
            "help": f"Identifier for the dataset. Has to be in {DATASET_FACTORY.keys()}"
            ". Can either point to one of the predefined TorchDrug datasets or it can "
            "be `custom` if the user brings their own dataset. If `custom`, then the "
            "arguments `file_path`, `target_field` and `smiles_field` below have to be"
            " specified."
        }
    )
    file_path: str = field(
        default="",
        metadata={
            "help": "Ignored unless `datase_name` is `custom`. In that case it's "
            "a path to a .csv file containing the training data."
        },
    )
    dataset_path: str = field(
        default=DATA_ROOT,
        metadata={
            "help": "Path where the TorchDrug dataset will be stored. This is ignored "
            "if `datase_name` is `custom`."
        },
    )
    target_field: str = field(
        default="",
        metadata={
            "help": "Ignored unless `datase_name` is `custom`. In that case it's a str "
            "with name of the column containing the property that can be optimized."
            "Currently TorchDrug only supports `plogp` and `qed`."
        },
    )
    smiles_field: str = field(
        default="smiles",
        metadata={
            "help": "Ignored unless `datase_name` is `custom`. In that case it's the "
            "name of the column containing the SMILES strings."
        },
    )
    transform: str = field(
        default="lambda x: x",
        metadata={
            "help": "Optional data transformation function. Has to be a lambda function"
            " (written as a string) that operates on the batch dictionary."
            "See torchdrug docs for details."
        },
    )
    verbose: int = field(
        default=1, metadata={"help": "Output verbosity level for dataset."}
    )
    lazy: bool = field(
        default=False,
        metadata={
            "help": "If yes, molecules are processed in the dataloader. This is faster "
            "for setup but slower at training time."
        },
    )
    node_feature: str = field(
        default="default",
        metadata={"help": "Node features (or node feature list) to extract."},
    )
    edge_feature: str = field(
        default="default",
        metadata={"help": "Edge features (or edge feature list) to extract."},
    )
    graph_feature: Optional[str] = field(
        default=None,
        metadata={"help": "Graph features (or graph feature list) to extract."},
    )
    with_hydrogen: bool = field(
        default=False,
        metadata={"help": "Whether hydrogens are stored in molecular graph."},
    )
    no_kekulization: bool = field(
        default=False,
        metadata={
            "help": "Whether SMILES kekulization is used. Per default, it is used."
        },
    )


@dataclass
class TorchDrugSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to TorchDrug trainer."""

    __name__ = "saving_args"

    model_path: str = field(
        metadata={"help": "Path where the model artifacts are stored."}
    )
    training_name: str = field(metadata={"help": "Name used to identify the training."})
    dataset_name: str = field(
        metadata={
            "help": f"Identifier for the dataset. Has to be in {DATASET_FACTORY.keys()}"
            ". Can either point to one of the predefined TorchDrug datasets or it can "
            "be `custom` if the user brings their own dataset. If `custom`, then the "
            "arguments `file_path`, `target_field` and `smiles_field` below have to be"
            " specified."
        }
    )
    task: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optimization task for goal-driven generation."
            "Currently, TorchDrug only supports `plogp` and `qed`."
        },
    )
    file_path: str = field(
        default="",
        metadata={
            "help": "Ignored unless `datase_name` is `custom`. In that case it's "
            "a path to a .csv file containing the training data."
        },
    )
    epochs: int = field(default=10, metadata={"help": "Number of epochs."})
