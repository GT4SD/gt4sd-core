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
"""PaccMann training utilities."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..core import TrainingPipeline, TrainingPipelineArguments


class PaccMannTrainingPipeline(TrainingPipeline):
    """PyTorch lightining training pipelines."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
    ) -> None:
        """Generic training function for PaccMann training.

        Args:
            training_args: training arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.

        Raises:
            NotImplementedError: the generic trainer does not implement the pipeline.
        """
        raise NotImplementedError


@dataclass
class PaccMannTrainingArguments(TrainingPipelineArguments):
    """Arguments related to PaccMann trainer."""

    __name__ = "training_args"

    model_path: str = field(
        metadata={"help": "Path where the model artifacts are stored."}
    )
    training_name: str = field(metadata={"help": "Name used to identify the training."})
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to model checkpoint for weights initialization. Leave None if you want to train a model from scratch"
        },
    )
    epochs: int = field(default=50, metadata={"help": "Number of epochs."})
    batch_size: int = field(default=256, metadata={"help": "Size of the batch."})
    learning_rate: float = field(
        default=0.0005, metadata={"help": "Learning rate used in training."}
    )
    optimizer: str = field(
        default="adam", metadata={"help": "Optimizer used during training."}
    )
    log_interval: int = field(
        default=100, metadata={"help": "Number of steps between log intervals."}
    )
    save_interval: int = field(
        default=1000, metadata={"help": "Number of steps between model save intervals."}
    )
    eval_interval: int = field(
        default=500, metadata={"help": "Number of steps between evaluation intervals."}
    )


@dataclass
class PaccMannDataArguments(TrainingPipelineArguments):
    """Arguments related to PaccMann data loading."""

    __name__ = "dataset_args"

    train_smiles_filepath: str = field(
        metadata={"help": "Training file containing SMILES in .smi format."}
    )
    test_smiles_filepath: str = field(
        metadata={"help": "Testing file containing SMILES in .smi format."}
    )
    smiles_language_filepath: str = field(
        default="none", metadata={"help": "Optional SMILES language file."}
    )
    add_start_stop_token: bool = field(
        default=True, metadata={"help": "Whether start and stop token should be added."}
    )
    selfies: bool = field(
        default=True, metadata={"help": "Whether SELFIES representations are used."}
    )
    num_workers: int = field(
        default=0, metadata={"help": "Number of workers used in data loading."}
    )
    pin_memory: bool = field(
        default=False, metadata={"help": "Whether memory in the data loader is pinned."}
    )
    augment_smiles: bool = field(
        default=False, metadata={"help": "Whether SMILES augumentation is used."}
    )
    canonical: bool = field(
        default=False, metadata={"help": "Whether SMILES canonicalization is used."}
    )
    kekulize: bool = field(
        default=False, metadata={"help": "Whether SMILES kekulization is used."}
    )
    all_bonds_explicit: bool = field(
        default=False, metadata={"help": "Whether all bonds are explicit."}
    )
    all_hs_explicit: bool = field(
        default=False, metadata={"help": "Whether all hydrogens are explicit."}
    )
    remove_bonddir: bool = field(
        default=False, metadata={"help": "Remove bond directionality."}
    )
    remove_chirality: bool = field(
        default=False, metadata={"help": "Remove chirality."}
    )


@dataclass
class PaccMannSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to PaccMann trainer."""

    __name__ = "saving_args"

    model_path: str = field(
        metadata={"help": "Path where the model artifacts are stored."}
    )
    training_name: str = field(metadata={"help": "Name used to identify the training."})
