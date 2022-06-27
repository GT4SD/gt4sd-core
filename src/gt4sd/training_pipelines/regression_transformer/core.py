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
"""Regression Transformer training utilities."""
import os
from dataclasses import dataclass, field
from typing import Optional

from ..core import TrainingPipelineArguments
from .utils import TransformersTrainingArgumentsCLI

DATA_ROOT = os.path.join(
    os.path.expanduser("~"), ".gt4sd", "data", "RegressionTransformer"
)
os.makedirs(DATA_ROOT, exist_ok=True)


@dataclass
class RegressionTransformerTrainingArguments(
    TransformersTrainingArgumentsCLI, TrainingPipelineArguments
):
    """
    Arguments related to RegressionTransformer trainer.
    NOTE: All arguments from `transformers.training_args.TrainingArguments` can be used.
    Only additional ones are specified below.
    """

    __name__ = "training_args"

    training_name: str = field(
        default="rt_training", metadata={"help": "Name used to identify the training."}
    )
    epochs: int = field(default=10, metadata={"help": "Number of epochs."})
    batch_size: int = field(default=16, metadata={"help": "Size of the batch."})
    log_interval: int = field(
        default=100, metadata={"help": "Number of steps between log intervals."}
    )
    gradient_interval: int = field(
        default=1, metadata={"help": "Gradient accumulation steps"}
    )

    max_span_length: int = field(
        default=5, metadata={"help": "Max length of a span of masked tokens for PLM."}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for PLM."
        },
    )
    alternate_steps: int = field(
        default=50,
        metadata={
            "help": "Per default, training alternates between property prediction and "
            "conditional generation. This argument specifies the alternation frequency."
            "If you set it to 0, no alternation occurs and we fall back to vanilla "
            "permutation language modeling (PLM). Default: 50."
        },
    )
    cc_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether the cycle-consistency loss is computed during the conditional "
            "generation task. Defaults to True."
        },
    )


@dataclass
class RegressionTransformerDataArguments(TrainingPipelineArguments):
    """Arguments related to RegressionTransformer data loading."""

    __name__ = "dataset_args"

    data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a `.csv` file with the data. File has to contain a `text` column "
            "(with the string input, e.g, SMILES, AAS, natural text)"
        },
    )
    train_data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data file. Should contain text and properties"
            "in RT-compatible format, e.g. QED on SELFIES: `<qed>0.123|[C][C][O]. "
            "Dependent on the tokenizer, can also be natural text, AA sequences etc."
            "NOTE: Only used if `data_path` is not specified."
        },
    )
    test_data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The evaluation data file. Should contain text and properties"
            "in RT-compatible format, e.g. QED on SELFIES: `<qed>0.123|[C][C][O]. "
            "Dependent on the tokenizer, can also be natural text, AA sequences etc."
            "NOTE: Only used if `data_path` is not specified."
        },
    )
    test_fraction: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "Fraction of data used for testing. Only used if `data_path` is specified."
        },
    )
    augment: Optional[int] = field(
        default=0,
        metadata={
            "help": "Factor by which the training data is augmented. Only used if `data_path` is "
            "specified. The data modality (SMILES, SELFIES, AAS, natural text) is "
            "inferred. NOTE: For natural text, no augmentation is supported. Defaults to "
            "0, meaning no augmentation. "
        },
    )

    line_by_line: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether lines of text in the dataset are to be handled as distinct samples."
        },
    )


@dataclass
class RegressionTransformerSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to RegressionTransformer trainer."""

    __name__ = "saving_args"

    # TODO: Not really sure what else should be here.
    model_path: str = field(
        metadata={"help": "Path where the model artifacts are stored."}
    )
    training_name: str = field(metadata={"help": "Name used to identify the training."})
    epochs: int = field(default=10, metadata={"help": "Number of epochs."})
