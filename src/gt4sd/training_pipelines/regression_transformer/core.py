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

from ...configuration import gt4sd_configuration_instance
from ..core import TrainingPipelineArguments
from .utils import TransformersTrainingArgumentsCLI

DATA_ROOT = os.path.join(
    gt4sd_configuration_instance.gt4sd_local_cache_path, "data", "RegressionTransformer"
)
os.makedirs(DATA_ROOT, exist_ok=True)


@dataclass
class RegressionTransformerTrainingArguments(
    TrainingPipelineArguments, TransformersTrainingArgumentsCLI
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
    num_train_epochs: int = field(default=10, metadata={"help": "Number of epochs."})
    batch_size: int = field(default=16, metadata={"help": "Size of the batch."})
    log_interval: int = field(
        default=100, metadata={"help": "Number of steps between log intervals."}
    )
    gradient_interval: int = field(
        default=1, metadata={"help": "Gradient accumulation steps"}
    )
    eval_steps: int = field(
        default=1000,
        metadata={"help": "The time interval at which validation is performed."},
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
    cg_collator: str = field(
        default="vanilla_cg",
        metadata={
            "help": "The collator class. Following options are implemented: "
            "'vanilla_cg': Collator class that does not mask the properties but anything else as a regular DataCollatorForPermutationLanguageModeling. Can optionally replace the properties with sampled values. "
            "NOTE: This collator can deal with multiple properties. "
            "'multientity_cg': A training collator the conditional-generation task that can handle multiple entities. "
            "Default: vanilla_cg."
        },
    )
    entity_to_mask: int = field(
        default=-1,
        metadata={
            "help": "Only applies if `cg_collator='multientity_cg'`. The entity that is being masked during training. 0 corresponds to first entity and so on. -1 corresponds to "
            "a random sampling scheme where the entity-to-be-masked is determined "
            "at runtime in the collator. NOTE: If 'mask_entity_separator' is true, "
            "this argument will not have any effect. Defaults to -1."
        },
    )
    entity_separator_token: str = field(
        default=".",
        metadata={
            "help": "Only applies if `cg_collator='multientity_cg'`.The token that is used to separate "
            "entities in the input. Defaults to '.' (applicable to SMILES & SELFIES)"
        },
    )
    mask_entity_separator: bool = field(
        default=False,
        metadata={
            "help": "Only applies if `cg_collator='multientity_cg'`. Whether or not the entity separator token can be masked. If True, *all** textual tokens can be masked and we "
            "the collator behaves like the `vanilla_cg ` even though it is a `multientity_cg`. If False, the exact behavior "
            "depends on the entity_to_mask argument. Defaults to False."
        },
    )


@dataclass
class RegressionTransformerDataArguments(TrainingPipelineArguments):
    """Arguments related to RegressionTransformer data loading."""

    __name__ = "dataset_args"

    train_data_path: str = field(
        metadata={
            "help": "Path to a `.csv` file with the input training data. The file has to "
            "contain a `text` column (with the string input, e.g, SMILES, AAS, natural "
            "text) and an arbitrary number of numerical columns."
        },
    )
    test_data_path: str = field(
        metadata={
            "help": "Path to a `.csv` file with the input testing data. The file has to "
            "contain a `text` column (with the string input, e.g, SMILES, AAS, natural "
            "text) and an arbitrary number of numerical columns."
        },
    )
    augment: Optional[int] = field(
        default=0,
        metadata={
            "help": "Factor by which the training data is augmented. The data modality "
            "(SMILES, SELFIES, AAS, natural text) is inferred from the tokenizer. "
            "NOTE: For natural text, no augmentation is supported. Defaults to 0, "
            "meaning no augmentation. "
        },
    )
    save_datasets: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to save the datasets to disk. Datasets will be saved as `.txt` file to "
            "the same location where `train_data_path` and `test_data_path` live. Defaults to False."
        },
    )


@dataclass
class RegressionTransformerSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to RegressionTransformer trainer."""

    __name__ = "saving_args"

    model_path: str = field(
        metadata={"help": "Path where the model artifacts are stored."}
    )
    checkpoint_name: str = field(
        default=str(),
        metadata={
            "help": "Name for the checkpoint that should be copied to inference model. "
            "Has to be a subfolder of `model_path`. Defaults to empty string meaning that "
            "files are taken from `model_path` (i.e., after training finished)."
        },
    )
