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
import inspect
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from pytoda.smiles.transforms import Augment
from pytoda.transforms import AugmentByReversing
from sklearn.utils import shuffle
from terminator.selfies import encoder
from terminator.tokenization import ExpressionBertTokenizer
from transformers.hf_argparser import string_to_bool
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

TRANSFORM_FACTORY = {"SELFIES": encoder}
AUGMENT_FACTORY = {
    "SMILES": Augment(),
    "SELFIES": Augment(),
    "AAS": AugmentByReversing(),
}


class Property:
    name: str
    minimum: float = 0
    maximum: float = 0
    expression_separator: str = "|"
    normalize: bool = False

    def __init__(self, name: str):
        self.name = name
        self.mask_lengths: List = []

    def update(self, line: str):
        prop = line.split(self.name)[-1].split(self.expression_separator)[0]
        try:
            val = float(prop)
        except ValueError:
            logger.error(f"Could not convert property {prop} in {line} to float.")
        if val < self.minimum:
            self.minimum = val
        elif val > self.maximum:
            self.maximum = val
        self.mask_lengths.append(len(prop))

    @property
    def mask_length(self) -> int:
        """
        How many tokens are being masked for this property.
        """
        counts = Counter(self.mask_lengths)
        if len(counts) > 1:
            logger.warning(
                f"Not all {self.name} properties have same number of tokens: {counts}"
            )
        return int(counts.most_common(1)[0][0])


def add_tokens_from_lists(
    tokenizer: ExpressionBertTokenizer, train_data: List[str], test_data: List[str]
) -> Tuple[ExpressionBertTokenizer, Dict[str, Property], List[str], List[str]]:
    """
    Addding tokens to a tokenizer from parsed datasets hold in memory.

    Args:
        tokenizer: The tokenizer.
        train_data: List of strings, one per sample.
        test_data: List of strings, one per sample.

    Returns:
       Tuple with:
            tokenizer with updated vocabulary.
            dictionary of property names and full property objects.
            list of strings with training samples.
            list of strings with testing samples.
    """
    num_tokens = len(tokenizer)
    properties: Dict[str, Property] = {}
    all_tokens: Set = set()
    for data in [train_data, test_data]:
        for line in data:
            # Grow the set of all tokens in the dataset
            toks = tokenizer.tokenize(line)
            all_tokens = all_tokens.union(toks)
            # Grow the set of all properties (assumes that the text follows the last `|`)
            props = [
                x.split(">")[0] + ">"
                for x in line.split(tokenizer.expression_separator)[:-1]
            ]
            for prop in props:
                if prop not in properties.keys():
                    properties[prop] = Property(prop)
                properties[prop].update(line)

    # Finish adding new tokens
    tokenizer.add_tokens(list(all_tokens))
    tokenizer.update_vocab(all_tokens)  # type:ignore
    logger.info(f"Added {len(tokenizer)-num_tokens} new tokens to tokenizer.")

    return tokenizer, properties, train_data, test_data


def prepare_datasets_from_files(
    tokenizer: ExpressionBertTokenizer,
    train_path: str,
    test_path: str,
    augment: int = 0,
) -> Tuple[ExpressionBertTokenizer, Dict[str, Property], List[str], List[str]]:
    """
    Converts datasets saved in provided `.csv` paths into RT-compatible datasets.
    NOTE: Also adds the new tokens from train/test data to provided tokenizer.

    Args:
        tokenizer: The tokenizer.
        train_path: Path to the training data.
        test_path: Path to the testing data.
        augment: Factor by which each training sample is augmented.

    Returns:
       Tuple with:
            tokenizer with updated vocabulary.
            dict of property names and property objects.
            list of strings with training samples.
            list of strings with testing samples.
    """

    # Setup data transforms and augmentations
    train_data: List[str] = []
    test_data: List[str] = []
    properties: List[str] = []

    aug = AUGMENT_FACTORY.get(tokenizer.language, lambda x: x)
    trans = TRANSFORM_FACTORY.get(tokenizer.language, lambda x: x)

    for i, (data, path) in enumerate(
        zip([train_data, test_data], [train_path, test_path])
    ):

        if not path.endswith(".csv"):
            raise TypeError(f"Please provide a csv file not {path}.")

        # Load data
        df = shuffle(pd.read_csv(path))
        if "text" not in df.columns:
            raise ValueError("Please provide text in the `text` column.")

        if i == 1 and set(df.columns) != set(properties + ["text"]):
            raise ValueError(
                "Train and test data have to have identical columns, not "
                f"{set(properties + ['text'])} and {set(df.columns)}."
            )
        properties = sorted(list(set(properties).union(list(df.columns))))
        properties.remove("text")

        # Parse data and create RT-compatible format
        for j, row in df.iterrows():
            line = "".join(
                [
                    f"<{p}>{row[p]:.3f}{tokenizer.expression_separator}"
                    for p in properties
                ]
                + [trans(row.text)]  # type: ignore
            )
            data.append(line)

        # Perform augmentation on training data if applicable
        if i == 0 and augment is not None and augment > 1:
            for _ in range(augment):
                for j, row in df.iterrows():
                    line = "".join(
                        [
                            f"<{p}>{row[p]:.3f}{tokenizer.expression_separator}"
                            for p in properties
                        ]
                        + [trans(aug(row.text))]  # type: ignore
                    )
                    data.append(line)

    return add_tokens_from_lists(
        tokenizer=tokenizer, train_data=train_data, test_data=test_data
    )


def get_train_config_dict(
    training_args: Dict[str, Any], properties: Set
) -> Dict[str, Any]:
    return {
        "alternate_steps": training_args["alternate_steps"],
        "reset_training_loss": True,
        "cg_collator": training_args["cg_collator"],
        "cc_loss": training_args["cc_loss"],
        "property_tokens": list(properties),
        "cg_collator_params": {
            "do_sample": False,
            "property_tokens": list(properties),
            "plm_probability": training_args["plm_probability"],
            "max_span_length": training_args["max_span_length"],
            "entity_separator_token": training_args["entity_separator_token"],
            "mask_entity_separator": training_args["mask_entity_separator"],
            "entity_to_mask": training_args["entity_to_mask"],
        },
    }


@dataclass
class TransformersTrainingArgumentsCLI(TrainingArguments):
    """
    GT4SD ships with a CLI to launch training. This conflicts with some data types
    native in `transformers.training_arguments.TrainingArguments` especially iterables
    which cannot be easily passed from CLI.
    Therefore, this class changes the affected attributes to CLI compatible datatypes.
    """

    label_names: Optional[str] = field(  # type: ignore
        default=None,
        metadata={
            "help": "A string containing keys in your dictionary of inputs that correspond to the labels."
            "A single string, but can contain multiple keys separated with comma: `key1,key2`"
        },
    )
    report_to: Optional[str] = field(  # type: ignore
        default=None,
        metadata={
            "help": "The list of integrations to report the results and logs to."
            "A single string, but can contain multiple keys separated with comma: `i1,i2`"
        },
    )
    sharded_ddp: str = field(
        default="",
        metadata={
            "help": "Whether or not to use sharded DDP training (in distributed training only). The base option "
            "should be `simple`, `zero_dp_2` or `zero_dp_3` and you can add CPU-offload to `zero_dp_2` or `zero_dp_3` "
            "like this: zero_dp_2 offload` or `zero_dp_3 offload`. You can add auto-wrap to `zero_dp_2` or "
            "with the same syntax: zero_dp_2 auto_wrap` or `zero_dp_3 auto_wrap`.",
        },
    )
    tf32: Optional[str] = field(  # type: ignore
        default="no",
        metadata={
            "help": (
                "Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental"
                " API and it may change."
            )
        },
    )
    disable_tqdm: Optional[str] = field(  # type: ignore
        default="no",
        metadata={"help": "Whether or not to disable the tqdm progress bars."},
    )
    greater_is_better: Optional[str] = field(  # type: ignore
        default="no",
        metadata={
            "help": "Whether the `metric_for_best_model` should be maximized or not."
        },
    )
    remove_unused_columns: Optional[str] = field(  # type: ignore
        default="yes",
        metadata={
            "help": "Remove columns not required by the model when using an nlp.Dataset."
        },
    )
    load_best_model_at_end: Optional[str] = field(  # type: ignore
        default=None,
        metadata={
            "help": "Whether or not to load the best model found during training at the end of training."
        },
    )
    ddp_find_unused_parameters: Optional[str] = field(  # type: ignore
        default="no",
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    evaluation_strategy: Optional[str] = field(  # type: ignore
        default="no",
        metadata={
            "help": (
                "The evaluation strategy to adopt during training. Possible values are:"
                " - 'no': No evaluation is done during training."
                " - 'steps'`: Evaluation is done (and logged) every `eval_steps`."
                " - 'epoch': Evaluation is done at the end of each epoch."
            )
        },
    )
    lr_scheduler_type: Optional[str] = field(  # type: ignore
        default="linear",
        metadata={
            "help": (
                "The scheduler type to use. See the documentation of "
                "`transformers.SchedulerType` for all possible values."
            )
        },
    )
    logging_strategy: Optional[str] = field(  # type: ignore
        default="steps",
        metadata={
            "help": (
                "The logging strategy to adopt during training. Possible values are:"
                " - 'no': No logging is done during training."
                " - 'steps'`: Logging is done every `logging_steps`."
                " - 'epoch': Logging is done at the end of each epoch."
            )
        },
    )
    save_strategy: Optional[str] = field(  # type: ignore
        default="steps",
        metadata={
            "help": (
                "The saving strategy to adopt during training. Possible values are:"
                " - 'no': No saving is done during training."
                " - 'steps'`: Saving is done every `saving_steps`."
                " - 'epoch': Saving is done at the end of each epoch."
            )
        },
    )
    hub_strategy: Optional[str] = field(  # type: ignore
        default="every_save",
        metadata={
            "help": (
                "Optional, defaults to `every_save`. Defines the scope of what is pushed "
                "to the Hub and when. Possible values are:"
                " - `end`: push the model, its configuration, the tokenizer (if passed "
                "       along to the Trainer and a draft of a model card when the "
                "       Trainer.save_model method is called."
                " - `every_save`: push the model, its configuration, the tokenizer (if "
                "       passed along to the Trainer and a draft of a model card each time "
                "       there is a model save. The pushes are asynchronous to not block "
                "       training, and in case the save are very frequent, a new push is "
                "       only attempted if the previous one is finished. A last push is made "
                "       with the final model at the end of training."
                " - `checkpoint`: like `every_save` but the latest checkpoint is also "
                "       pushed in a subfolder named last-checkpoint, allowing you to resume "
                "       training easily with `trainer.train(resume_from_checkpoint)`."
                " - `all_checkpoints`: like `checkpoint` but all checkpoints are pushed "
                "       like they appear in the output folder (so you will get one "
                "       checkpoint folder per folder in your final repository)."
            )
        },
    )
    optim: Optional[str] = field(  # type: ignore
        default="adamw_hf",
        metadata={
            "help": (
                "The optimizer to use. One of  `adamw_hf`, `adamw_torch`, `adafactor` "
                "or `adamw_apex_fused`."
            )
        },
    )

    def __post_init__(self):
        """
        Necessary because the our ArgumentParser (that is based on argparse) converts
        empty strings to None. This is prohibitive since the HFTrainer relies on
        them being actual strings. Only concerns a few arguments.
        """
        if self.sharded_ddp is None:
            self.sharded_ddp = ""
        if self.fsdp is None:  # type: ignore
            self.fsdp = ""

        self.disable_tqdm = string_to_bool(self.disable_tqdm)
        self.tf32 = string_to_bool(self.tf32)

        super().__post_init__()


def get_hf_training_arg_object(training_args: Dict[str, Any]) -> TrainingArguments:
    """
    A method to convert a training_args Dictionary into a HuggingFace
    `TrainingArguments` object.
    This routine also takes care of removing arguments that are not necessary.

    Args:
        training_args: A dictionary of training arguments.

    Returns:
        object of type `TrainingArguments`.
    """

    # Get attributes of parent class
    org_attrs = dict(inspect.getmembers(TrainingArguments))

    # Remove attributes that were specified by child classes
    hf_training_args = {k: v for k, v in training_args.items() if k in org_attrs.keys()}

    # Instantiate class object
    hf_train_object = TrainingArguments(training_args["output_dir"])

    # Set attributes manually (since this is a `dataclass` not everything can be passed
    # to constructor)
    for k, v in hf_training_args.items():
        setattr(hf_train_object, k, v)

    return hf_train_object
