import inspect
import logging
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pytoda.smiles.transforms import Augment
from pytoda.transforms import AugmentByReversing
from terminator.selfies import encoder
from terminator.tokenization import ExpressionBertTokenizer
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

TRANSFORM_FACTORY = {"SELFIES": encoder}
AUGMENT_FACTORY = {
    "SMILES": Augment(),
    "SELFIES": Augment(),
    "AAS": AugmentByReversing(),
}


def prepare_and_split_data(
    path: str, test_fraction: float, augment: int, language: str
) -> Tuple[List[str], List[str]]:
    """

    Args:
        path: A path to a `.csv` file with the data.
        test_fraction: Fraction of data used for testing.
        augment: Factor by which each training sample is augmented.
        language: Language of the tokenizer.

    Returns:
        Tuple of training and test dataset.
    """

    if test_fraction <= 0 or test_fraction >= 1:
        raise ValueError(f"Test fraction has to be 0 < t < 1, not {test_fraction}")
    if not path.endswith(".csv"):
        raise TypeError(f"Please provide a csv file not {path}.")
    logger.info(f"Using {int(test_fraction*100)}% of the data for testing.")

    # Load data
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError("Please provide text in the `text` column.")

    # Split dataset
    idxs = list(range(len(df)))
    np.random.shuffle(idxs)
    test_idxs = idxs[: int(len(df) * test_fraction)]
    train_idxs = idxs[int(len(df) * test_fraction) :]

    # Setup data transforms and augmentations
    aug = AUGMENT_FACTORY.get(language, lambda x: x)
    trans = TRANSFORM_FACTORY.get(language, lambda x: x)

    # Create RT-compatible dataset
    train_data, test_data = [], []
    properties = list(df.columns)
    properties.remove("text")

    for i, row in df.iterrows():
        line = "".join([f"<{p}>{row[p]:.3f}|" for p in properties] + [trans(row.text)])
        if i in test_idxs:
            test_data.append(line)
        else:
            train_data.append(line)

    # Perform augmentation on training data if applicable
    if augment > 1:
        for _ in range(augment):
            for i in train_idxs:
                row = df.iloc[i]
                line = "".join(
                    [f"<{p}>{row[p]:.3f}|" for p in properties] + [trans(aug(row.text))]
                )
                train_data.append(line)

    logger.info("Data splitting completed.")
    return train_data, test_data


def add_tokens_from_lists(
    tokenizer: ExpressionBertTokenizer, train_data: List[str], test_data: List[str]
) -> Tuple[ExpressionBertTokenizer, Set]:
    """
    Addding tokens to a tokenizer from parsed datasets hold in memory.

    Args:
        tokenizer: The tokenizer.
        train_data: List of strings, one per sample.
        test_data: List of strings, one per sample.

    Returns:
       Tuple with:
            tokenizer with updated vocabulary.
            set of property tokens.
    """
    num_tokens = len(tokenizer)
    properties = set()
    all_tokens = set()
    for data in [train_data, test_data]:
        for i, line in enumerate(data):
            # Grow the set of all tokens in the dataset
            toks = tokenizer.tokenize(line)
            all_tokens = all_tokens.union(toks)
            # Grow the set of all properties (assumes that the text follows the last `|`)
            props = [x.split(">")[0] + ">" for x in line.split("|")[:-1]]
            properties = properties.union(props)

    # Finish adding new tokens
    tokenizer.add_tokens(list(all_tokens))
    tokenizer.update_vocab(all_tokens)
    logger.info(f"Added {len(tokenizer)-num_tokens} new tokens to tokenizer.")

    return tokenizer, properties


def add_tokens_from_files(
    tokenizer: ExpressionBertTokenizer, train_path: str, test_path: str
) -> Tuple[ExpressionBertTokenizer, Set]:
    """
    Addding tokens to a tokenizer from paths to training/testing data files.

    Args:
        tokenizer: The tokenizer.
        train_path: Path to the training data.
        test_path: Path to the testing data.

    Returns:
       Tuple with:
            tokenizer with updated vocabulary.
            set of property tokens.
    """

    datasets = []
    for path in [train_path, test_path]:
        with open(path, encoding="utf-8") as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        datasets.append(lines)

    return add_tokens_from_lists(
        tokenizer=tokenizer, train_data=datasets[0], test_data=datasets[1]
    )


def get_train_config_dict(
    training_args: Dict[str, Any], properties: Set
) -> Dict[str, Any]:
    return {
        "alternate_steps": training_args["alternate_steps"],
        "reset_training_loss": True,
        "cc_loss": training_args["cc_loss"],
        "property_tokens": list(properties),
        "cg_collator_params": {
            "do_sample": False,
            "property_tokens": list(properties),
            "plm_probability": training_args["plm_probability"],
            "max_span_length": training_args["max_span_length"],
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

    label_names: Optional[str] = field(
        default=None,
        metadata={
            "help": "A string containing keys in your dictionary of inputs that correspond to the labels."
            "A single string, but can contain multiple keys separated with comma: `key1,key2`"
        },
    )
    report_to: Optional[str] = field(
        default=None,
        metadata={
            "help": "The list of integrations to report the results and logs to."
            "A single string, but can contain multiple keys separated with comma: `i1,i2`"
        },
    )
    sharded_ddp: str = field(
        default=" ",
        metadata={
            "help": "Whether or not to use sharded DDP training (in distributed training only). The base option "
            "should be `simple`, `zero_dp_2` or `zero_dp_3` and you can add CPU-offload to `zero_dp_2` or `zero_dp_3` "
            "like this: zero_dp_2 offload` or `zero_dp_3 offload`. You can add auto-wrap to `zero_dp_2` or "
            "with the same syntax: zero_dp_2 auto_wrap` or `zero_dp_3 auto_wrap`.",
        },
    )

    # TODO: Have to parse these ones into lists


def get_hf_training_arg_object(training_args: Dict[str, Any]) -> TrainingArguments:
    """
    A method to convert a training_args Dictionary into a HuggingFace
    `TrainingArguments` object.
    This routine also takes care of removing arguments that are not necessary.

    Args:
        training_args: A dictionary of training

    Returns:
        object of type `TrainingArguments`.
    """

    # Get attributes of parent class
    org_attrs = dict(inspect.getmembers(TrainingArguments))

    # Remove attributes that were specified by child classes
    hf_training_args = {k: v for k, v in training_args.items() if k in org_attrs.keys()}

    # Instantiate class object
    hf_train_object = TrainingArguments(hf_training_args['output_dir'])

    # Set attributes manually (since this is a `dataclass` not everything can be passed
    # to constructor)
    for k, v in hf_training_args.items():
        setattr(hf_train_object, k, v)

    return hf_train_object
