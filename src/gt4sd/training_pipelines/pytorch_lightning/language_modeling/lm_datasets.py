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
"""Dataset routines-filtering, dataset building."""

import json
import os
from functools import lru_cache
from typing import Any, Callable, Dict, List, Union

import sentencepiece as _sentencepiece
import pytorch_lightning as pl
from datasets import DatasetDict
from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    default_data_collator,
)
from transformers.tokenization_utils_base import BatchEncoding

# Sentencepiece has to be loaded before lightning
_sentencepiece


class LMDataset(Dataset):
    """LM dataset class."""

    def __init__(
        self,
        filepath: str,
        tokenizer: Callable,
    ) -> None:
        """Initialize the LM data module.

        Args:
            filepath: path where the dataset is located.
            tokenizer: tokenize function to be used in the module.
        """

        self.filepath = filepath
        self.tokenizer = tokenizer
        self.length = LMDataset.count_examples(filepath)

        if not self.filepath.endswith(".jsonl") and not self.filepath.endswith(".json"):
            raise ValueError(f"{filepath} is not a .jsonl or a json.")

    @lru_cache()
    def examples_reader(self) -> List[Dict[str, str]]:
        """Read instances from a filepath.

        Returns:
           list of instances.
        """
        with open(self.filepath) as fp:
            return [json.loads(line.strip()) for line in fp]

    @staticmethod
    def count_examples(filepath: str) -> int:
        """Count instances of a filepath.

        Args:
           filepath: path of the dataset.
        Returns:
           number of examples existed in the given filepath.
        """

        def _make_gen(reader):
            while True:
                b = reader(2**16)
                if not b:
                    break
                yield b

        with open(filepath, "rb") as f:
            count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))  # type: ignore
        return count

    def __len__(self) -> int:
        """Number of instances of the dataset.

        Returns:
           number of instances
        """
        return self.length

    def __getitem__(self, index) -> BatchEncoding:
        """Get an item of the dataset.

        Args:
            index: index of the item.
        Returns:
            tokenized item.
        """

        examples = self.examples_reader()
        example = self.tokenizer(examples[index])

        return example


class DataModule(pl.LightningDataModule):
    """Pytorch-lightning-style data module for LM dataset."""

    def __init__(self, dataset_args: Dict[str, Any], tokenizer: AutoTokenizer) -> None:
        """Initialize the data module.

        Args:
            dataset_args: dictionary containing the arguments for the lightning data module creation.
            tokenizer: tokenizer to be used in the module.
        """

        super().__init__()

        self.dataset: DatasetDict

        self.dataset_args = dataset_args

        self.tokenizer = tokenizer

        self.data_collator = default_data_collator

        if "num_dataloader_workers" not in self.dataset_args:

            self.dataset_args["num_dataloader_workers"] = 8

            cpus_count = os.cpu_count()
            if cpus_count is not None:
                self.dataset_args["num_dataloader_workers"] = min(
                    self.dataset_args["num_dataloader_workers"], cpus_count
                )

    def build_dataset(self, path: str) -> Dataset:
        """
        Build the dataset.

        Args:
            path: path where the dataset is located.
        Returns:
            a torch Dataset.
        """

        if path.endswith(".jsonl") or path.endswith(".json"):
            return LMDataset(path, self.tokenize_function)
        elif os.path.isdir(path):
            return ConcatDataset(
                datasets=[
                    LMDataset(os.path.join(path, filename), self.tokenize_function)
                    for filename in os.listdir(path)
                    if filename.endswith(".jsonl") or filename.endswith(".json")
                ]
            )
        else:
            raise TypeError(f"{path} type is not supported for dataset")

    def tokenize_function(
        self, examples: Dict[str, Union[int, slice]]
    ) -> BatchEncoding:
        """Tokenize the given examples.

        Args:
            examples: list of examples.
        Returns:
            tokenized examples.
        """

        truncation = self.dataset_args.get("truncation", True)
        padding = self.dataset_args.get("padding", "max_length")
        max_length = self.dataset_args.get("max_length", 512)

        return self.tokenizer(  # type: ignore
            examples["text"],
            truncation=truncation,
            padding=padding,
            max_length=max_length,
        )

    def load(self) -> None:
        """Load datasets from the given files."""

        self.datasets = {
            "train": self.build_dataset(self.dataset_args["train_file"]),
            "validation": self.build_dataset(self.dataset_args["validation_file"]),
        }

        logger.info(
            f"Training set size: {len(self.datasets['train'])} - Validation set size: {len(self.datasets['validation'])}"  # type: ignore
        )

    def train_dataloader(self) -> DataLoader:
        """Create the DataLoader for the traning step.

        Returns:
            pytorch-like dataloader.
        """
        return DataLoader(
            self.datasets["train"],  # type: ignore
            batch_size=self.dataset_args["batch_size"],
            num_workers=self.dataset_args["num_dataloader_workers"],
            collate_fn=self.data_collator,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the DataLoader for the traning step.

        Returns:
            pytorch-like dataloader.
        """
        return DataLoader(
            self.datasets["validation"],  # type: ignore
            batch_size=self.dataset_args["batch_size"],
            num_workers=self.dataset_args["num_dataloader_workers"],
            collate_fn=self.data_collator,
        )


class MLMDataModule(DataModule):
    """Pytorch-lightning-style data module for MLM dataset."""

    def __init__(
        self, dataset_args: Dict[str, Union[float, str, int]], tokenizer: AutoTokenizer
    ) -> None:
        """Initialize the data module.

        Args:
            dataset_args: dictionary containing the metadata for the lightning data module creation.
            tokenizer: tokenizer to be used in the module.
        """
        super().__init__(dataset_args, tokenizer)

        self.data_collator = DataCollatorForLanguageModeling(
            self.tokenizer, self.dataset_args["mlm_probability"]  # type: ignore
        )

        self.load()


class CGMDataModule(DataModule):
    """Pytorch-lightning-style data module for conditional generation dataset."""

    def __init__(
        self, dataset_args: Dict[str, Union[float, str, int]], tokenizer: AutoTokenizer
    ) -> None:
        """
        Initialize the data module.

        Args:
            dataset_args: dictionary containing the metadata for the lightning data module creation.
            tokenizer: tokenizer to be used in the module.
        """
        super().__init__(dataset_args, tokenizer)

        self.load()

    def tokenize_function(
        self, examples: Dict[str, Union[int, slice]]
    ) -> BatchEncoding:
        """Tokenize the given examples.

        Args:
            examples: list of examples.
        Returns:
            tokenized examples.
        """

        truncation = self.dataset_args.get("truncation", True)
        padding = self.dataset_args.get("padding", "max_length")
        max_length = self.dataset_args.get("max_length", 512)

        source = self.tokenizer(  # type: ignore
            examples["source"],
            truncation=truncation,
            padding=padding,
            max_length=max_length,
        )

        targets = self.tokenizer(  # type: ignore
            examples["target"],
            truncation=truncation,
            padding=padding,
            max_length=max_length,
        )

        return BatchEncoding(
            data={
                "input_ids": source["input_ids"],
                "attention_mask": source["attention_mask"],
                "labels": targets["input_ids"],
                "decoder_attention_mask": targets["attention_mask"],
            }
        )


class CLMDataModule(DataModule):
    """Pytorch-lightning-style data module for CLM dataset."""

    def __init__(
        self, dataset_args: Dict[str, Union[float, str, int]], tokenizer: AutoTokenizer
    ) -> None:
        """
        Initialize the data module.

        Args:
            dataset_args: dictionary containing the metadata for the lightning data module creation.
            tokenizer: tokenizer to be used in the module.
        """
        super().__init__(dataset_args, tokenizer)

        self.load()

    def tokenize_function(
        self, examples: Dict[str, Union[int, slice]]
    ) -> BatchEncoding:
        """Tokenize the given examples.

        Args:
            examples: list of examples.
        Returns:
            tokenized examples.
        """

        tokenized_data = super().tokenize_function(examples)

        tokenized_data["labels"] = tokenized_data["input_ids"].copy()

        return tokenized_data


class PLMDataModule(DataModule):
    """Pytorch-lightning-style data module for PLM dataset."""

    def __init__(
        self, dataset_args: Dict[str, Union[float, str, int]], tokenizer: AutoTokenizer
    ) -> None:
        """Initialize the data module.

        Args:
            dataset_args: dictionary containing the metadata for the lightning data module creation.
            tokenizer: tokenizer to be used in the module.
        """
        super().__init__(dataset_args, tokenizer)

        self.data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=self.tokenizer,  # type: ignore
            plm_probability=self.dataset_args["plm_probability"],
            max_span_length=self.dataset_args["max_span_length"],
        )

        self.load()
