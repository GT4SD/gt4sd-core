#!/usr/bin/env python
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

"""Pytorch lightning checkpoint to HF transformers converter."""


import logging
import sys
from typing import cast

from transformers import Trainer, TrainingArguments

from ..training_pipelines.pytorch_lightning.language_modeling.core import (
    LanguageModelingSavingArguments,
)
from ..training_pipelines.pytorch_lightning.language_modeling.models import (
    LM_MODULE_FACTORY,
)
from .argument_parser import ArgumentParser, DataClassType


def convert_pl_to_hf(arguments: LanguageModelingSavingArguments) -> None:
    """Method to convert pytorch lightning checkpoint to HF transformers model.

    Args:
        arguments: a LanguageModelingSavingArguments instance that contains all the
                   needed arguments for conversion.
    Raises:
        ValueError: in case the provided training type is not supported.
    """

    if arguments.training_type is None:
        raise ValueError(
            "training_type is required for saving from pytorch lightning checkpoing."
        )

    if arguments.model_name_or_path is None:
        raise ValueError(
            "model_name_or_path is required for saving from pytorch lightning checkpoing."
        )

    if arguments.ckpt is None:
        raise ValueError(
            "ckpt is required for saving from pytorch lightning checkpoing."
        )

    training_type = arguments.training_type
    model_name_or_path = arguments.model_name_or_path
    tokenizer_name_or_path = arguments.tokenizer_name_or_path
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path
    ckpt = arguments.ckpt
    output_path = arguments.hf_model_path

    if training_type not in LM_MODULE_FACTORY:
        ValueError(
            f"LM training type {training_type} is not supported. Supported types: {', '.join(LM_MODULE_FACTORY.keys())}."
        )
    model_module_class = LM_MODULE_FACTORY[training_type]

    model_module = model_module_class.load_from_checkpoint(
        ckpt,
        model_args={
            "model_name_or_path": model_name_or_path,
            "tokenizer": tokenizer_name_or_path,
        },
    )

    trainer = Trainer(
        model=model_module.model,
        tokenizer=model_module.tokenizer,
        args=TrainingArguments(output_dir=output_path),
    )
    trainer.save_model()  # type:ignore


def main() -> None:
    """Convert pytorch lightning checkpoint to HF transformers model.

    Parsing from the command line the following parameters:
        - training type of the given checkpoint.
        - model name or path, a HF's model name.
        - tokenizer name or path, a HF's tokenizer name.
        - path of the checkpoint.
        - path where the HF model will be saved.

    Raises:
        ValueError: in case the provided training type is not supported.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    arguments = ArgumentParser(
        cast(DataClassType, LanguageModelingSavingArguments)
    ).parse_args_into_dataclasses(return_remaining_strings=True)[0]

    convert_pl_to_hf(arguments)


if __name__ == "__main__":
    main()
