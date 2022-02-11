#!/usr/bin/env python

"""Pytorch lightning checkpoint to HF transformers converter."""


import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, cast

from transformers import Trainer, TrainingArguments

from ..training_pipelines.pytorch_lightning.language_modeling.models import (
    LM_MODULE_FACTORY,
)
from .argument_parser import ArgumentParser, DataClassType


@dataclass
class PyTorchLightningToTransformersArguments:
    """PyTorchLightning to Transformers converter arguments."""

    __name__ = "pl_to_hf_converter_args"

    training_type: str = field(
        metadata={
            "help": f"Training type of the converted model, supported types: {', '.join(LM_MODULE_FACTORY.keys())}."
        },
    )
    model_name_or_path: str = field(
        metadata={"help": "Model name or path."},
    )
    ckpt: str = field(
        metadata={"help": "Path to checkpoint."},
    )
    output_path: str = field(
        metadata={"help": "Path to the converted model."},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Tokenizer name or path. If not provided defaults to model_name_or_path."
        },
    )


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
        cast(DataClassType, PyTorchLightningToTransformersArguments)
    ).parse_args_into_dataclasses(return_remaining_strings=True)[0]

    training_type = arguments.training_type
    model_name_or_path = arguments.model_name_or_path
    tokenizer_name_or_path = arguments.tokenizer_name_or_path
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path
    ckpt = arguments.ckpt
    output_path = arguments.output_path

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
    trainer.save_model()


if __name__ == "__main__":
    main()
