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

"""Transformers pretrained model to SentenceTransformer model converter."""


import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import cast

from sentence_transformers import SentenceTransformer, __version__, models

from .argument_parser import ArgumentParser, DataClassType


@dataclass
class TransformersToSentenceTransformersArguments:
    """Transformers to Sentence Transformers converter arguments."""

    __name__ = "hf_to_st_converter_args"

    model_name_or_path: str = field(
        metadata={"help": "HF model name or path."},
    )
    pooling: str = field(
        metadata={
            "help": "Comma separated pooling modes. Supported types: cls, max, mean, mean_sqrt."
        },
    )
    output_path: str = field(
        metadata={"help": "Path to the converted model."},
    )


def main() -> None:
    """Convert HF pretrained model to SentenceTransformer.

    Create a SentenceTransformer model having a given HF model as
    word embedding model plus an optional pooling layer. We can
    also concatenate multiple poolings together.

    Parsing from the command line the following parameters:
        - HF pretrained model to be used as word embedding model.
        - the pooling mode (more than one can be provided as a list), the implemented
            options are "cls", "max", "mean", "mean" and "sqrt".
        - path to save the generated SentenceTransformer model.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    arguments = ArgumentParser(
        cast(DataClassType, TransformersToSentenceTransformersArguments)
    ).parse_args_into_dataclasses(return_remaining_strings=True)[0]

    model_name_or_path = arguments.model_name_or_path
    pooling = [
        polling_argument.strip() for polling_argument in arguments.pooling.split(",")
    ]
    output_path = arguments.output_path

    word_embedding_model = models.Transformer(model_name_or_path)

    pooling_mode_cls_token = False
    pooling_mode_max_tokens = False
    pooling_mode_mean_tokens = False
    pooling_mode_mean_sqrt_len_tokens = False

    if "cls" in pooling:
        pooling_mode_cls_token = True
    if "max" in pooling:
        pooling_mode_max_tokens = True
    if "mean" in pooling:
        pooling_mode_mean_tokens = True
    if "mean_sqrt" in pooling:
        pooling_mode_mean_sqrt_len_tokens = True

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode=None,
        pooling_mode_cls_token=pooling_mode_cls_token,
        pooling_mode_max_tokens=pooling_mode_max_tokens,
        pooling_mode_mean_tokens=pooling_mode_mean_tokens,
        pooling_mode_mean_sqrt_len_tokens=pooling_mode_mean_sqrt_len_tokens,
    )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    model.save(output_path)

    config_filepath = os.path.join(output_path, "config.json")
    if os.path.exists(config_filepath):
        with open(config_filepath) as fp:
            config = json.load(fp)
            config["__version__"] = __version__
        with open(config_filepath, "wt") as fp:
            json.dump(config, fp, indent=2)


if __name__ == "__main__":
    main()
