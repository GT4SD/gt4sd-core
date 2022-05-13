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
"""Language modeling training utilities."""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import sentencepiece as _sentencepiece
from pytorch_lightning import LightningDataModule, LightningModule

from ...core import TrainingPipelineArguments
from ..core import PyTorchLightningTrainingPipeline
from .lm_datasets import CGMDataModule, CLMDataModule, MLMDataModule, PLMDataModule
from .models import LM_MODULE_FACTORY, CGMModule, CLMModule, MLMModule, PLMModule

# sentencepiece has to be loaded before lightning to avoid segfaults
_sentencepiece

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LanguageModelingTrainingPipeline(PyTorchLightningTrainingPipeline):
    """Language modeling training pipelines."""

    def get_data_and_model_modules(
        self,
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
        **kwargs,
    ) -> Tuple[LightningDataModule, LightningModule]:
        """Get data and model modules for training.

        Args:
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.

        Returns:
            the data and model modules.
        """

        if (
            model_args["model_config_name"] is None
            and model_args["model_name_or_path"] is None
        ):
            raise ValueError("Model config or model name/path should be provided")

        if (
            model_args["model_config_name"] is not None
            and model_args["model_name_or_path"] is not None
        ):
            logger.warning(
                "Config name is omitted. Start fine-tuning using {}".format(
                    model_args["model_name_or_path"]
                )
            )

        if model_args["tokenizer"] is None:

            if model_args["model_name_or_path"] is not None:
                model_args["tokenizer"] = model_args["model_name_or_path"]
            else:
                model_args["tokenizer"] = model_args["model_config_name"]
                logger.warning(
                    "{} tokenizer is going to be used in the training".format(
                        model_args["tokenizer"]
                    )
                )

        logger.info(f"Model arguments: {model_args}")
        logger.info(f"Dataset arguments: {dataset_args}")

        if model_args["type"] == "mlm":
            data_module, model_module = self.get_mlm_modules(model_args, dataset_args)
        elif model_args["type"] == "clm":
            data_module, model_module = self.get_clm_modules(model_args, dataset_args)  # type: ignore
        elif model_args["type"] == "plm":
            data_module, model_module = self.get_plm_modules(model_args, dataset_args)  # type: ignore
        elif model_args["type"] == "cgm":
            data_module, model_module = self.get_cgm_modules(model_args, dataset_args)  # type: ignore
        else:
            raise ValueError(f"LM training type {model_args['type']} not supported")

        model_module.model.resize_token_embeddings(len(data_module.tokenizer))  # type: ignore

        return data_module, model_module

    def get_mlm_modules(
        self,
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
    ) -> Tuple[MLMDataModule, MLMModule]:
        """Get model and data module for clm.

        Args:
            model_args: dictionary containing all the parameters for the mode configuration.
            dataset_args: dictionary containing all the necessary parameters for the dataset creation.
        Returns:
            model and data module for clm.
        """

        model_module = MLMModule(model_args)
        data_module = MLMDataModule(dataset_args, tokenizer=model_module.tokenizer)

        return data_module, model_module

    def get_clm_modules(
        self,
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
    ) -> Tuple[CLMDataModule, CLMModule]:
        """Get model and data module for clm.

        Args:
            model_args: dictionary containing all the parameters for the mode configuration.
            dataset_args: dictionary containing all the necessary parameters for the dataset creation.
        Returns:
            model and data module for clm.
        """

        model_module = CLMModule(model_args)
        data_module = CLMDataModule(dataset_args, tokenizer=model_module.tokenizer)

        return data_module, model_module

    def get_plm_modules(
        self,
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
    ) -> Tuple[PLMDataModule, PLMModule]:
        """Get model and data module for plm.

        Args:
            model_args: dictionary containing all the parameters for the mode configuration.
            dataset_args: dictionary containing all the necessary parameters for the dataset creation.
        Returns:
            model and data module for plm.
        """

        model_module = PLMModule(model_args)
        data_module = PLMDataModule(dataset_args, tokenizer=model_module.tokenizer)

        return data_module, model_module

    def get_cgm_modules(
        self,
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
    ) -> Tuple[CGMDataModule, CGMModule]:
        """Get model and data module for Conditional Generation model.

        Args:
            model_args: dictionary containing all the parameters for the mode configuration.
            dataset_args: dictionary containing all the necessary parameters for the dataset creation.
        Returns:
            model and data module for plm.
        """

        model_module = CGMModule(model_args)
        data_module = CGMDataModule(dataset_args, tokenizer=model_module.tokenizer)

        return data_module, model_module


@dataclass
class LanguageModelingModelArguments(TrainingPipelineArguments):
    """
    Arguments pertaining to which model/config we are going to fine-tune, or train from scratch.
    """

    __name__ = "model_args"

    type: str = field(
        metadata={"help": "The language modeling type, for example mlm."},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization, for example bert-base-uncased."
        },
    )
    model_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path."},
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the tokenizer to be used, default: tokenizer of utilizing model."
        },
    )
    lr: float = field(
        default=2e-5,
        metadata={"help": "The learning rate."},
    )
    lr_decay: float = field(
        default=0.5,
        metadata={"help": "The learning rate decay."},
    )
    cache_dir: Union[str, None] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co."
        },
    )


@dataclass
class LanguageModelingDataArguments(TrainingPipelineArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    __name__ = "dataset_args"

    train_file: str = field(
        metadata={
            "help": "The input training data file (a text file), for example path/to/file."
        }
    )
    validation_file: str = field(
        metadata={
            "help": "The input evaluation data file to evaluate the perplexity on (a text file), for example path/to/file."
        },
    )
    max_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    plm_probability: float = field(
        default=0.16666,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for "
            "permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5,
        metadata={
            "help": "Maximum length of a span of masked tokens for permutation language modeling."
        },
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss."},
    )


@dataclass
class LanguageModelingSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to LM trainer."""

    __name__ = "saving_args"

    hf_model_path: str = field(
        metadata={"help": "Path to the converted HF model."},
        default="/tmp/gt4sd_lm_saving_tmp",
    )
    training_type: Optional[str] = field(
        metadata={
            "help": f"Training type of the converted model, supported types: {', '.join(LM_MODULE_FACTORY.keys())}."
        },
        default=None,
    )
    model_name_or_path: Optional[str] = field(
        metadata={
            "help": "Model name or path.",
        },
        default=None,
    )
    ckpt: Optional[str] = field(metadata={"help": "Path to checkpoint."}, default=None)
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Tokenizer name or path. If not provided defaults to model_name_or_path."
        },
    )
