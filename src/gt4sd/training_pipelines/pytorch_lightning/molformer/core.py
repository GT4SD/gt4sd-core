#
# MIT License
#
# Copyright (c) 2023 GT4SD team
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
"""Molformer training utilities."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import sentencepiece as _sentencepiece
import torch as _torch
import tensorflow as _tensorflow
import importlib_resources
from gt4sd_molformer.finetune.finetune_pubchem_light import (
    LightningModule as RegressionLightningModule,
)
from gt4sd_molformer.finetune.finetune_pubchem_light import (
    PropertyPredictionDataModule as RegressionDataModule,
)
from gt4sd_molformer.finetune.finetune_pubchem_light_classification import (
    LightningModule as ClassificationLightningModule,
)
from gt4sd_molformer.finetune.finetune_pubchem_light_classification import (
    PropertyPredictionDataModule as ClassificationDataModule,
)
from gt4sd_molformer.finetune.finetune_pubchem_light_classification_multitask import (
    MultitaskModel,
    PropertyPredictionDataModule,
)
from gt4sd_molformer.finetune.ft_tokenizer.ft_tokenizer import MolTranBertTokenizer
from gt4sd_molformer.training.train_pubchem_light import (
    LightningModule as PretrainingModule,
)
from gt4sd_molformer.training.train_pubchem_light import MoleculeModule
from gt4sd_trainer.hf_pl.pytorch_lightning_trainer import (
    PyTorchLightningTrainingPipeline,
)
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities import seed

from ...core import TrainingPipelineArguments

# imports that have to be loaded before lightning to avoid segfaults
_sentencepiece
_tensorflow
_torch

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MolformerTrainingPipeline(PyTorchLightningTrainingPipeline):
    """Molformer training pipelines for crystals."""

    def __init__(
        self,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.modules_getter = {
            "pretraining": self.get_pretraining_modules,
            "classification": self.get_classification_modules,
            "multitask_classification": self.get_multitask_classification_modules,
            "regression": self.get_regression_modules,
        }

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

        if model_args["type"] not in self.modules_getter:
            raise ValueError(f"Training type {model_args['type']} is not supported.")

        # alignments with gt4sd-molformer
        model_args["run_id"] = datetime.now().strftime("%m%d%Y%H%M")
        dataset_args["n_embd"] = model_args["n_embd"]
        model_args["dataset_names"] = "valid test".split()  # type: ignore
        model_args["dataset_name"] = dataset_args["dataset_name"]
        model_args["measure_name"] = dataset_args["measure_name"]
        model_args["mode"] = model_args["pooling_mode"]
        del model_args["pooling_mode"]

        seed.seed_everything(model_args["seed"])  # type: ignore

        data_module, model_module = self.modules_getter[model_args["type"]](  # type: ignore
            model_args, dataset_args
        )

        return data_module, model_module

    def get_pretraining_modules(
        self,
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
    ) -> Tuple[LightningDataModule, LightningModule]:
        """Get data and model modules for pretraing.

        Args:
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.

        Returns:
            the data and model modules.
        """

        model_args["max_len"] = dataset_args["max_len"]

        train_config = {
            "batch_size": dataset_args["batch_size"],
            "num_workers": dataset_args["num_workers"],
            "pin_memory": True,
        }

        data_module = MoleculeModule(
            dataset_args["max_len"], dataset_args["data_path"], train_config
        )
        data_module.setup()

        model_module = PretrainingModule(model_args, data_module.get_vocab())

        return data_module, model_module

    def get_classification_modules(
        self,
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
    ) -> Tuple[LightningDataModule, LightningModule]:
        """Get data and model modules for pretraing.

        Args:
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.

        Returns:
            the data and model modules.
        """

        # return data_module, model_module

        bert_vocab_path = (
            importlib_resources.files("gt4sd_molformer") / "finetune/bert_vocab.txt"
        )

        tokenizer = MolTranBertTokenizer(bert_vocab_path)
        data_module = ClassificationDataModule(dataset_args, tokenizer)

        if model_args["pretrained_path"] is not None:
            model_module = ClassificationLightningModule(
                model_args, tokenizer
            ).load_from_checkpoint(
                model_args["pretrained_path"],
                strict=False,
                config=model_args,
                tokenizer=tokenizer,
                vocab=len(tokenizer.vocab),
            )
        else:
            model_module = ClassificationLightningModule(model_args, tokenizer)

        return data_module, model_module

    def get_multitask_classification_modules(
        self,
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
    ) -> Tuple[LightningDataModule, LightningModule]:
        """Get data and model modules for pretraing.

        Args:
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.

        Returns:
            the data and model modules.
        """

        model_args["num_tasks"] = len(dataset_args["measure_names"])  # type: ignore

        if model_args["num_tasks"] == 0:
            raise ValueError("Missing class names.")

        bert_vocab_path = (
            importlib_resources.files("gt4sd_molformer") / "finetune/bert_vocab.txt"
        )

        tokenizer = MolTranBertTokenizer(bert_vocab_path)

        data_module = PropertyPredictionDataModule(dataset_args, tokenizer)

        if model_args["pretrained_path"] is not None:
            model_module = MultitaskModel(model_args, tokenizer).load_from_checkpoint(
                model_args["pretrained_path"],
                strict=False,
                config=model_args,
                tokenizer=tokenizer,
                vocab=len(tokenizer.vocab),
            )
        else:
            model_module = MultitaskModel(model_args, tokenizer)

        return data_module, model_module

    def get_regression_modules(
        self,
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
    ) -> Tuple[LightningDataModule, LightningModule]:
        """Get data and model modules for pretraing.

        Args:
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.

        Returns:
            the data and model modules.
        """

        bert_vocab_path = (
            importlib_resources.files("gt4sd_molformer") / "finetune/bert_vocab.txt"
        )

        tokenizer = MolTranBertTokenizer(bert_vocab_path)

        if model_args["pretrained_path"] is not None:

            model_module = RegressionLightningModule(
                model_args, tokenizer
            ).load_from_checkpoint(
                model_args["pretrained_path"],
                strict=False,
                config=model_args,
                tokenizer=tokenizer,
                vocab=len(tokenizer.vocab),
            )

        else:
            model_module = RegressionLightningModule(model_args, tokenizer)

        data_module = RegressionDataModule(dataset_args, tokenizer)

        return data_module, model_module


@dataclass
class MolformerDataArguments(TrainingPipelineArguments):
    """Data arguments related to Molformer trainer."""

    __name__ = "dataset_args"

    batch_size: int = field(default=512, metadata={"help": "Batch size."})

    data_path: str = field(
        default="", metadata={"help": "Pretraining - path to the data file."}
    )

    max_len: int = field(default=100, metadata={"help": "Max of length of SMILES."})
    train_load: Optional[str] = field(
        default=None, metadata={"help": "Where to load the model."}
    )
    num_workers: Optional[int] = field(
        default=1, metadata={"help": "Number of workers."}
    )
    dataset_name: str = field(
        default="sol",
        metadata={
            "help": "Finetuning - Name of the dataset to be found in the data root directory."
        },
    )
    measure_name: str = field(
        default="measure",
        metadata={"help": "Finetuning - Measure name to be used as groundtruth."},
    )
    data_root: str = field(
        default="my_data_root",
        metadata={"help": "Finetuning - Data root for the dataset."},
    )
    train_dataset_length: Optional[int] = field(
        default=None, metadata={"help": "Finetuning - Length of training dataset."}
    )
    eval_dataset_length: Optional[int] = field(
        default=None, metadata={"help": "Finetuning - Length of evaluation dataset."}
    )
    aug: bool = field(default=False, metadata={"help": "aug."})
    measure_names: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Class names for multitask classification."},
    )


@dataclass
class MolformerModelArguments(TrainingPipelineArguments):
    """Model arguments related to Molformer trainer."""

    __name__ = "model_args"

    type: str = field(
        default="classification",
        metadata={
            "help": "The training type, for example pretraining or classification."
        },
    )

    n_head: int = field(default=8, metadata={"help": "GPT number of heads."})
    n_layer: int = field(default=12, metadata={"help": "GPT number of layers."})
    q_dropout: float = field(default=0.5, metadata={"help": "Encoder layers dropout."})
    d_dropout: float = field(default=0.1, metadata={"help": "Decoder layers dropout."})
    n_embd: int = field(default=768, metadata={"help": "Latent vector dimensionality."})
    fc_h: int = field(
        default=512, metadata={"help": "Fully connected hidden dimensionality."}
    )
    dropout: float = field(
        default=0.1, metadata={"help": "Dropout used in finetuning."}
    )
    dims: List[int] = field(default_factory=lambda: [])
    num_classes: Optional[int] = field(
        default=None, metadata={"help": "Finetuning - Number of classes"}
    )

    restart_path: str = field(
        default="", metadata={"help": "path to  trainer file to continue training."}
    )

    lr_start: float = field(default=3 * 1e-4, metadata={"help": "Initial lr value."})

    lr_multiplier: int = field(default=1, metadata={"help": "lr weight multiplier."})

    seed: int = field(default=12345, metadata={"help": "Seed."})

    min_len: int = field(
        default=1, metadata={"help": "minimum length to be generated."}
    )

    root_dir: str = field(default=".", metadata={"help": "location of root dir."})

    num_feats: int = field(
        default=32, metadata={"help": "number of random features for FAVOR+."}
    )
    pooling_mode: str = field(
        default="cls", metadata={"help": "type of pooling to use."}
    )
    fold: int = field(default=0, metadata={"help": "number of folds for fine tuning."})
    pretrained_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the base pretrained model."}
    )
    results_dir: str = field(
        default=".",
        metadata={"help": "Path to save evaluation results during training."},
    )
    debug: bool = field(default=False, metadata={"help": "Debug training"})


@dataclass
class MolformerTrainingArguments(TrainingPipelineArguments):
    """Training arguments related to Molformer trainer."""

    __name__ = "pl_trainer_args"

    accumulate_grad_batches: int = field(
        default=1,
        metadata={
            "help": "Accumulates grads every k batches or as set up in the dict."
        },
    )
    strategy: str = field(
        default="ddp",
        metadata={
            "help": "The accelerator backend to use (previously known as distributed_backend)."
        },
    )
    gpus: int = field(default=-1, metadata={"help": "number of gpus to use."})
    max_epochs: int = field(default=1, metadata={"help": "max number of epochs."})
    monitor: Optional[str] = field(
        default=None,
        metadata={"help": "Quantity to monitor in order to store a checkpoint."},
    )
    save_top_k: int = field(
        default=1,
        metadata={
            "help": "The best k models according to the quantity monitored will be saved."
        },
    )
    mode: str = field(
        default="min",
        metadata={"help": "Quantity to monitor in order to store a checkpoint."},
    )
    every_n_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of training steps between checkpoints."},
    )
    every_n_epochs: Optional[int] = field(
        default=None,
        metadata={"help": "Number of epochs between checkpoints."},
    )
    save_last: Optional[bool] = field(
        default=None,
        metadata={
            "help": "When True, always saves the model at the end of the epoch to a file last.ckpt"
        },
    )
    save_dir: Optional[str] = field(
        default="logs", metadata={"help": "Save directory for logs and output."}
    )
    basename: Optional[str] = field(
        default="lightning_logs", metadata={"help": "Experiment name."}
    )
    val_check_interval: float = field(
        default=1.0, metadata={"help": " How often to check the validation set."}
    )
    gradient_clip_val: float = field(
        default=50, metadata={"help": "Gradient clipping value."}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path/URL of the checkpoint from which training is resumed."},
    )


@dataclass
class MolformerSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to Molformer trainer."""

    __name__ = "saving_args"
