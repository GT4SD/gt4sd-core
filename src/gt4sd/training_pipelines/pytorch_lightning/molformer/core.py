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
"""Molformer training utilities."""

import logging
from dataclasses import Dict, Union, dataclass, field, Optional, List, Tuple

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
from pytorch_lightning import LightningDataModule, LightningModule

from ..core import TrainingPipeline, TrainingPipelineArguments

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MolformerTrainingPipeline(TrainingPipeline):
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

        if model_args["type"] not in self.training_types:
            raise ValueError(f"Training type {model_args['type']} is not supported.")

        data_module, model_module = self.modules_getter[model_args["type"]](
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

        train_config = {
            "batch_size": model_args["n_batch"],
            "num_workers": model_args["n_workers"],
            "pin_memory": True,
        }

        data_module = MoleculeModule(
            dataset_args["max_len"], dataset_args["datapath"], train_config
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

        bert_vocab_path = (
            importlib_resources.files("gt4sd_molformer") / "finetune/bert_vocab.txt"
        )

        tokenizer = MolTranBertTokenizer(bert_vocab_path)

        data_module = PropertyPredictionDataModule(dataset_args, tokenizer)
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
        data_module = RegressionDataModule(dataset_args, tokenizer)

        model_module = RegressionLightningModule(model_args, tokenizer)

        return data_module, model_module


@dataclass
class MolformerDataArguments(TrainingPipelineArguments):
    """Data arguments related to Molformer trainer."""

    __name__ = "dataset_args"

    datapath: str = field(
        metadata={
            "help": "Path to the dataset."
            "The dataset should follow the directory structure as described in https://github.com/txie-93/Molformer"
        },
    )
    n_batch: int = field(default=512, metadata={"help": "Batch size."})
    train_size: Optional[int] = field(
        default=None, metadata={"help": "Number of training data to be loaded."}
    )
    val_size: Optional[int] = field(
        default=None, metadata={"help": "Number of validation data to be loaded."}
    )
    test_size: Optional[int] = field(
        default=None, metadata={"help": "Number of testing data to be loaded."}
    )
    max_len: int = field(default=100, metadata={"help": "Max of length of SMILES."})
    train_load: Optional[str] = field(
        default=None, metadata={"help": "Where to load the model."}
    )
    n_workers: Optional[int] = field(default=1, metadata={"help": "Number of workers."})


@dataclass
class MolformerModelArguments(TrainingPipelineArguments):
    """Model arguments related to Molformer trainer."""

    __name__ = "model_args"

    type: str = field(
        metadata={
            "help": "The training type, for example pretraining or classification."
        },
    )

    n_head: int = field(default=8, metadata={"help": "GPT number of heads."})
    n_laye: int = field(default=12, metadata={"help": "GPT number of layers."})
    q_dropout: float = field(default=0.5, metadata={"help": "Encoder layers dropout."})
    d_dropout: float = field(default=0.1, metadata={"help": "Decoder layers dropout."})
    n_embd: int = field(default=768, metadata={"help": "Latent vector dimensionality."})
    fc_h: int = field(
        default=512, metadata={"help": "Fully connected hidden dimensionality."}
    )
    unlike_alpha: float = field(
        default=1.0, metadata={"help": "unlikelihood loss alpha weight."}
    )
    dropout: float = field(
        default=0.1, metadata={"help": "Dropout used in finetuning."}
    )
    dims: List[int] = field(default_factory=lambda: [])
    num_classes: Optional[int] = field(default=None)

    vocab_load: Optional[str] = field(
        default=None, metadata={"help": "Where to load the vocab."}
    )
    n_samples: Optional[int] = field(
        default=None, metadata={"help": "Number of samples to sample."}
    )
    dataset_name: str = field(default="sol")
    measure_name: str = field(default="measure")
    data_root: str = field(
        default="/dccstor/medscan7/smallmolecule/runs/ba-predictor/small-data/affinity"
    )
    train_dataset_length: Optional[int] = field(default=None)
    eval_dataset_length: Optional[int] = field(default=None)
    desc_skip_connection: Optional[bool] = field(default=False)

    finetune_path: str = field(
        default="", metadata={"help": "path to  trainer file to continue training."}
    )
    restart_path: str = field(
        default="", metadata={"help": "path to  trainer file to continue training."}
    )
    from_scratch: bool = field(
        default=False, metadata={"help": "train on qm9 from scratch."}
    )
    unlikelihood: bool = field(
        default=False, metadata={"help": "use unlikelihood loss with gpt pretrain."}
    )
    lr_start: float = field(default=3 * 1e-4, metadata={"help": "Initial lr value."})
    lr_end: float = field(
        default=3 * 1e-4, metadata={"help": "Maximum lr weight value."}
    )
    lr_multiplier: int = field(default=1, metadata={"help": "lr weight multiplier."})
    n_last: int = field(
        default=1000, metadata={"help": "Number of iters to smooth loss calc."}
    )
    seed: int = field(default=12345, metadata={"help": "Seed."})
    gen_save: Optional[str] = field(
        default=None, metadata={"help": "Where to save the gen molecules."}
    )
    val_load: Optional[str] = field(
        default=None, metadata={"help": "Where to load the model."}
    )
    beam_size: int = field(default=0, metadata={"help": "Number of beams to generate."})
    num_seq_returned: int = field(
        default=0,
        metadata={"help": "number of beams to be returned (must be <= beam_size."},
    )
    min_len: int = field(
        default=1, metadata={"help": "minimum length to be generated."}
    )
    nucleus_thresh: float = field(
        default=0.9, metadata={"help": "nucleus sampling threshold."}
    )
    data_path: str = field(default="", metadata={"help": "path to pubchem file."})
    pretext_size: int = field(
        default=0, metadata={"help": "number of k-mers to pretext."}
    )
    model_save_dir: str = field(
        default="./models_dump/",
        metadata={"help": "Where to save the models/log/config/vocab."},
    )
    model_save: str = field(
        default="model.pt", metadata={"help": "Where to save the model."}
    )
    log_file: Optional[str] = field(
        default=None, metadata={"help": "Where to save the log."}
    )
    tb_loc: Optional[str] = field(
        default=None, metadata={"help": "Where to save the tensorflow location."}
    )
    config_save: Optional[str] = field(
        default=None, metadata={"help": "Where to save the config."}
    )
    vocab_save: Optional[str] = field(
        default=None, metadata={"help": "Where to save the vocab."}
    )
    debug: bool = field(
        default=False, metadata={"help": "do not erase cache at end of program."}
    )
    fast_dev_run: bool = field(
        default=False,
        metadata={
            "help": "This flag runs a “unit test” by running n if set to n (int) else 1 if set to True training and validation batch(es)."
        },
    )
    freeze_model: bool = field(
        default=False,
        metadata={"help": "freeze weights of bert model during fine tuning."},
    )
    resume: bool = field(default=False, metadata={"help": "Resume from a saved model."})
    rotate: bool = field(
        default=False, metadata={"help": "use rotational relative embedding."}
    )
    model_load: Optional[str] = field(
        default=None, metadata={"help": "Where to load the model."}
    )
    root_dir: str = field(default=".", metadata={"help": "location of root dir."})
    config_load: Optional[str] = field(
        default=None, metadata={"help": "Where to load the config."}
    )

    model_arch: Optional[str] = field(
        default=None, metadata={"help": "used to teack model arch in params."}
    )
    num_feats: int = field(
        default=32, metadata={"help": "number of random features for FAVOR+."}
    )
    pooling_mode: str = field(
        default="cls", metadata={"help": "type of pooling to use."}
    )
    fold: int = field(default=0, metadata={"help": "number of folds for fine tuning."})


@dataclass
class MolformerTrainingArguments(TrainingPipelineArguments):
    """Training arguments related to Molformer trainer."""

    __name__ = "training_args"

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
    val_check_interval: int = field(
        default=50000, metadata={"help": " How often to check the validation set."}
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
