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
"""Moses VAE training pipeline."""
import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict

from guacamol_baselines.moses_baselines.vae_train import main

from ...core import TrainingPipelineArguments
from ..core import MosesTrainingArguments, MosesTrainingPipeline

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MosesVAETrainingPipeline(MosesTrainingPipeline):
    """Moses VAE training pipelines."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
    ) -> None:
        """Generic training function for Moses VAE training.

        Args:
            training_args: training arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.
        """
        params = {**training_args, **model_args, **dataset_args}

        os.makedirs(os.path.dirname(params["model_save"]), exist_ok=True)
        os.makedirs(os.path.dirname(params["log_file"]), exist_ok=True)
        os.makedirs(os.path.dirname(params["config_save"]), exist_ok=True)
        os.makedirs(os.path.dirname(params["vocab_save"]), exist_ok=True)

        args = argparse.Namespace(**params)
        main(args)


@dataclass
class MosesVAEModelArguments(TrainingPipelineArguments):
    """Arguments related to Moses VAE model."""

    __name__ = "model_args"

    q_cell: str = field(default="gru", metadata={"help": "Encoder rnn cell type."})
    q_bidir: bool = field(
        default=True,
        metadata={"help": "Whether to add second direction in the encoder."},
    )
    q_d_h: int = field(default=256, metadata={"help": "Encoder h dimensionality."})
    q_n_layers: int = field(default=1, metadata={"help": "Encoder number of layers."})
    q_dropout: float = field(default=0.5, metadata={"help": "Encoder layers dropout."})
    d_cell: str = field(default="gru", metadata={"help": "Decoder rnn cell type."})
    d_n_layers: int = field(default=3, metadata={"help": "Decoder number of layers."})
    d_dropout: float = field(default=0, metadata={"help": "Decoder layers dropout"})
    d_z: int = field(default=128, metadata={"help": "Latent vector dimensionality"})
    d_d_h: int = field(default=512, metadata={"help": "Latent vector dimensionality"})
    freeze_embeddings: bool = field(
        default=False, metadata={"help": "If to freeze embeddings while training"}
    )


@dataclass
class MosesVAETrainingArguments(MosesTrainingArguments):
    """Arguments related to Moses VAE training."""

    n_batch: int = field(default=512, metadata={"help": "Batch size."})
    grad_clipping: int = field(
        default=50, metadata={"help": "Gradients clipping size."}
    )
    kl_start: int = field(
        default=0, metadata={"help": "Epoch to start change kl weight from."}
    )
    kl_w_start: float = field(default=0, metadata={"help": "Initial kl weight value."})
    kl_w_end: float = field(default=0.05, metadata={"help": "Maximum kl weight value."})
    lr_start: float = field(default=3 * 1e-4, metadata={"help": "Initial lr value."})
    lr_n_period: int = field(
        default=10, metadata={"help": "Epochs before first restart in SGDR."}
    )
    lr_n_restarts: int = field(
        default=10, metadata={"help": "Number of restarts in SGDR."}
    )
    lr_n_mult: int = field(
        default=1, metadata={"help": "Mult coefficient after restart in SGDR."}
    )
    lr_end: float = field(
        default=3 * 1e-4, metadata={"help": "Maximum lr weight value."}
    )
    n_last: int = field(
        default=1000, metadata={"help": "Number of iters to smooth loss calc."}
    )
    n_jobs: int = field(default=1, metadata={"help": "Number of threads."})
    n_workers: int = field(default=1, metadata={"help": "Number of workers."})
