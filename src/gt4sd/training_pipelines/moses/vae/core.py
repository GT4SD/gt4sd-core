import argparse
import logging
from dataclasses import dataclass, field
from typing import Any, Dict

from guacamol_baselines.moses_baselines.vae_train import main

from ...core import TrainingPipelineArguments
<<<<<<< HEAD
from ..core import MosesTrainingArguments, MosesTrainingPipeline
=======
from ..core import MosesTrainingPipeline
>>>>>>> refs/rewritten/chore-merging-with-remote-

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MosesVAETrainingPipeline(MosesTrainingPipeline):
    """Moses VAE training pipelines."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        common_args: Dict[str, Any],
    ) -> None:  # type: ignore
        params = {**common_args, **model_args, **training_args}
        parser = argparse.ArgumentParser()
        for k, v in params.items():
            parser.add_argument("--" + k, default=v)
        args = parser.parse_known_args()[0]
        main(args)


@dataclass
class MosesVAEModelArguments(TrainingPipelineArguments):
<<<<<<< HEAD
    """Arguments related to MOSES VAE model."""
=======
    """Arguments related to VAE Trainer."""
>>>>>>> refs/rewritten/chore-merging-with-remote-

    __name__ = "model_args"

    q_cell: str = field(default="gru", metadata={"help": "Encoder rnn cell type."})
    q_bidir: int = field(
        default=1, metadata={"help": "If to add second direction to encoder."}
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
<<<<<<< HEAD
class MosesVAETrainingArguments(MosesTrainingArguments):
    """Arguments related to MOSES VAE training."""
=======
class MosesVAETrainingArguments(TrainingPipelineArguments):
    """Arguments related to VAE Trainer."""

    __name__ = "training_args"
>>>>>>> refs/rewritten/chore-merging-with-remote-

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
<<<<<<< HEAD
    n_workers: int = field(default=1, metadata={"help": "Number of workers."})
=======
    n_workers: int = field(default=1, metadata={"help": ""})
>>>>>>> refs/rewritten/chore-merging-with-remote-
