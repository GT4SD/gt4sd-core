import logging
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict
from guacamol_baselines.moses_baselines.aae_train import main
from ...core import TrainingPipelineArguments
from ..core import MosesTrainingPipeline

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MosesAAETrainingPipeline(MosesTrainingPipeline):
    """Moses AAE training pipelines."""

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
        print(vars(args))
        main(args)


@dataclass
class MosesAAEModelArguments(TrainingPipelineArguments):
    """Arguments related to AAE Trainer."""

    __name__ = "model_args"

    embedding_size: int = field(default=32, metadata={"help": ""})
    encoder_hidden_size: int = field(default=512, metadata={"help": ""})
    encoder_num_layers: int = field(default=1, metadata={"help": ""})
    encoder_bidirectional: bool = field(default=True, metadata={"help": ""})
    encoder_dropout: float = field(default=0, metadata={"help": ""})
    decoder_hidden_size: int = field(default=512, metadata={"help": ""})
    decoder_num_layers: int = field(default=2, metadata={"help": ""})
    decoder_dropout: float = field(default=0, metadata={"help": ""})
    latent_size: float = field(default=128, metadata={"help": ""})
    discriminator_layers: list = field(
        default_factory=lambda: [640, 256], metadata={"help": ""}
    )


@dataclass
class MosesAAETrainingArguments(TrainingPipelineArguments):
    """Arguments related to AAE Trainer."""

    __name__ = "training_args"

    pretrain_epochs: int = field(default=0, metadata={"help": ""})
    train_epochs: int = field(default=120, metadata={"help": ""})
    n_batch: int = field(default=512, metadata={"help": ""})
    lr: float = field(default=1e-3, metadata={"help": ""})
    step_size: int = field(default=20, metadata={"help": ""})
    gamma: float = field(default=0.5, metadata={"help": ""})
    n_jobs: int = field(default=1, metadata={"help": ""})
    n_workers: int = field(default=1, metadata={"help": ""})
    discriminator_steps: int = field(default=1, metadata={"help": ""})
    weight_decay: int = field(default=0, metadata={"help": ""})
