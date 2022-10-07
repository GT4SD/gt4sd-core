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
"""GFlowNet training utilities."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import sentencepiece as _sentencepiece
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from ....frameworks.gflownet.dataloader.data_module import (
    GFlowNetDataModule,
    GFlowNetTask,
)
from ....frameworks.gflownet.dataloader.dataset import GFlowNetDataset
from ....frameworks.gflownet.envs.graph_building_env import (
    GraphBuildingEnv,
    GraphBuildingEnvContext,
)
from ....frameworks.gflownet.loss import ALGORITHM_FACTORY
from ....frameworks.gflownet.ml.models import MODEL_FACTORY
from ....frameworks.gflownet.ml.module import GFlowNetModule
from ...core import TrainingPipelineArguments
from ..core import PytorchLightningTrainingArguments, PyTorchLightningTrainingPipeline

# sentencepiece has to be loaded before lightning to avoid segfaults
_sentencepiece

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GFlowNetTrainingPipeline(PyTorchLightningTrainingPipeline):
    """gflownet training pipelines."""

    def train(  # type: ignore
        self,
        pl_trainer_args: Dict[str, Any],
        model_args: Dict[str, Union[float, str, int]],
        dataset_args: Dict[str, Union[float, str, int]],
        dataset: GFlowNetDataset,
        environment: GraphBuildingEnv,
        context: GraphBuildingEnvContext,
        task: GFlowNetTask,
    ) -> None:
        """Generic training function for PyTorch Lightning-based training.

        Args:
            pl_trainer_args: pytorch lightning trainer arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.
            dataset: dataset to be used for training.
            environment: environment to be used for training.
            context: context to be used for training.
            task: task to be used for training.
        """

        logger.info(f"Trainer arguments: {pl_trainer_args}")

        if pl_trainer_args[
            "resume_from_checkpoint"
        ] is not None and not pl_trainer_args["resume_from_checkpoint"].endswith(
            ".ckpt"
        ):
            pl_trainer_args["resume_from_checkpoint"] = None

        pl_trainer_args["callbacks"] = {
            "model_checkpoint_callback": {"save_top_k": pl_trainer_args["save_top_k"]}
        }

        pl_trainer_args["callbacks"] = self.add_callbacks(pl_trainer_args["callbacks"])

        pl_trainer_args["logger"] = TensorBoardLogger(
            pl_trainer_args["save_dir"], name=pl_trainer_args["basename"]
        )

        trainer = Trainer(
            profiler=pl_trainer_args["profiler"],
            logger=pl_trainer_args["logger"],
            log_every_n_steps=pl_trainer_args["trainer_log_every_n_steps"],
            callbacks=pl_trainer_args["callbacks"],
            max_epochs=pl_trainer_args["epochs"],
            strategy=pl_trainer_args["strategy"],
            fast_dev_run=pl_trainer_args["development_mode"],
        )

        data_module, model_module = self.get_data_and_model_modules(
            model_args,
            dataset_args,
            pl_trainer_args,
            dataset,
            environment,
            context,
            task,
        )
        trainer.fit(model_module, data_module)

    # TODO: compatible signature with Pytorch Lightning training pipelines
    def get_data_and_model_modules(  # type: ignore
        self,
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
        pl_trainer_args: Dict[str, Any],
        dataset: GFlowNetDataset,
        environment: GraphBuildingEnv,
        context: GraphBuildingEnvContext,
        task: GFlowNetTask,
    ) -> Tuple[LightningDataModule, LightningModule]:
        """Get data and model modules for training.

        Args:
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.

        Returns:
            the data and model modules.
        """

        configuration = {**model_args, **dataset_args, **pl_trainer_args}
        if configuration["algorithm"] in ALGORITHM_FACTORY:
            algorithm = ALGORITHM_FACTORY[configuration["algorithm"]](
                configuration,
                environment,
                context,
            )
        else:
            raise ValueError(
                "Algorithm configuration is not given in the specified config file."
            )

        if configuration["model"] in MODEL_FACTORY:
            model = MODEL_FACTORY[configuration["model"]](
                configuration,
                context,
            )
        else:
            raise ValueError(
                "Models configuration is not given in the specified config file."
            )

        dm = GFlowNetDataModule(
            configuration=configuration,
            dataset=dataset,
            environment=environment,
            context=context,
            task=task,
            algorithm=algorithm,
            model=model,
        )
        dm.prepare_data()

        module = GFlowNetModule(
            configuration=configuration,
            dataset=dataset,
            environment=environment,
            context=context,
            task=task,
            algorithm=algorithm,
            model=model,
        )

        return dm, module


@dataclass
class GFlowNetPytorchLightningTrainingArguments(PytorchLightningTrainingArguments):
    """
    Arguments related to pytorch lightning trainer.
    """

    __name__ = "pl_trainer_args"

    strategy: Optional[str] = field(
        default="ddp", metadata={"help": "Training strategy."}
    )
    accumulate_grad_batches: int = field(
        default=1,
        metadata={
            "help": "Accumulates grads every k batches or as set up in the dict."
        },
    )

    trainer_log_every_n_steps: int = field(
        default=50,
        metadata={"help": "log every k steps."},
    )

    val_check_interval: int = field(
        default=5000, metadata={"help": " How often to check the validation set."}
    )
    save_dir: Optional[str] = field(
        default="logs", metadata={"help": "Save directory for logs and output."}
    )
    basename: Optional[str] = field(
        default="lightning_logs", metadata={"help": "Experiment name."}
    )
    gradient_clip_val: float = field(
        default=0.0, metadata={"help": "Gradient clipping value."}
    )
    limit_val_batches: int = field(
        default=500, metadata={"help": "How much of validation dataset to check."}
    )
    log_every_n_steps: int = field(
        default=500, metadata={"help": "How often to log within steps."}
    )
    max_epochs: int = field(
        default=3,
        metadata={"help": "Stop training once this number of epochs is reached."},
    )
    epochs: int = field(
        default=3,
        metadata={"help": "Stop training once this number of epochs is reached."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path/URL of the checkpoint from which training is resumed."},
    )
    gpus: Optional[int] = field(
        default=-1,
        metadata={"help": "Number of gpus to train on."},
    )

    monitor: Optional[str] = field(
        default=None,
        metadata={"help": "Quantity to monitor in order to store a checkpoint."},
    )
    save_last: Optional[bool] = field(
        default=None,
        metadata={
            "help": "When True, always saves the model at the end of the epoch to a file last.ckpt"
        },
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
    check_val_every_n_epoch: Optional[int] = field(
        default=5,
        metadata={"help": "Number of validation epochs between checkpoints."},
    )
    auto_lr_find: bool = field(
        default=True,
        metadata={
            "help": "Select whether to run a learning rate finder to try to optimize initial learning for faster convergence."
        },
    )
    profiler: Optional[str] = field(
        default="simple",
        metadata={
            "help": "To profile individual steps during training and assist in identifying bottlenecks."
        },
    )
    learning_rate: float = field(
        default=0.0001,
        metadata={"help": "The learning rate."},
    )

    test_output_path: Optional[str] = field(
        default="./test",
        metadata={
            "help": "Path where to save latent encodings and predictions for the test set when an epoch ends."
        },
    )
    num_workers: int = field(
        default=0,
        metadata={"help": "number of workers. Defaults to 1."},
    )

    log_dir: str = field(
        default="./log/",
        metadata={"help": "The directory to save logs."},
    )

    num_training_steps: int = field(
        default=1000,
        metadata={"help": "The number of training steps."},
    )

    validate_every: int = field(
        default=1000,
        metadata={"help": "The number of training steps between validation."},
    )

    seed: int = field(
        default=142857,
        metadata={"help": "The random seed."},
    )

    device: str = field(
        default="cpu",
        metadata={"help": "The device to use."},
    )
    development_mode: bool = field(
        default=False,
        metadata={"help": "Whether to run in development mode. "},
    )


@dataclass
class GFlowNetModelArguments(TrainingPipelineArguments):
    """
    Arguments related to model.
    """

    __name__ = "model_args"

    algorithm: str = field(
        default="trajectory_balance",
        metadata={"help": "The algorithm to use for training the model. "},
    )
    context: str = field(  # type: ignore
        default=None,
        metadata={"help": "The environment context to use for training the model. "},
    )
    environment: str = field(  # type: ignore
        default=None,
        metadata={"help": "The environment to use for training the model. "},
    )
    model: str = field(
        default="graph_transformer_gfn",
        metadata={"help": "The model to use for training the model. "},
    )
    sampling_model: str = field(
        default="graph_transformer_gfn",
        metadata={"help": "The model used to generate samples. "},
    )
    task: str = field(
        default="qm9",
        metadata={"help": "The task to use for training the model. "},
    )

    bootstrap_own_reward: bool = field(
        default=False,
        metadata={"help": "Whether to bootstrap the own reward. "},
    )

    num_emb: int = field(
        default=128,
        metadata={"help": "The number of embeddings. "},
    )

    num_layers: int = field(
        default=4,
        metadata={"help": "The number of layers. "},
    )

    tb_epsilon: float = field(
        default=1e-10,
        metadata={"help": "The epsilon. "},
    )

    illegal_action_logreward: float = field(
        default=-50.0,
        metadata={"help": "The illegal action log reward. "},
    )

    reward_loss_multiplier: float = field(
        default=1.0,
        metadata={"help": "The reward loss multiplier. "},
    )

    temperature_sample_dist: str = field(
        default="uniform",
        metadata={"help": "The temperature sample distribution. "},
    )

    temperature_dist_params: str = field(
        default="(.5, 32)",
        metadata={"help": "The temperature distribution parameters. "},
    )

    weight_decay: float = field(
        default=1e-8,
        metadata={"help": "The weight decay. "},
    )
    momentum: float = field(
        default=0.9,
        metadata={"help": "The momentum. "},
    )

    adam_eps: float = field(
        default=1e-8,
        metadata={"help": "The adam epsilon. "},
    )

    lr_decay: float = field(
        default=20000,
        metadata={"help": "The learning rate decay steps. "},
    )

    z_lr_decay: float = field(
        default=20000,
        metadata={"help": "The learning rate decay steps for z."},
    )

    clip_grad_type: str = field(
        default="norm",
        metadata={"help": "The clip grad type. "},
    )

    clip_grad_param: float = field(
        default=10.0,
        metadata={"help": "The clip grad param. "},
    )

    random_action_prob: float = field(
        default=0.001,
        metadata={"help": "The random action probability. "},
    )

    sampling_tau: float = field(
        default=0.0,
        metadata={"help": "The sampling temperature. "},
    )

    max_nodes: int = field(
        default=9,
        metadata={"help": "The maximum number of nodes. "},
    )

    num_offline: int = field(
        default=10,
        metadata={"help": "The number of offline samples. "},
    )


@dataclass
class GFlowNetDataArguments(TrainingPipelineArguments):
    """
    Arguments related to data.
    """

    __name__ = "dataset_args"

    dataset: str = field(
        default="qm9",
        metadata={"help": "The dataset to use for training the model. "},
    )
    dataset_path: str = field(
        default="./data/qm9",
        metadata={"help": "The path to the dataset to use for training the model. "},
    )

    batch_size: int = field(
        default=64,
        metadata={"help": "Batch size of the training. Defaults to 64."},
    )
    global_batch_size: int = field(
        default=16,
        metadata={"help": "Global batch size of the training. Defaults to 16."},
    )

    validation_split: Optional[float] = field(
        default=None,
        metadata={
            "help": "Proportion used for validation. Defaults to None, a.k.a., use indices file if provided otherwise uses half of the data for validation."
        },
    )
    validation_indices_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Indices to use for validation. Defaults to None, a.k.a., use validation split proportion, if not provided uses half of the data for validation."
        },
    )
    stratified_batch_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Stratified batch file for sampling. Defaults to None, a.k.a., no stratified sampling."
        },
    )
    stratified_value_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Stratified value name. Defaults to None, a.k.a., no stratified sampling. Needed in case a stratified batch file is provided."
        },
    )
    num_data_loader_workers: int = field(
        default=8,
        metadata={"help": "The number of data loader workers. "},
    )
    sampling_iterator: bool = field(
        default=True,
        metadata={"help": "Whether to use a sampling iterator. "},
    )

    ratio: float = field(
        default=0.9,
        metadata={"help": "The ratio. "},
    )


@dataclass
class GFlowNetSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to Granular trainer."""

    __name__ = "saving_args"

    model_path: str = field(
        metadata={"help": "Path to the checkpoint file to be used."}
    )
