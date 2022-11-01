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
"""Diffusion training utilities. Code adapted from: https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from rdkit.Chem import Descriptors
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm

from ..core import TrainingPipeline, TrainingPipelineArguments

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# NOTE: This import is needed because importing torchvision before rdkit Descriptors
# can cause segmentation faults.
Descriptors


class DiffusionForVisionTrainingPipeline(TrainingPipeline):
    """Diffusion training pipelines for image generation."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
    ) -> None:
        """Generic training function for Diffusion models.

        Args:
            training_args: training arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.

        Raises:
            NotImplementedError: the generic trainer does not implement the pipeline.
        """
        params = {**training_args, **dataset_args, **model_args}

        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != params["local_rank"]:
            params["local_rank"] = env_local_rank

        logging_dir = os.path.join(params["output_dir"], params["logging_dir"])
        model_path = params["model_path"]
        training_name = params["training_name"]

        accelerator = Accelerator(
            mixed_precision=params["mixed_precision"],
            log_with="tensorboard",
            logging_dir=logging_dir,
        )
        logger.info(f"Model with name {training_name} starts.")

        model_dir = os.path.join(model_path, training_name)
        log_path = os.path.join(model_dir, "logs")
        val_dir = os.path.join(log_path, "val_logs")

        os.makedirs(os.path.join(model_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(model_dir, "results"), exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # unet decoder
        model = UNet2DModel(
            sample_size=params["resolution"],
            in_channels=params["in_channels"],
            out_channels=params["out_channels"],
            layers_per_block=params["layers_per_block"],
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        # ddpm noise schedule
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=params["num_train_timesteps"]
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            betas=(params["adam_beta1"], params["adam_beta2"]),
            weight_decay=params["adam_weight_decay"],
            eps=params["adam_epsilon"],
        )

        augmentations = Compose(
            [
                Resize(params["resolution"], interpolation=InterpolationMode.BILINEAR),
                CenterCrop(params["resolution"]),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize([0.5], [0.5]),
            ]
        )
        # specify dataset by name or path
        if params["dataset_name"] is not None:
            dataset = load_dataset(
                params["dataset_name"],
                params["dataset_config_name"],
                cache_dir=params["cache_dir"],
                use_auth_token=True if params["use_auth_token"] else None,
                split="train",
            )
            logger.info("dataset name: " + params["dataset_name"])
        else:
            if params["train_data_dir"] is None:
                raise ValueError(
                    "You must specify either a dataset name from the hub or a train data directory."
                )
            dataset = load_dataset(
                "imagefolder",
                data_dir=params["train_data_dir"],
                cache_dir=params["cache_dir"],
                split="train",
            )
            logger.info("dataset path: " + params["train_data_dir"])

        def transforms(examples):
            try:
                images = [
                    augmentations(image.convert("RGB")) for image in examples["img"]
                ]
            except KeyError:
                images = [
                    augmentations(image.convert("RGB")) for image in examples["image"]
                ]
            return {"input": images}

        dataset.set_transform(transforms)  # type: ignore
        train_dataloader = torch.utils.data.DataLoader(  # type: ignore
            dataset, batch_size=params["train_batch_size"], shuffle=True  # type: ignore
        )

        # specify learning rate scheduler
        lr_scheduler = get_scheduler(
            params["lr_scheduler"],
            optimizer=optimizer,
            num_warmup_steps=params["lr_warmup_steps"],
            num_training_steps=(len(train_dataloader) * params["num_epochs"])
            // params["gradient_accumulation_steps"],
        )

        # preparare for distributed training if neeeded
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        # initialize the ema model
        ema_model = EMAModel(
            model,
            inv_gamma=params["ema_inv_gamma"],
            power=params["ema_power"],
            max_value=params["ema_max_decay"],
        )

        if accelerator.is_main_process:
            run = os.path.split(__file__)[-1].split(".")[0]
            accelerator.init_trackers(run)

        global_step = 0
        # start training
        for epoch in range(params["num_epochs"]):

            model.train()
            # progress bar visualization
            progress_bar = tqdm(
                total=len(train_dataloader),
                disable=not accelerator.is_local_main_process,
            )
            progress_bar.set_description(f"Epoch {epoch}")

            for _, batch in enumerate(train_dataloader):

                clean_images = batch["input"]
                # Sample noise that we'll add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bsz = clean_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=clean_images.device,
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                with accelerator.accumulate(model):
                    # Predict the noise residual
                    noise_pred = model(noisy_images, timesteps)["sample"]
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    if params["use_ema"]:
                        ema_model.step(model)
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                if params["use_ema"]:
                    logs["ema_decay"] = ema_model.decay
                progress_bar.set_postfix(**logs)

                accelerator.log(logs, step=global_step)
                global_step += 1

                if params["dummy_training"]:
                    break

            progress_bar.close()
            # wait for all the processes to finish
            accelerator.wait_for_everyone()

            # Generate sample images for visual inspection
            if accelerator.is_main_process and params["is_sampling"]:
                if (
                    epoch % params["save_images_epochs"] == 0
                    or epoch == params["num_epochs"] - 1
                ):
                    # inference/sampling
                    pipeline = DDPMPipeline(
                        unet=accelerator.unwrap_model(
                            ema_model.averaged_model if params["use_ema"] else model
                        ),
                        scheduler=noise_scheduler,
                    )

                    generator = torch.manual_seed(0)
                    # run pipeline in inference (sample random noise and denoise)
                    images = pipeline(
                        generator=generator,
                        batch_size=params["eval_batch_size"],
                        output_type="numpy",
                    )["sample"]

                    # denormalize the images and save to tensorboard
                    images_processed = (images * 255).round().astype("uint8")
                    accelerator.trackers[0].writer.add_images(
                        "test_samples",
                        images_processed.transpose(0, 3, 1, 2),
                        epoch,
                    )

                if (
                    epoch % params["save_model_epochs"] == 0
                    or epoch == params["num_epochs"] - 1
                ):
                    pipeline.save_pretrained(params["output_dir"])
            accelerator.wait_for_everyone()

            if params["dummy_training"]:
                break

        accelerator.end_training()
        logger.info("Training done, shutting down.")


@dataclass
class DiffusionDataArguments(TrainingPipelineArguments):
    """Data arguments related to diffusion trainer."""

    __name__ = "dataset_args"

    dataset_name: str = field(default="", metadata={"help": "Dataset name."})
    dataset_config_name: str = field(
        default="", metadata={"help": "Dataset config name."}
    )
    train_data_dir: str = field(default="", metadata={"help": "Train data directory."})
    resolution: int = field(default=64, metadata={"help": "Resolution."})
    train_batch_size: int = field(default=16, metadata={"help": "Train batch size."})
    eval_batch_size: int = field(default=16, metadata={"help": "Eval batch size."})
    num_epochs: int = field(default=100, metadata={"help": "Number of epochs."})


@dataclass
class DiffusionModelArguments(TrainingPipelineArguments):
    """Model arguments related to Diffusion trainer."""

    __name__ = "model_args"

    model_path: str = field(metadata={"help": "Path to the model file."})
    training_name: str = field(metadata={"help": "Name of the training run."})

    num_train_timesteps: int = field(
        default=1000, metadata={"help": "Number of noise steps."}
    )
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate."})
    lr_scheduler: str = field(
        default="cosine", metadata={"help": "Learning rate scheduler."}
    )
    lr_warm_up_steps: int = field(
        default=500, metadata={"help": "Learning rate warm up steps."}
    )
    adam_beta1: float = field(default=0.95, metadata={"help": "Adam beta1."})
    adam_beta2: float = field(default=0.999, metadata={"help": "Adam beta2."})
    adam_weight_decay: float = field(
        default=1e-6, metadata={"help": "Adam weights decay."}
    )
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Adam eps."})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Gradient accumulation steps."}
    )
    in_channels: int = field(default=3, metadata={"help": "Input channels."})
    out_channels: int = field(default=3, metadata={"help": "Output channels."})
    layers_per_block: int = field(default=2, metadata={"help": "Layers per block."})


@dataclass
class DiffusionTrainingArguments(TrainingPipelineArguments):
    """Training arguments related to Diffusion trainer."""

    __name__ = "training_args"

    local_rank: int = field(default=-1, metadata={"help": "Local rank of the process."})
    output_dir: str = field(
        default="ddpm-cifar10-32", metadata={"help": "Output directory."}
    )
    logging_dir: str = field(default="logs/", metadata={"help": "Logging directory."})
    overwrite_output_dir: bool = field(
        default=False, metadata={"help": "Overwrite output directory."}
    )
    cache_dir: str = field(default=".cache/", metadata={"help": "Cache directory."})
    save_images_epochs: int = field(
        default=10, metadata={"help": "Save images every n epochs."}
    )
    save_model_epochs: int = field(
        default=10, metadata={"help": "Save model every n epochs."}
    )
    use_ema: bool = field(default=True, metadata={"help": "Use ema."})
    ema_inv_gamma: float = field(default=1.0, metadata={"help": "Ema inverse gamma."})
    ema_power: float = field(default=0.75, metadata={"help": "Ema power."})
    ema_max_decay: float = field(default=0.9999, metadata={"help": "Ema max delay."})
    mixed_precision: str = field(
        default="no",
        metadata={"help": "Mixed precision. Choose from 'no', 'fp16', 'bf16'."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Use the token generated when using huggingface-hub (necessary to use this script with private models)."
        },
    )
    dummy_training: bool = field(
        default=False,
        metadata={"help": "Run dummy training to test the pipeline."},
    )
    is_sampling: bool = field(
        default=True,
        metadata={"help": "Run sampling."},
    )


@dataclass
class DiffusionSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to Diffusion trainer."""

    __name__ = "saving_args"
