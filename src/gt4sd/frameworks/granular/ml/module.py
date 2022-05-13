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
"""Model combiner module."""

import os
from typing import Any, Callable, Dict, List, Tuple, cast

import sentencepiece as _sentencepiece
import pandas as pd
import pytorch_lightning as pl
import torch

from .models import GranularBaseModel, GranularEncoderDecoderModel
from .models.model_builder import building_models, define_latent_models_input_size

# sentencepiece has to be loaded before lightning to avoid segfaults
_sentencepiece


class GranularModule(pl.LightningModule):
    """Module from granular."""

    def __init__(
        self,
        architecture_autoencoders: List[Dict[str, Any]],
        architecture_latent_models: List[Dict[str, Any]],
        lr: float = 1e-4,
        test_output_path: str = "./test",
        **kwargs,
    ) -> None:
        """Construct GranularModule.

        Args:
            architecture_autoencoders: list of autoencoder architecture configurations.
            architecture_latent_models: list of latent model architecture configurations.
            lr: learning rate for Adam optimizer. Defaults to 1e-4.
            test_output_path: path where to save latent encodings and predictions for the test set
                when an epoch ends. Defaults to a a folder called "test" in the current working directory.
        """
        super().__init__()
        self.save_hyperparameters()

        architecture_latent_models = define_latent_models_input_size(
            architecture_autoencoders, architecture_latent_models
        )
        self.architecture_autoencoders = architecture_autoencoders
        self.architecture_latent_models = architecture_latent_models

        self.autoencoders = building_models(self.architecture_autoencoders)
        self.latent_models = building_models(self.architecture_latent_models)

        self.lr = lr
        self.test_output_path = test_output_path
        for model in self.autoencoders + self.latent_models:
            setattr(self, model.name, model)

    def _autoencoder_step(
        self, batch: Any, model: GranularEncoderDecoderModel, model_step_fn: Callable
    ) -> Tuple[Any, Any, Any]:
        """Autoencoder module forward pass.

        Args:
            batch: batch representation.
            model: a module.
            model_step_fn: callable for the step.

        Returns:
            a tuple containing the latent representation, the loss and the logs for the module.
        """
        return model_step_fn(
            input_data=batch[model.input_key],
            target_data=batch[model.target_key],
            device=self.device,
            current_epoch=self.current_epoch,
        )

    def _latent_step(
        self,
        batch: Any,
        model: GranularBaseModel,
        model_step_fn: Callable,
        z: Dict[int, Any],
    ) -> Tuple[Any, Any, Any]:
        """Latent module forward pass.

        Args:
            batch: batch representation.
            model: a module.
            model_step_fn: callable for the step.
            z: latent encodings.

        Returns:
            a tuple containing the latent step ouput, the loss and the logs for the module.
        """
        z_model_input = torch.cat(
            [
                torch.squeeze(z[pos]) if len(z[pos].size()) == 3 else z[pos]
                for pos in model.from_position
            ],
            dim=1,
        )
        return model_step_fn(
            input_data=z_model_input,
            target_data=batch[model.target_key],
            device=self.device,
            current_epoch=self.current_epoch,
        )

    def training_step(  # type:ignore
        self, batch: Any, *args, **kwargs
    ) -> Dict[str, Any]:
        """Training step implementation.

        Args:
            batch: batch representation.

        Returns:
            loss and logs.
        """
        loss = 0.0
        z = dict()
        logs = dict()

        for model in self.autoencoders:
            z[model.position], loss_model, logs_model = self._autoencoder_step(
                batch=batch,
                model=cast(GranularEncoderDecoderModel, model),
                model_step_fn=model.step,
            )
            logs.update({model.name + f"/{k}": v for k, v in logs_model.items()})
            loss += loss_model

        for model in self.latent_models:
            _, loss_model, logs_model = self._latent_step(
                batch=batch, model=model, model_step_fn=model.step, z=z
            )
            logs.update({model.name + f"/{k}": v for k, v in logs_model.items()})
            loss += loss_model

        logs.update({"total_loss": loss})
        self.log_dict(
            {f"train/{k}": v for k, v in logs.items()}, on_epoch=False, prog_bar=False
        )
        logs_epoch = {f"train_epoch/{k}": v for k, v in logs.items()}
        logs_epoch["step"] = self.current_epoch
        self.log_dict(logs_epoch, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "logs": logs}

    def validation_step(  # type:ignore
        self, batch: Any, *args, **kwargs
    ) -> Dict[str, Any]:
        """Validation step implementation.

        Args:
            batch: batch representation.

        Returns:
            loss and logs.
        """
        loss = 0.0
        z = dict()
        logs = dict()

        for model in self.autoencoders:
            z[model.position], loss_model, logs_model = self._autoencoder_step(
                batch=batch,
                model=cast(GranularEncoderDecoderModel, model),
                model_step_fn=model.val_step,
            )
            logs.update({model.name + f"/{k}": v for k, v in logs_model.items()})
            loss += loss_model

        for model in self.latent_models:
            _, loss_model, logs_model = self._latent_step(
                batch=batch, model=model, model_step_fn=model.val_step, z=z
            )
            logs.update({model.name + f"/{k}": v for k, v in logs_model.items()})
            loss += loss_model

        logs.update({"total_loss": loss})
        self.log_dict(
            {f"val/{k}": v for k, v in logs.items()}, on_epoch=True, prog_bar=True
        )

        return {"loss": loss, "logs": logs}

    def test_step(  # type:ignore
        self, batch: Any, batch_idx: int, *args, **kwargs
    ) -> Dict[str, Any]:
        """Testing step implementation.

        Args:
            batch: batch representation.
            batch_idx: batch index, unused.

        Returns:
            loss, logs, and latent encodings.
        """
        loss = 0.0
        z = dict()
        logs = dict()

        for model in self.autoencoders:
            z[model.position], loss_model, logs_model = self._autoencoder_step(
                batch=batch,
                model=cast(GranularEncoderDecoderModel, model),
                model_step_fn=model.val_step,
            )
            logs.update({model.name + f"/{k}": v for k, v in logs_model.items()})
            loss += loss_model

        for model in self.latent_models:
            _, loss_model, logs_model = self._latent_step(
                batch=batch, model=model, model_step_fn=model.val_step, z=z
            )
            logs.update({model.name + f"/{k}": v for k, v in logs_model.items()})
            loss += loss_model

        logs.update({"total_loss": loss})
        self.log_dict(
            {f"val/{k}": v for k, v in logs.items()}, on_epoch=True, prog_bar=True
        )
        return {"loss": loss, "logs": logs, "z": z}

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:  # type:ignore
        """Callback called at the end of an epoch on test outputs.

        Dump encodings and targets for the test set.

        Args:
            outputs: outputs for test batches.
        """
        z = {}
        targets = {}
        z_keys = [key for key in outputs[0]["z"]]
        targets_keys = [key for key in outputs[0]["targets"]]
        for key in z_keys:
            z[key] = (
                torch.cat(
                    [torch.squeeze(an_output["z"][key]) for an_output in outputs], dim=0
                )
                .detach()
                .cpu()
                .numpy()
            )

        for key in targets_keys:
            targets[key] = (
                torch.cat(
                    [torch.squeeze(an_output["targets"][key]) for an_output in outputs],
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )

        pd.to_pickle(z, f"{self.test_output_path}{os.path.sep}z_build.pkl")
        pd.to_pickle(targets, f"{self.test_output_path}{os.path.sep}targets.pkl")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers.

        Returns:
            an optimizer, currently only Adam is supported.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
