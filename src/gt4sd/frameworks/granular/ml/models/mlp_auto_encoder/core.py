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
"""MLP autoencoder implementation."""

from argparse import ArgumentParser
from typing import Any, Dict, Tuple

from ....arg_parser.utils import str2bool
from ..base_model import GranularEncoderDecoderModel
from ..loss import LOSS_FACTORY
from ..module import MlpDecoder, MlpEncoder


class MlpAutoEncoder(GranularEncoderDecoderModel):
    """MlpAutoencoder - Multi Layer Perceptron autoencoder."""

    def __init__(
        self,
        name: str,
        position: int,
        data: Dict[str, str],
        input_size_enc: int = 256,
        hidden_size_enc: int = 256,
        n_layers_enc: int = 2,
        activation_enc: str = "linear",
        dropout_enc: float = 0.0,
        hidden_size_dec: int = 256,
        n_layers_dec: int = 2,
        activation_dec: str = "linear",
        dropout_dec: float = 0.0,
        output_size_dec: int = 256,
        latent_size: int = 196,
        loss_function: str = "mse",
        **kwargs,
    ) -> None:
        """Construct MlpAutoEncoder.

        Args:
            name: model name.
            position: position of the model.
            data: data name mappings.
            input_size_enc: encoder input size. Defaults to 256.
            hidden_size_enc: encoder hidden size. Defaults to 256.
            n_layers_enc: number of layers for the encoder. Defaults to 2.
            activation_enc: activation function for the encoder. Defaults to "linear".
            dropout_enc: encoder dropout rate. Defaults to 0.0.
            hidden_size_dec: decoder hidden size. Defaults to 256.
            n_layers_dec: number of layers for the decoder. Defaults to 2.
            activation_dec: activation function for the decoder. Defaults to "linear".
            dropout_dec: decoder dropout rate. Defaults to 0.0.
            output_size_dec: decoder output size. Defaults to 256.
            latent_size: size of the latent space. Defaults to 196.
            loss_function: loss function. Defaults to "mse".

        Raises:
            ValueError: in case the provided loss function is not supported.
        """
        super().__init__(name=name, data=data)
        self.position = position
        self.input_key = f"{name}_{data['input']}"
        self.target_key = f"{name}_{data['target']}"

        self.latent_size = latent_size
        self.input_size_enc = input_size_enc
        self.hidden_size_enc = hidden_size_enc
        self.n_layers_enc = n_layers_enc
        self.activation_enc = activation_enc
        self.dropout_enc = dropout_enc
        self.output_size_enc = latent_size

        self.hidden_size_dec = hidden_size_dec
        self.n_layers_dec = n_layers_dec
        self.activation_dec = activation_dec
        self.dropout_dec = dropout_dec
        self.output_size_dec = output_size_dec

        self.loss_function_name = loss_function.lower()
        if self.loss_function_name not in LOSS_FACTORY:
            raise ValueError(
                f"loss_function={self.loss_function_name} not supported. Pick a valid one: {sorted(list(LOSS_FACTORY.keys()))}"
            )
        self.loss_function = LOSS_FACTORY[self.loss_function_name]

        self.encoder = MlpEncoder(
            input_size=input_size_enc,
            hidden_size=hidden_size_enc,
            output_size=latent_size,
            n_layers=n_layers_enc,
            activation=activation_enc,
            dropout=dropout_enc,
        )
        self.decoder = MlpDecoder(
            latent_size=latent_size,
            hidden_size=hidden_size_dec,
            output_size=output_size_dec,
            n_layers=n_layers_dec,
            activation=activation_dec,
            dropout=dropout_dec,
        )
        self.epoch_counter = 0

    def decode(self, z: Any, *args, **kwargs) -> Any:
        """Decode a latent space point.

        Args:
            z: latent point.

        Returns:
            decoded sample.
        """
        return self.decoder(z)

    def encode(self, x: Any, *args, **kwargs) -> Any:
        """Encode a sample.

        Args:
            x: input sample.

        Returns:
            latent encoding.
        """
        return self.encoder(x)

    def _run_step(self, x: Any, *args, **kwargs) -> Any:
        """Run a step in the model.

        Args:
            x: model input.

        Returns:
            model step output.
        """
        z = self.encoder(x)
        x_out = self.decoder(z)
        return z, x_out

    def step(
        self,
        input_data: Any,
        target_data: Any,
        device: str = "cpu",
        current_epoch: int = 0,
        *args,
        **kwargs,
    ) -> Tuple[Any, Any, Any]:
        """Training step for the model.

        Args:
            input_data: input for the step.
            target_data: target for the step.
            device: string representing the device to use. Defaults to "cpu".
            current_epoch: current epoch. Defaults to 0.

        Returns:
            a tuple containing the step output, the loss and the logs for the module.
        """
        x = input_data
        x_target = target_data

        z, x_hat = self._run_step(x)
        loss = self.loss_function(x_hat, x_target)
        logs = {"reconstruction_loss": loss, "loss": loss}
        return z, loss, logs

    def val_step(
        self,
        input_data: Any,
        target_data: Any,
        device: str = "cpu",
        current_epoch: int = 0,
        *args,
        **kwargs,
    ) -> Any:
        """Validation step for the model.

        Args:
            input_data: input for the step.
            target_data: target for the step.
            device: string representing the device to use. Defaults to "cpu".
            current_epoch: current epoch. Defaults to 0.

        Returns:
            a tuple containing the step output, the loss and the logs for the module.
        """
        return self.step(input_data, target_data, device, current_epoch)

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser, name: str, *args, **kwargs
    ) -> ArgumentParser:
        """Adding to a parser model specific arguments.

        Args:
            parent_parser: patent parser.
            name: model name.

        Returns:
            updated parser.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(f"--data_path_{name}", type=str)
        parser.add_argument(f"--data_file_{name}", type=str)
        parser.add_argument(f"--dataset_type_{name}", type=str)
        parser.add_argument(f"--position_{name}", type=int, nargs="+")
        parser.add_argument(f"--input_{name}", type=str)
        parser.add_argument(f"--target_{name}", type=str)
        parser.add_argument(f"--checkpoint_path_{name}", type=str)
        parser.add_argument(f"--checkpoint_model_name_{name}", type=str)
        parser.add_argument(f"--start_from_checkpoint_{name}", type=str2bool)
        parser.add_argument(f"--freeze_weights_{name}", type=str2bool)
        parser.add_argument(f"--input_size_enc_{name}", type=int)
        parser.add_argument(f"--hidden_size_enc_{name}", type=int)
        parser.add_argument(f"--n_layers_enc_{name}", type=int)
        parser.add_argument(f"--activation_enc_{name}", type=str)
        parser.add_argument(f"--dropout_enc_{name}", type=float)
        parser.add_argument(f"--hidden_size_dec_{name}", type=int)
        parser.add_argument(f"--dropout_dec_{name}", type=float)
        parser.add_argument(f"--n_layers_dec_{name}", type=int)
        parser.add_argument(f"--activation_dec_{name}", type=str)
        parser.add_argument(f"--ouptput_size_enc_{name}", type=int)
        parser.add_argument(f"--latent_size_{name}", type=int)
        parser.add_argument(f"--loss_function_{name}", type=str)

        return parser
