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
"""NoEncoding implementation."""

from argparse import ArgumentParser
from typing import Any, Dict, Tuple

from ....arg_parser.utils import str2bool
from ..base_model import GranularEncoderDecoderModel


class NoEncoding(GranularEncoderDecoderModel):
    """NoEncoding module for adding inputs directly in the latent space."""

    def __init__(
        self,
        name: str,
        position: int,
        data: Dict[str, str],
        latent_size: int = 2,
        **kwargs,
    ) -> None:
        """Construct NoEncoding.

        Args:
            name: model name.
            position: position of the model.
            data: data name mappings.
            latent_size: latent size. Defaults to 2.
        """
        super().__init__(name=name, data=data)
        self.position = position
        self.input_key = f"{name}_{data['input']}"
        self.target_key = f"{name}_{data['target']}"
        self.latent_size = latent_size

    def decode(self, z: Any, *args, **kwargs) -> Any:
        """Decode a latent space point.

        Args:
            z: latent point.

        Returns:
            decoded sample.
        """
        return z

    def encode(self, x: Any, *args, **kwargs) -> Any:
        """Encode a sample.

        Args:
            x: input sample.

        Returns:
            latent encoding.
        """
        return x

    def inference(self, z: Any, *args, **kwargs) -> Any:
        """Run the model in inference mode.

        Args:
            z: sample.

        Returns:
            generated output.
        """
        return z

    def forward(self, x: Any, *args, **kwargs) -> Any:
        """Forward pass in the model.

        Args:
            x: model input.

        Returns:
            model output.
        """
        return x

    def _run_step(self, x: Any, *args, **kwargs) -> Any:
        """Run a step in the model.

        Args:
            x: model input.

        Returns:
            model step output.
        """
        return x

    def encode_decode(self, x: Any, *args, **kwargs) -> Any:
        """Encode and decode a sample.

        Args:
            x: input sample.

        Returns:
            decoded sample.
        """
        z, x_out = self._run_step(x)
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
        z = input_data

        loss = 0
        logs = {"loss": loss}

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
        z = input_data

        loss = 0
        logs = {"loss": loss}
        return z, loss, logs

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser, name: str, *args, **kwargs
    ) -> ArgumentParser:
        """Adding to a parser model specific arguments.

        Args:
            parent_parser: patent parser.
            name: model name.

        Returns:
            update parser.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(f"--data_path_{name}", type=str)
        parser.add_argument(f"--data_file_{name}", type=str)
        parser.add_argument(f"--position_{name}", type=int, nargs="+")
        parser.add_argument(f"--input_{name}", type=str)
        parser.add_argument(f"--target_{name}", type=str)
        parser.add_argument(f"--checkpoint_path_{name}", type=str)
        parser.add_argument(f"--start_from_checkpoint_{name}", type=str2bool)
        parser.add_argument(f"--checkpoint_model_name_{name}", type=str)
        parser.add_argument(f"--latent_size_{name}", type=int)

        return parser
