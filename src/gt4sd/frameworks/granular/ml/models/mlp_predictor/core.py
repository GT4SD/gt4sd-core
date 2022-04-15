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
"""MLP predictor implementation."""

import logging
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple

from ....arg_parser.utils import str2bool
from ..base_model import GranularBaseModel
from ..loss import LOSS_FACTORY
from ..module import Mlp

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MlpPredictor(GranularBaseModel):
    """MlpPredictor - Multi Layer Perceptron predictor."""

    def __init__(
        self,
        name: str,
        from_position: List[int],
        data: Dict[str, str],
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int,
        activation: str,
        dropout: float,
        loss_function: str,
        class_weights: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        """Construct MlpPredictor.

        Args:
            name: model name.
            from_position: list of input model positions.
            data: data name mappings.
            input_size: size of the input.
            hidden_size: size of the hidden layers.
            output_size: size of the output.
            n_layers: number of layers.
            activation: name of the activation.
            dropout: dropout rate.
            loss_function: name of the loss function.
            class_weights: weights for the classes. Defaults to None, a.k.a., no weighting.

        Raises:
            ValueError: in case the provided loss function is not supported.
        """
        super().__init__(name=name, data=data)
        self.from_position = from_position
        self.target_key = name + "_" + data["target"]
        self.loss_function_name = loss_function.lower()
        if self.loss_function_name not in LOSS_FACTORY:
            raise ValueError(
                f"loss_function={self.loss_function_name} not supported. Pick a valid one: {sorted(list(LOSS_FACTORY.keys()))}"
            )
        self.loss_function = LOSS_FACTORY[self.loss_function_name]
        self.class_weights = class_weights
        self.mlp = Mlp(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            n_layers=n_layers,
            activation=activation,
            dropout=dropout,
        )

    def _run_step(self, x: Any, *args, **kwargs) -> Any:
        """Run a step in the model.

        Args:
            x: model input.

        Returns:
            model step output.
        """
        return self.mlp(x)

    def predict(self, x: Any, *args, **kwargs) -> Any:
        """Forward pass in the model.

        Args:
            x: model input.

        Returns:
            model output.
        """
        return self._run_step(x)

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
        output = self._run_step(input_data)
        loss = self.loss_function(output, target_data)
        logs = {f"{self.loss_function_name}_loss": loss, "loss": loss}
        return output, loss, logs

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
        output = self._run_step(input_data)
        loss = self.loss_function(output, target_data)
        logs = {f"{self.loss_function_name}_loss": loss, "loss": loss}

        if self.loss_function_name == "bce":
            output_label = (output > 0.5).float()
            correct_label = (output_label == target_data).float().sum()
            accuracy = correct_label / output_label.shape[0]
            logs["accuracy"] = accuracy
        return output, loss, logs

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
        parser.add_argument(f"--target_{name}", type=str)
        parser.add_argument(f"--from_position_{name}", type=int, nargs="+")
        parser.add_argument(f"--checkpoint_path_{name}", type=str)
        parser.add_argument(f"--checkpoint_model_name_{name}", type=str)
        parser.add_argument(f"--start_from_checkpoint_{name}", type=str2bool)
        parser.add_argument(f"--freeze_weights_{name}", type=str2bool)
        parser.add_argument(f"--n_layers_{name}", type=int)
        parser.add_argument(f"--activation_{name}", type=str)
        parser.add_argument(f"--dropout_{name}", type=float)
        parser.add_argument(f"--loss_function_{name}", type=str)
        parser.add_argument(f"--hidden_size_{name}", type=int)
        parser.add_argument(f"--output_size_{name}", type=int)

        return parser
