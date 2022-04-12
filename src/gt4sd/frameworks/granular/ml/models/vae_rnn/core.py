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
"""VaeRnn implementation."""

from argparse import ArgumentParser
from typing import Any, Dict, Tuple

import torch
from torch import nn

from ....arg_parser.utils import str2bool
from ....tokenizer import Tokenizer
from ..base_model import GranularEncoderDecoderModel
from ..loss import LOSS_FACTORY
from ..module import RnnDecoder, RnnEncoder
from ..utils import KLAnnealer


class VaeRnn(GranularEncoderDecoderModel):
    """VaeRnn - variational encoder using RNN with Gaussian prior and approximate posterior."""

    def __init__(
        self,
        name: str,
        position: int,
        data: Dict[str, str],
        vocab_size: int,
        embedding_size: int,
        tokenizer: Tokenizer,
        hidden_size_enc: int = 265,
        n_layers_enc: int = 2,
        hidden_size_dec: int = 265,
        n_layers_dec: int = 2,
        bidirectional: bool = False,
        latent_size: int = 196,
        teacher_forcing: bool = True,
        loss_function: str = "ce",
        kl_low: float = 0.0,
        kl_high: float = 0.1,
        kl_n_epochs: int = 100,
        kl_start_epoch: int = 0,
        inference_check_frequency: int = 50,
        **kwargs,
    ) -> None:
        """Construct VaeRnn.

        Args:
            name: model name.
            position: position of the model.
            data: data name mappings.
            vocab_size: size of the vocabulary.
            embedding_size: size of the embedding vectors.
            tokenizer: tokenizer.
            hidden_size_enc: encoder hidden size. Defaults to 256.
            n_layers_enc: number of layers for the encoder. Defaults to 2.
            hidden_size_dec: decoder hidden size. Defaults to 256.
            n_layers_dec: number of layers for the decoder. Defaults to 2.
            bidirectional: whether the RNN cell is bidirectional. Defaults to False.
            latent_size: latent size. Defaults to 196.
            teacher_forcing: whether to teacher forcing. Defaults to True.
            loss_function: loss function. Defaults to "ce".
            kl_low: low KL weight. Defaults to 0.0.
            kl_high: high KL weight. Defaults to 0.1.
            kl_n_epochs: KL number of epochs. Defaults to 100.
            kl_start_epoch: KL starting epoch. Defaults to 0.
            inference_check_frequency: frequency for checking inference quality. Defaults to 50.

        Raises:
            ValueError: in case the provided loss function is not supported.
        """
        super().__init__(name=name, data=data)
        self.position = position
        self.input_key = f"{name}_{data['input']}"
        self.target_key = f"{name}_{data['target']}"

        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.teacher_forcing = teacher_forcing
        self.tokenizer = tokenizer
        self.hidden_size_enc = hidden_size_enc
        self.n_layers_enc = (n_layers_enc,)
        self.hidden_size_dec = hidden_size_dec
        self.n_layers_dec = (n_layers_dec,)
        self.hidden_factor = (2 if bidirectional else 1) * n_layers_enc

        self.loss_function_name = loss_function.lower()
        if self.loss_function_name not in LOSS_FACTORY:
            raise ValueError(
                f"loss_function={self.loss_function_name} not supported. Pick a valid one: {sorted(list(LOSS_FACTORY.keys()))}"
            )
        self.loss_function = LOSS_FACTORY[self.loss_function_name]

        self.fc_mu = nn.Linear(self.hidden_factor * hidden_size_enc, self.latent_size)
        self.fc_var = nn.Linear(self.hidden_factor * hidden_size_enc, self.latent_size)
        self.encoder = RnnEncoder(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size_enc,
            n_layers=n_layers_enc,
            bidirectional=bidirectional,
        )
        self.decoder = RnnDecoder(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size_dec,
            n_layers=n_layers_dec,
            latent_size=latent_size,
        )

        self.epoch_counter = 0
        self.klannealer = KLAnnealer(
            kl_low=kl_low,
            kl_high=kl_high,
            n_epochs=kl_n_epochs,
            start_epoch=kl_start_epoch,
        )
        self.inference_check_frequency = inference_check_frequency

    def decode(self, z: Any, max_len: int = 127, *args, **kwargs) -> Any:
        """Decode a latent space point.

        Args:
            z: latent point.
            max_len: maximum sequence length. Defaults to 127.

        Returns:
            tuple with decoded texts and token indices.
        """
        decoded_texts, token_indices = self.decoder.inference_direct(
            z, self.encoder.embedding, self.tokenizer, max_len=max_len
        )
        return decoded_texts, token_indices

    def _sampling_step(self, x: Any, *args, **kwargs) -> Any:
        """Run a sampling step in the model.

        Args:
            x: model input.

        Returns:
            model sampling step output.
        """
        x, input_embedding = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return p, q, z, input_embedding

    def encode(self, x: Any, *args, **kwargs) -> Any:
        """Encode a sample.

        Args:
            x: input sample.

        Returns:
            latent encoding.
        """
        _, _, z, _ = self._sampling_step(x)
        return z

    def encode_decode(self, x: Any, max_len: int = 127, *args, **kwargs) -> Any:
        """Encode and decode a sample.

        Args:
            x: input sample.
            max_len: maximum sequence length. Defaults to 127.

        Returns:
            decoded sample.
        """
        z = self.encode(x)
        return self.decode(z, max_len=max_len)

    def inference(self, x: Any, *args, **kwargs) -> Any:  # type:ignore
        """Run the model in inference mode.

        Args:
            x: sample.

        Returns:
            generated output.
        """
        max_len = x.size(1)
        _, _, z, _ = self._sampling_step(x)
        return self.decode(z, max_len=max_len)

    def _run_step(self, x: Any, *args, **kwargs) -> Any:
        """Run a step in the model.

        Args:
            x: model input.

        Returns:
            model step output.
        """
        p, q, z, input_embedding = self._sampling_step(x)
        return z, self.decoder(z, input_embedding), p, q

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
        x_out = target_data
        # teacher forcing
        if self.teacher_forcing:
            x_out = x_out[:, 1:].long()
            x = x[:, :-1]

        z, x_hat, p, q = self._run_step(x)

        x_hat = x_hat.view(-1, x_hat.size(-1))
        x_target = x_out.contiguous().view(-1)

        reconstruction_loss = self.loss_function(x_hat, x_target)

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)
        kl_scaling_factor = self.klannealer(current_epoch)
        kl = log_qz - log_pz
        kl = kl.mean()
        kl_scaled = kl * kl_scaling_factor

        loss = kl_scaled + reconstruction_loss
        logs = {
            "reconstruction_loss": reconstruction_loss,
            "kl_scaled": kl_scaled,
            "kl_unscaled": kl,
            "kl_scaling_factor": kl_scaling_factor,
            "loss": loss,
        }

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
        x = input_data
        z, loss, logs = self.step(
            input_data=input_data,
            target_data=target_data,
            device=device,
            current_epoch=current_epoch,
        )
        if current_epoch % self.inference_check_frequency == 0 and current_epoch > 0:
            decoded_texts, token_indices = self.inference(x)
            reconstructed_texts = 0
            decoded_splitted_texts = [
                text.split(self.tokenizer.eos_token, 1)[0] for text in decoded_texts
            ]
            for _, text in enumerate(decoded_splitted_texts):
                if self.tokenizer.pad_token not in text:
                    reconstructed_texts += 1
            valid_percentage = float(reconstructed_texts) / x.size(0)
            reconstructed_bits = torch.sum(
                x[:, 1:] == token_indices[:, : x[:, 1:].size(1)]
            ).item()
            reconstructed_bits_percentage = reconstructed_bits / x.numel()
            logs.update(
                {
                    "reconstructed_bits": reconstructed_bits_percentage,
                    "validity": valid_percentage,
                }
            )

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
            updated parser.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(f"--data_path_{name}", type=str)
        parser.add_argument(f"--data_file_{name}", type=str)
        parser.add_argument(f"--dataset_type_{name}", type=str)
        parser.add_argument(f"--position_{name}", type=int, nargs="+")
        parser.add_argument(f"--build_vocab{name}", type=str2bool)
        parser.add_argument(f"--vocab_file{name}", type=str)
        parser.add_argument(f"--input_{name}", type=str)
        parser.add_argument(f"--target_{name}", type=str)
        parser.add_argument(f"--checkpoint_path_{name}", type=str)
        parser.add_argument(f"--checkpoint_model_name_{name}", type=str)
        parser.add_argument(f"--start_from_checkpoint_{name}", type=str2bool)
        parser.add_argument(f"--freeze_weights_{name}", type=str2bool)
        parser.add_argument(f"--hidden_size_enc_{name}", type=int)
        parser.add_argument(f"--hidden_size_dec_{name}", type=int)
        parser.add_argument(f"--n_layers_enc_{name}", type=int)
        parser.add_argument(f"--n_layers_dec_{name}", type=int)
        parser.add_argument(f"--bidirectional_{name}", type=str2bool)
        parser.add_argument(f"--latent_size_{name}", type=int)
        parser.add_argument(f"--kl_low_{name}", type=float)
        parser.add_argument(f"--kl_high_{name}", type=float)
        parser.add_argument(f"--kl_n_epochs_{name}", type=int)
        parser.add_argument(f"--kl_start_epoch_{name}", type=int)
        parser.add_argument(f"--inference_check_frequency_{name}", type=int)

        return parser
