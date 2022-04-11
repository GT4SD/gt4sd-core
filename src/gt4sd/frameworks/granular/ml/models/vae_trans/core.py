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
"""VaeTrans implementation."""

from argparse import ArgumentParser
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .....torch import get_device_from_tensor
from ....arg_parser.utils import str2bool
from ..base_model import GranularEncoderDecoderModel
from ..loss import LOSS_FACTORY
from ..module import TransformerDecoder, TransformerEncoder
from ..utils import KLAnnealer


class VaeTrans(GranularEncoderDecoderModel):
    """Transformer-based VAE with Gaussian Prior and approx posterior."""

    def __init__(
        self,
        name: str,
        position: int,
        data: Dict[str, str],
        vocab_size: int,
        tokenizer,
        hidden_size_enc: int = 256,
        n_layers_enc: int = 2,
        hidden_size_dec: int = 256,
        n_layers_dec: int = 2,
        kl_coeff: float = 0.1,
        latent_size: int = 196,
        feedforward_size: int = 512,
        heads: int = 4,
        dropout: float = 0.1,
        bypass_bottleneck: bool = False,
        seq_len: int = 127,
        teacher_forcing: bool = True,
        loss_function: str = "ce",
        kl_low: float = 0.0,
        kl_high: float = 0.1,
        kl_n_epochs: int = 100,
        kl_start_epoch: int = 0,
        inference_check_frequency: int = 50,
        **kwargs,
    ):
        """Construct VaeRnn.

        Args:
            name: model name.
            position: position of the model.
            data: data name mappings.
            vocab_size: size of the vocabulary.
            tokenizer: tokenizer.
            hidden_size_enc: encoder hidden size. Defaults to 256.
            n_layers_enc: number of layers for the encoder. Defaults to 2.
            hidden_size_dec: decoder hidden size. Defaults to 256.
            n_layers_dec: number of layers for the decoder. Defaults to 2.
            kl_coeff: KL coefficient. Defaults to 0.1.
            latent_size: latent size. Defaults to 196.
            feedforward_size: size of the feed forward network. Default to 512.
            heads: number of heads. Defauls to 4.
            dropout: dropout rate. Defaults to 0.1.
            bypass_bottleneck: whether the bottleneck should be by passed.
                Defaults to False.
            seq_len: length of the sequence. Defaults to 127.
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
        self.kl_coeff = kl_coeff
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.teacher_forcing = teacher_forcing
        self.seq_len = seq_len

        self.loss_function_name = loss_function.lower()
        if self.loss_function_name not in LOSS_FACTORY:
            raise ValueError(
                f"loss_function={self.loss_function_name} not supported. Pick a valid one: {sorted(list(LOSS_FACTORY.keys()))}"
            )
        self.loss_function = LOSS_FACTORY[self.loss_function_name]

        self.predict_len1 = nn.Linear(self.latent_size, self.latent_size * 2)
        self.predict_len2 = nn.Linear(self.latent_size * 2, self.seq_len)

        self.encoder = TransformerEncoder(
            hidden_size_enc,
            feedforward_size,
            seq_len,
            dropout,
            heads,
            n_layers_enc,
            vocab_size,
            bypass_bottleneck,
        )
        self.decoder = TransformerDecoder(
            hidden_size_dec,
            feedforward_size,
            seq_len,
            dropout,
            heads,
            n_layers_dec,
            latent_size,
            vocab_size,
            bypass_bottleneck,
            self.encoder.conv_output_shape,
        )
        self.fc_mu = nn.Linear(self.encoder.conv_output_len, self.latent_size)
        self.fc_var = nn.Linear(self.encoder.conv_output_len, self.latent_size)

        self.klannealer = KLAnnealer(
            kl_low=kl_low,
            kl_high=kl_high,
            n_epochs=kl_n_epochs,
            start_epoch=kl_start_epoch,
        )
        self.inference_check_frequency = inference_check_frequency

    def forward(self, x: Any, tgt: torch.Tensor, *args, **kwrgs) -> Any:  # type:ignore
        """Forward pass in the model.

        Args:
            x: model input.
            tgt: target tensor

        Returns:
            model output.
        """
        x_out, _, _, z, _, _ = self._run_step(x, tgt)
        return x_out, z

    def predict_mask_length(self, mem: torch.Tensor) -> Any:
        """Predicts mask length from latent memory so mask can be re-created during inference.

        Args:
            mem: latent memory.

        Returns:
            mask length.
        """
        pred_len = self.predict_len1(mem)
        pred_len = self.predict_len2(pred_len)
        pred_len = F.softmax(pred_len, dim=-1)
        pred_len = torch.topk(pred_len, 1)[1]
        return pred_len

    def _sampling_step(self, x: Any, *args, **kwargs) -> Any:
        """Run a sampling step in the model.

        Args:
            x: model input.

        Returns:
            model sampling step output.
        """
        src_mask = (x != self.tokenizer.pad_token_id).unsqueeze(-2)
        x = self.encoder(x, src_mask)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return self.sample(mu, log_var)

    def encode(self, x: Any, *args, **kwargs) -> Any:
        """Encode a sample.

        Args:
            x: input sample.

        Returns:
            latent encoding.
        """
        _, _, z = self._sampling_step(x)
        return z

    def decode(self, z: Any, *args, **kwargs) -> Any:
        """Decode a latent space point.

        Args:
            z: latent point.

        Returns:
            tuple with decoded texts and token indices.
        """
        mask_lens = self.predict_mask_length(z)
        decoded_texts, token_indices = self.decoder.inference_direct(
            z, mask_lens, self.tokenizer
        )
        return decoded_texts, token_indices

    def encode_decode(self, x: Any, *args, **kwargs) -> Any:
        """Encode and decode a sample.

        Args:
            x: input sample.

        Returns:
            decoded sample.
        """
        z = self.encode(x)
        _, token_indices = self.decode(z, x.device)
        return token_indices

    def inference(  # type:ignore
        self, x: Any, *args, **kwargs
    ) -> Any:
        """Run the model in inference mode.

        Args:
            x: sample.

        Returns:
            generated output.
        """
        device = get_device_from_tensor(x)
        z = self.encode(x)
        decoded_texts, token_indices = self.decode(z, device)
        return decoded_texts, token_indices

    def _run_step(self, x: Any, tgt: torch.Tensor) -> Any:  # type:ignore
        """Run a step in the model.

        Args:
            x: model input.
            tgt: target tensor

        Returns:
            model step output.
        """
        src_mask = (x != self.tokenizer.pad_token_id).unsqueeze(-2)
        tgt_mask = (tgt != self.tokenizer.pad_token_id).unsqueeze(-2)
        attn_shape = (1, tgt.size(-1), tgt.size(-1))
        subsequent_mask = (
            torch.from_numpy(np.triu(np.ones(attn_shape), k=1).astype("uint8")) == 0
        )
        tgt_mask = tgt_mask & torch.autograd.Variable(
            subsequent_mask.type_as(tgt_mask.data)
        )
        x = self.encoder(x, src_mask)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        mask_lens = self.predict_len1(mu)
        mask_lens = self.predict_len2(mask_lens)
        true_len = src_mask.sum(dim=-1).contiguous().view(-1)
        p, q, z = self.sample(mu, log_var)
        x_out = self.decoder(tgt, z, src_mask, tgt_mask)
        return x_out, p, q, z, mask_lens, true_len

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

        if self.teacher_forcing:
            x_tgt_in = x_target[:, :-1]
            x_tgts_out = x_target.long()[:, 1:]
        else:
            x_tgt_in = x_target
            x_tgts_out = x_target.long()

        x_pred_out, p, q, z, pred_len, true_len = self._run_step(x, x_tgt_in)

        len_loss = self.loss_function(pred_len, true_len)

        x_pred_out = x_pred_out.contiguous().view(-1, x_pred_out.size(2))
        x_tgts_out = x_tgts_out.contiguous().view(-1)
        reconstruction_loss = self.loss_function(x_pred_out, x_tgts_out)

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)
        kl_scaling_factor = self.klannealer(current_epoch)
        kl = log_qz - log_pz
        kl = kl.mean()
        kl_scaled = kl * kl_scaling_factor

        loss = kl_scaled + reconstruction_loss + len_loss
        logs = {
            "reconstruction_loss": reconstruction_loss,
            "kl_scaled": kl_scaled,
            "kl_unscaled": kl,
            "kl_scaling_factor": kl_scaling_factor,
            "len_loss": len_loss,
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
        z, loss, logs = self.step(input_data, target_data, device, current_epoch)

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
            reconstructed_bits = torch.sum(x == token_indices).item()
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
        parser.add_argument("--feedforward_size", type=int)
        parser.add_argument("--heads", type=int)
        parser.add_argument("--dropout", type=float)
        parser.add_argument("--bypass_bottleneck", type=str2bool)
        parser.add_argument(f"--kl_low_{name}", type=float)
        parser.add_argument(f"--kl_high_{name}", type=float)
        parser.add_argument(f"--kl_n_epochs_{name}", type=int)
        parser.add_argument(f"--kl_start_epoch_{name}", type=int)
        parser.add_argument(f"--inference_check_frequency_{name}", type=int)

        return parser
