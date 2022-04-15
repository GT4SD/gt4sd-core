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
"""Generic modules."""

import copy
import math
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ....torch import get_device_from_tensor
from ...tokenizer import Tokenizer
from .activation import ACTIVATION_FACTORY


class Mlp(nn.Module):
    """MLP module."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int,
        activation: str,
        dropout: float,
        **kwargs,
    ) -> None:
        """Construct Mlp.

        Args:
            input_size: size of the input.
            hidden_size: size of the hidden layers.
            output_size: size of the output.
            n_layers: number of layers.
            activation: name of the activation.
            dropout: dropout rate.
        """
        super().__init__()
        activation = activation.lower()
        self.activation = ACTIVATION_FACTORY.get(activation, None)
        self.first_layer = nn.Linear(input_size, hidden_size)
        middle_layers: List[nn.Module] = list()
        for _ in range(n_layers):
            middle_layers.append(nn.Linear(hidden_size, hidden_size))
            middle_layers.append(nn.ReLU())
            middle_layers.append(nn.Dropout(p=dropout))
        self.middle_layers = nn.Sequential(*middle_layers)
        self.last_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.output_dim = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass in the model.

        Args:
            x: model input.

        Returns:
            model output.
        """
        z = self.first_layer(x)
        z = self.relu(z)
        z = self.middle_layers(z)
        z = self.last_layer(z)
        if self.activation:
            z = self.activation(z)
        return z


class MlpEncoder(Mlp):
    """MLP encoder."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int,
        activation: str,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        """Construct MlpEncoder.

        Args:
            input_size: size of the input.
            hidden_size: size of the hidden layers.
            output_size: size of the output.
            n_layers: number of layers.
            activation: name of the activation.
            dropout: dropout rate. Defaults to 0.0.
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            n_layers=n_layers,
            activation=activation,
            dropout=dropout,
        )


class MlpDecoder(Mlp):
    """MLP decoder."""

    def __init__(
        self,
        latent_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int,
        activation: str,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        """Construct MlpEncoder.

        Args:
            latent_size: size of the input.
            hidden_size: size of the hidden layers.
            output_size: size of the output.
            n_layers: number of layers.
            activation: name of the activation.
            dropout: dropout rate. Defaults to 0.0.
        """
        super().__init__(
            input_size=latent_size,
            hidden_size=hidden_size,
            output_size=output_size,
            n_layers=n_layers,
            activation=activation,
            dropout=dropout,
        )


class RnnEncoder(nn.Module):
    """RNN encoder."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int = 256,
        n_layers: int = 2,
        bidirectional: bool = False,
        latent_size: int = 196,
    ) -> None:
        """Construct RnnEncoder.

        Args:
            vocab_size: size of the vocabulary.
            embedding_size: size of the embedding vectors.
            hidden_size: hidden size. Defaults to 256.
            n_layers: number of layers. Defaults to 2.
            bidirectional: whether the RNN cell is bidirectional. Defaults to False.
            latent_size: latent size. Defaults to 196.
        """
        super().__init__()
        self.input_size = embedding_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.latent_size = latent_size
        self.hidden_factor = (2 if bidirectional else 1) * n_layers
        self.rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_size
        )

    def forward(
        self, input_sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass in the model.

        Args:
            input_sequence: input sequence tensor.

        Returns:
            a tuple containing hidden state and embedded sequence.
        """
        input_embedding = self.embedding(input_sequence)
        _, hidden = self.rnn(input_embedding)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.contiguous().view(hidden.size(0), -1)
        return hidden, input_embedding


class RnnDecoder(nn.Module):
    """RNN decoder."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int = 256,
        n_layers: int = 2,
        latent_size: int = 196,
    ) -> None:
        """Construct RnnDecoder.

        Args:
            vocab_size: size of the vocabulary.
            embedding_size: size of the embedding vectors.
            hidden_size: hidden size. Defaults to 256.
            n_layers: number of layers. Defaults to 2.
            latent_size: latent size. Defaults to 196.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.latent_size = latent_size
        self.hidden_factor = n_layers
        self.rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.latent2hidden = torch.nn.Linear(
            latent_size, hidden_size * self.hidden_factor
        )
        self.outputs2vocab = torch.nn.Linear(hidden_size, vocab_size)

    def forward(
        self, latent: torch.Tensor, input_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass in the model.

        Args:
            latent: latent tensor.
            input_embedding: input embedding.

        Returns:
            model output.
        """
        hidden = self.latent2hidden(latent)
        hidden = hidden.view(-1, self.hidden_factor, self.hidden_size)
        hidden = hidden.permute(1, 0, 2).contiguous()
        hidden = torch.tanh(hidden)
        outputs, _ = self.rnn(input_embedding, hidden)
        b, seq_len, hsize = outputs.size()
        outputs = outputs.contiguous().view(-1, hsize)
        outputs = self.outputs2vocab(outputs)
        outputs = outputs.view(b, seq_len, self.vocab_size)
        return outputs

    def inference_direct(
        self,
        latent: torch.Tensor,
        embedding: nn.Module,
        tokenizer: Tokenizer,
        max_len: int,
    ) -> Tuple[List[str], torch.Tensor]:
        """Direct inference from latent space.

        Args:
            latent: latent tensor.
            embedding: embedding module.
            tokenizer: tokenizer.
            max_len: maximum sequence length.

        Returns:
            a tuple containing decoded strings and indices.
        """
        batch_size = latent.size(0)
        hidden = self.latent2hidden(latent)
        hidden = hidden.view(batch_size, self.hidden_factor, self.hidden_size)
        hidden = hidden.permute(1, 0, 2).contiguous()
        hidden = torch.tanh(hidden)
        input_sequence = torch.full(
            (batch_size, 1), tokenizer.sos_token_id, device=latent.device
        ).long()
        logits_list = []
        for t in range(max_len):
            input_embedding = embedding(input_sequence)
            output, hidden = self.rnn(input_embedding, hidden)
            logits = self.outputs2vocab(output)
            logits_list.append(logits)
            input_sequence = torch.argmax(logits, dim=-1)

        logits_tensor = torch.cat(logits_list, dim=1)
        token_indices = torch.argmax(logits_tensor, dim=-1)
        decoded_texts = []
        for index in range(batch_size):
            tokens = [
                tokenizer.convert_id_to_token(vocab_index.item())
                for vocab_index in token_indices[index]
            ]
            text = "".join(tokens).split()[0]
            decoded_texts.append(text)
        return decoded_texts, token_indices


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot product attention (adapted from Viswani et al.).

    Args:
        query: query tensor.
        key: key tensor.
        value: value tesor.
        mask: mask to apply on attention score. Defaults to None, a.k.a., no mask.
        dropout: dropout layer. Defaults to None, a.k.a., no dropout.

    Returns:
        a tuple containing the applied attention and the attention weights.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module: nn.Module, n: int) -> nn.Module:
    """Produce N identical layers (adapted from http://nlp.seas.harvard.edu/2018/04/03/attention.html).

    Args:
        module: a module.
        n: number of clones.

    Returns:
        a module list.
    """
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> torch.Tensor:
    """Mask out subsequent positions (adapted from http://nlp.seas.harvard.edu/2018/04/03/attention.html).

    Args:
        size: size of the attention matrix.

    Returns:
        the mask tensor.
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


class ListModule(torch.nn.Module):
    """Create single pytorch module from list of modules."""

    def __init__(self, *args) -> None:
        """Construct ListModule."""
        super().__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx: int) -> Any:
        """Get item from the module list.

        Args:
            idx: index of the item.

        Raises:
            IndexError: in case the index is out of range.

        Returns:
            the item.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError("index {} is out of range".format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self) -> Any:
        """An iterator over the module list values.

        Returns:
            the iterator over values.
        """
        return iter(self._modules.values())

    def __len__(self):
        """Length of the module list.

        Returns:
            the number of modules.
        """
        return len(self._modules)


class MultiHeadedAttention(nn.Module):
    """Multihead attention implementation (based on Vaswani et al.)."""

    def __init__(self, h, d_model, dropout=0.1) -> None:
        """Construct MultiHeadedAttention.

        Args:
            h: number of heads.
            d_model: model size.
            dropout: dropout rate. Defaults to 0.1.
        """
        super().__init__()
        assert d_model % h == 0
        # we assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Any:
        """Forward pass in the model.

        Args:
            query: query tensor.
            key: key tensor.
            value: value tesor.
            mask: mask to apply on attention score. Defaults to None, a.k.a., no mask.
            return_attn: whether to return the attention matrix instead of the linear layer output.
                Defaults to False, a.k.a, do not return attention.

        Returns:
            either the last layer output of the attention matrix.
        """
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            linear_layer(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for linear_layer, x in zip(self.linears, (query, key, value))  # type:ignore
        ]

        # 2) apply attention on all the projected vectors in batch
        x, self.attn = attention(  # type:ignore
            query, key, value, mask=mask, dropout=self.dropout
        )  # type:ignore

        # 3) "concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        if return_attn:
            return self.attn
        else:
            return self.linears[-1](x)  # type:ignore


class PositionwiseFeedForward(nn.Module):
    """Feed forward implementation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """Construct PositionwiseFeedForward.

        Args:
            d_model: model size.
            d_ff: feed forward size.
            dropout: dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass in the model.

        Args:
            x: input tensor.

        Returns:
            feed forward output.
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class ConvBottleneck(nn.Module):
    """Set of convolutional layers to reduce memory matrix to single latent vector."""

    def __init__(self, size: int, number_of_layers: int = 3) -> None:
        """Construct ConvBottleneck.

        Args:
            size: input size.
            number_of_layers: convolutional layers number. Defaults to 3.
        """
        super().__init__()
        conv_layers = []
        in_d = size
        first = True
        for i in range(number_of_layers):
            out_d = int((in_d - 64) // 2 + 64)
            if first:
                kernel_size = 9
                first = False
            else:
                kernel_size = 8
            if i == 2:
                out_d = 64
            conv_layers.append(
                nn.Sequential(nn.Conv1d(in_d, out_d, kernel_size), nn.MaxPool1d(2))
            )
            in_d = out_d
        self.conv_layers = ListModule(*conv_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass in the model.

        Args:
            x: input tensor.

        Returns:
            model output.
        """
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        return x


class DeconvBottleneck(nn.Module):
    """Set of deconvolutional layers to reshape latent vector back into memory matrix."""

    def __init__(self, size: int, seq_len: int, dim_factor: int) -> None:
        """Construct DeconvBottleneck.

        Args:
            size: size of the deconvolutional padding.
            seq_len: length of the sequence.
            dim_factor: dimensionality factor.
        """
        super().__init__()
        deconv_layers = []

        in_d = 64

        out_fac = 9 * dim_factor + 8
        out_fac = out_fac - 1 + 50 + 1
        diff_seq = out_fac - seq_len

        for i in range(3):
            out_d = (size - in_d) // 4 + in_d
            stride = 3
            padding = 3
            dilation = 1
            kernel_size = 11
            output_padding = 0
            if i == 2:
                out_d = size
                stride = 1
                dilation = 5
                if diff_seq % 2 == 0:
                    padding = int(diff_seq / 2)
                    output_padding = 0
                else:
                    padding = math.ceil(diff_seq / 2)
                    output_padding = 1

            deconv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        in_d,
                        out_d,
                        kernel_size,
                        dilation=dilation,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                    )
                )
            )
            in_d = out_d
        self.deconv_layers = ListModule(*deconv_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass in the model.

        Args:
            x: input tensor.

        Returns:
            model output.
        """
        for deconv in self.deconv_layers:
            x = F.relu(deconv(x))
        return x


class Embeddings(nn.Module):
    "Transforms input token id tensors to size d_model embeddings."

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """Costruct Embeddings.

        Args:
            d_model: size of the embedding vectors.
            vocab_size: size of the vocabulary.
        """
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass in the model.

        Args:
            x: input tensor.

        Returns:
            model output.
        """
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Static sinusoidal positional encoding layer."""

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        """Construct PositionalEncoding.

        Args:
            d_model: model size.
            dropout: dropout rate.
            max_len: maximum sequence length. Defaults to 5000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass in the model.

        Args:
            x: input tensor.

        Returns:
            model output.
        """
        x = x + torch.autograd.Variable(
            self.pe[:, : x.size(1)], requires_grad=False  # type:ignore
        )
        return self.dropout(x)


class TorchLayerNorm(nn.Module):
    """Layer normalization using torch BatchNorm1d."""

    def __init__(self, features: int, eps=1e-6) -> None:
        """Construct TorchLayerNorm.

        Args:
            features: number of features.
            eps: espilon to add to denominator for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.bn = nn.BatchNorm1d(features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass in the model.

        Args:
            x: input tensor.

        Returns:
            model output.
        """
        return self.bn(x)


class LayerNorm(nn.Module):
    """Custom layer normalization."""

    def __init__(self, features: int, eps=1e-6) -> None:
        """Construct LayerNorm.

        Args:
            features: number of features.
            eps: espilon to add to denominator for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass in the model.

        Args:
            x: input tensor.

        Returns:
            model output.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class SublayerConnection(nn.Module):
    """A residual connection followed by a layer normalization.

    Note for code simplicity the norm is first as opposed to last. A dropout layer
    is also applied.
    """

    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable) -> torch.Tensor:
        """Forward pass in the model.

        Args:
            x: input tensor.
            sublayer: a callable returning a tensor.

        Returns:
            model output.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerEncoder(nn.Module):
    """Base transformer encoder architecture."""

    def __init__(
        self,
        hidden_size: int,
        ff_size: int,
        seq_len: int,
        dropout: float,
        heads: int,
        n_layers_enc: int,
        vocab_size: int,
        bypass_bottleneck: bool,
    ) -> None:
        """Construct TransformerEncoder.

        Args:
            hidden_size: hidden size.
            ff_size: feed forward size.
            seq_len: sequence length.
            dropout: dropout rate.
            heads: number of heads.
            n_layers_enc: number of encoding layers.
            vocab_size: vocabulary size.
            bypass_bottleneck: whether the bottleneck should be by passed.
        """
        super().__init__()

        self.position = PositionalEncoding(hidden_size, dropout)
        self.embedding = nn.Sequential(
            Embeddings(hidden_size, vocab_size * 2), self.position
        )

        self.self_attn = MultiHeadedAttention(heads, hidden_size)
        self.feed_forward = PositionwiseFeedForward(hidden_size, ff_size, dropout)
        layer = TransformerEncoderLayer(
            hidden_size, seq_len, self.self_attn, self.feed_forward, dropout
        )
        self.layers = clones(layer, n_layers_enc)

        self.conv_bottleneck = ConvBottleneck(hidden_size)
        self.norm = LayerNorm(hidden_size)

        self.bypass_bottleneck = bypass_bottleneck
        conv_output_shape = self.calc_output_shape(seq_len, hidden_size)
        self.conv_output_len = conv_output_shape[1] * conv_output_shape[2]
        self.conv_output_shape = conv_output_shape

    def calc_output_shape(self, seq_len: int, hidden_size: int):
        """Compute output shape.

        Args:
            seq_len: sequence length.
            hidden_size: hidden size.

        Returns:
            convolutional bottleneck output shape.
        """
        x = torch.randn((1, hidden_size, seq_len))
        x_out = self.conv_bottleneck(x)
        return x_out.shape

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass in the model.

        Args:
            x: input tensor.
            mask: mask to apply in the attention layer.

        Returns:
            model output.
        """
        x = self.embedding(x)
        for _, attn_layer in enumerate(self.layers):  # type:ignore
            x = attn_layer(x, mask)
        mem = self.norm(x)
        mem = mem.permute(0, 2, 1)
        mem = self.conv_bottleneck(mem)
        mem = mem.contiguous().view(mem.size(0), -1)
        return mem


class TransformerEncoderLayer(nn.Module):
    """Self-attention/feedforward implementation."""

    def __init__(
        self,
        size: int,
        seq_len: int,
        self_attn: nn.Module,
        feed_forward: nn.Module,
        dropout: float,
    ) -> None:
        """Construct TransformerEncoderLayer.

        Args:
            size: model size.
            seq_len: sequence length.
            self_attn: self-attention layer.
            feed_forward: feed forward layer.
            dropout: droupout rate.
        """
        super().__init__()
        self.size = size
        self.seq_len = seq_len
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 2)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, return_attn: bool = False
    ) -> Any:
        """Forward pass in the model.

        Args:
            x: input tensor.
            mask: mask to apply in the attention layer.
            return_attn: whether to return the attention together with the output.
                Defaults to False, return only encoder output.

        Returns:
            model output.
        """
        if return_attn:
            attn = self.self_attn(x, x, x, mask, return_attn=True)
            x = self.sublayer[0](  # type:ignore
                x, lambda x: self.self_attn(x, x, x, mask)
            )
            return self.sublayer[1](x, self.feed_forward), attn  # type:ignore
        else:
            x = self.sublayer[0](  # type:ignore
                x, lambda x: self.self_attn(x, x, x, mask)
            )
            return self.sublayer[1](x, self.feed_forward)  # type:ignore


class TransformerDecoder(nn.Module):
    """Base transformer decoder architecture."""

    def __init__(
        self,
        hidden_size: int,
        ff_size: int,
        seq_len: int,
        dropout: float,
        heads: int,
        n_layers_dec: int,
        latent_size: int,
        vocab_size: int,
        bypass_bottleneck: bool,
        deconv_shape: Tuple[int, int, int],
    ) -> None:
        """Construct TransformerDecoder.

        Args:
            hidden_size: hidden size.
            ff_size: feed forward size.
            seq_len: sequence length.
            dropout: dropout rate.
            heads: number of heads.
            n_layers_enc: number of encoding layers.
            latent_size: latent size.
            vocab_size: vocabulary size.
            bypass_bottleneck: whether the bottleneck should be by passed.
            deconv_shape: shape of the deconvoluted samples. A tuple with three
                dimensions.
        """
        super().__init__()

        self.position = PositionalEncoding(hidden_size, dropout)
        self.embedding = nn.Sequential(
            Embeddings(hidden_size, vocab_size), self.position
        )
        self.attn_enc = MultiHeadedAttention(heads, hidden_size)
        self.ff_enc = PositionwiseFeedForward(hidden_size, ff_size, dropout)
        self.attn_dec_1 = MultiHeadedAttention(heads, hidden_size)
        self.attn_dec_2 = MultiHeadedAttention(heads, hidden_size)

        self.ff_dec = PositionwiseFeedForward(hidden_size, ff_size, dropout)

        encoder_layers = TransformerEncoderLayer(
            hidden_size, seq_len, self.attn_enc, self.ff_enc, dropout
        )
        decoder_layers = TransformerDecoderLayer(
            hidden_size,
            seq_len,
            self.attn_dec_1,
            self.attn_dec_2,
            self.ff_dec,
            dropout,
        )

        self.final_encodes = clones(encoder_layers, 1)
        self.layers = clones(decoder_layers, n_layers_dec)
        self.norm = LayerNorm(hidden_size)
        self.bypass_bottleneck = bypass_bottleneck
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.outputs2vocab = torch.nn.Linear(hidden_size, vocab_size)
        self.deconv_shape = deconv_shape
        self.deconv_bottleneck = DeconvBottleneck(
            hidden_size, seq_len=seq_len, dim_factor=deconv_shape[2]
        )
        self.linear = nn.Linear(latent_size, deconv_shape[2] * deconv_shape[1])

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass in the model.

        Args:
            x: input tensor.
            mem: memory tensor.
            src_mask: source sequence mask.
            tgt_mask: target sequence mask.

        Returns:
            model output.
        """
        x = self.embedding(x)
        if not self.bypass_bottleneck:
            mem = F.relu(self.linear(mem))
            mem = mem.view(-1, 64, self.deconv_shape[2])
            mem = self.deconv_bottleneck(mem)
            mem = mem.permute(0, 2, 1)
        for final_encode in self.final_encodes:  # type:ignore
            mem = final_encode(mem, src_mask)
        mem = self.norm(mem)
        for _, attn_layer in enumerate(self.layers):  # type:ignore
            x = attn_layer(x, mem, mem, src_mask, tgt_mask)
        x = self.norm(x)
        x = self.outputs2vocab(F.relu(x))
        return x

    def inference_direct(
        self,
        latent: torch.Tensor,
        mask_lengths: torch.Tensor,
        tokenizer: Tokenizer,
    ) -> Tuple[List[str], torch.Tensor]:
        """Direct inference from latent space.

        Args:
            latent: latent tensor.
            mask_lengths: masking tensor.
            tokenizer: tokenizer.

        Returns:
            a tuple containing decoded strings and indices.
        """
        device = get_device_from_tensor(latent)
        batch_size = latent.size(0)
        token_indices = torch.full(
            (batch_size, 1), tokenizer.sos_token_id, device=device
        ).long()

        src_mask = torch.zeros((latent.shape[0], 1, self.seq_len), device=device)

        for index in range(mask_lengths.shape[0]):
            mask_len = int(mask_lengths[index].item())
            src_mask[index, :, :mask_len] = torch.ones((1, 1, mask_len), device=device)
        self.eval()
        for i in range(self.seq_len - 1):
            trg_mask = subsequent_mask(token_indices.size(1)).long().to(device)
            logits = self(
                torch.autograd.Variable(token_indices), latent, src_mask, trg_mask
            )

            prob = F.softmax(logits[:, i, :], dim=-1)
            _, next_token = torch.max(prob, dim=1)

            next_token = next_token.unsqueeze(1)
            token_indices = torch.cat([token_indices, next_token], dim=1)

        decoded_texts = []
        for index in range(batch_size):
            tokens = [
                tokenizer.convert_id_to_token(vocab_index.item())
                for vocab_index in token_indices[index]
            ]
            text = "".join(tokens).split()[0]
            decoded_texts.append(text)
        return decoded_texts, token_indices


class TransformerDecoderLayer(nn.Module):
    """Self-attention/source-attention/feedforward implementation."""

    def __init__(
        self,
        size: int,
        seq_len: int,
        self_attn: nn.Module,
        src_attn: nn.Module,
        feed_forward: nn.Module,
        dropout: float,
    ) -> None:
        """Construct TransformerDecoderLayer.

        Args:
            size: model size.
            seq_len: sequence length.
            self_attn: self-attention layer.
            src_attn: source attention layer.
            feed_forward: feed forward layer.
            dropout: droupout rate.
        """
        super().__init__()
        self.size = size
        self.tgt_len = seq_len
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 3)

    def forward(
        self,
        x: torch.Tensor,
        memory_key: torch.Tensor,
        memory_val: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        return_attn: bool = False,
    ) -> Any:
        """Forward pass in the model.

        Args:
            x: input tensor
            memory_key: memory key tensor.
            memory_val: memory value tensor.s
            src_mask: mask to apply in the source attention layer.
            tgt_mask: mask to apply in the target attention layer.
            return_attn: whether to return the attention together with the output.
                Defaults to False, return only encoder output.

        Returns:
            model output.
        """
        m_key = memory_key
        m_val = memory_val
        if return_attn:
            x = self.sublayer[0](  # type:ignore
                x, lambda x: self.self_attn(x, x, x, tgt_mask)
            )
            src_attn = self.src_attn(x, m_key, m_val, src_mask, return_attn=True)
            x = self.sublayer[1](  # type:ignore
                x, lambda x: self.src_attn(x, m_key, m_val, src_mask)
            )
            return self.sublayer[2](x, self.feed_forward), src_attn  # type:ignore
        else:
            x = self.sublayer[0](  # type:ignore
                x, lambda x: self.self_attn(x, x, x, tgt_mask)
            )
            x = self.sublayer[1](  # type:ignore
                x, lambda x: self.src_attn(x, m_key, m_val, src_mask)
            )
            return self.sublayer[2](x, self.feed_forward)  # type:ignore
