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
import logging
from dataclasses import dataclass
from typing import Callable, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils import BaseOutput
from torch import Tensor, nn
from torch.nn import Embedding, Linear, Module, ModuleList, Sequential
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_sparse import SparseTensor

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class MoleculeGNNOutput(BaseOutput):
    """Hidden states output. Output of last layer of model."""

    sample: torch.FloatTensor


class MultiLayerPerceptron(nn.Module):
    """Multi-layer Perceptron. Note there is no activation or dropout in the last layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        dropout: float = 0,
    ) -> None:
        """Initialize multi-layer perceptron.

        Args:
            input_dim: input dimension
            hidden_dim: hidden dimensions
            activation: activation function
            dropout: dropout rate
        """
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            logger.info(
                f"Warning, activation passed {activation} is not string and ignored"
            )
            self.activation = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None  # type: ignore

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor of shape (batch_size, input_dim)

        Returns:
            output mlp.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


class ShiftedSoftplus(torch.nn.Module):
    """Shifted softplus activation function."""

    def __init__(self) -> None:
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    """CFConv layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        mlp: Callable,
        cutoff: float,
        smooth: bool,
    ) -> None:
        """Construct a CFConv layer.

        Args:
            in_channels: size of each input.
            out_channels: size of each output.
            num_filters: number of filters.
            mlp: mlp hidden dimensions.
            cutoff: cutoff distance.
            smooth: whether to use smooth cutoff.
        """
        super(CFConv, self).__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = mlp
        self.cutoff = cutoff
        self.smooth = smooth

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(
        self, x: torch.Tensor, edge_index, edge_length, edge_attr
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor.
            edge_index: edge indices.
            edge_length: edge lengths.
            edge_attr: edge attributes.

        Returns:
            output tensor.
        """
        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * np.pi / self.cutoff) + 1.0)
            C = (
                C * (edge_length <= self.cutoff) * (edge_length >= 0.0)
            )  # Modification: cutoff
        else:
            C = (edge_length <= self.cutoff).float()
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: torch.Tensor, W) -> torch.Tensor:
        return x_j * W


class InteractionBlock(torch.nn.Module):
    """Interaction block."""

    def __init__(
        self,
        hidden_channels: int,
        num_gaussians: int,
        num_filters: int,
        cutoff: float,
        smooth: bool,
    ) -> None:
        """Construct an interaction block.

        Args:
            hidden_channels: number of hidden channels.
            num_gaussians: number of gaussians.
            num_filters: number of filters.
            cutoff: cutoff distance.
            smooth: whether to use smooth cutoff.
        """
        super(InteractionBlock, self).__init__()
        mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(
            hidden_channels, hidden_channels, num_filters, mlp, cutoff, smooth
        )
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(
        self, x: torch.Tensor, edge_index, edge_length, edge_attr
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor.
            edge_index: edge indices.
            edge_length: edge lengths.
            edge_attr: edge attributes.

        Returns:
            output tensor.
        """
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNetEncoder(Module):
    """SchNet encoder."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        edge_channels: int = 100,
        cutoff: float = 10.0,
        smooth: bool = False,
    ) -> None:
        """Construct a SchNet encoder.

        Args:
            hidden_channels: number of hidden channels.
            num_filters: number of filters.
            num_interactions: number of interactions.
            edge_channels: number of edge channels.
            cutoff: cutoff distance.
            smooth: whether to use smooth cutoff.
        """
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff

        self.embedding = Embedding(100, hidden_channels, max_norm=10.0)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                hidden_channels, edge_channels, num_filters, cutoff, smooth
            )
            self.interactions.append(block)

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_length: torch.Tensor,
        edge_attr: torch.Tensor,
        embed_node: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            z: input tensor.
            edge_index: edge indices.
            edge_length: edge lengths.
            edge_attr: edge attributes.
            embed_node: whether to embed node.

        Returns:
            output tensor.
        """
        if embed_node:
            assert z.dim() == 1 and z.dtype == torch.long
            h = self.embedding(z)
        else:
            h = z
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)

        return h


class GINEConv(MessagePassing):
    """GINECONV layer.
    Custom class of the graph isomorphism operator from the "How Powerful are Graph Neural Networks?
    https://arxiv.org/abs/1810.00826 paper. Note that this implementation has the added option of a custom activation.
    """

    def __init__(
        self,
        mlp: Callable,
        eps: float = 0.0,
        train_eps: bool = False,
        activation: str = "softplus",
        **kwargs,
    ) -> None:
        """Construct a GINEConv layer.

        Args:
            mlp: MLP.
            eps: epsilon.
            train_eps: whether to train epsilon.
            activation: activation function.
        """
        super(GINEConv, self).__init__(aggr="add", **kwargs)
        self.nn = mlp
        self.initial_eps = eps

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor.
            edge_index: edge indices.
            edge_attr: edge attributes.
            size: size.

        Returns:
            output tensor.
        """

        if isinstance(x, torch.Tensor):
            x = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, torch.Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Message function.

        Args:
            x_j: input tensor.
            edge_attr: edge attributes.

        Returns:
            message passing aggregation.
        """
        if self.activation:
            return self.activation(x_j + edge_attr)
        else:
            return x_j + edge_attr

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


class GINEncoder(torch.nn.Module):
    """GIN encoder."""

    def __init__(
        self,
        hidden_dim: int,
        num_convs: int = 3,
        activation: str = "relu",
        short_cut: bool = True,
        concat_hidden: bool = False,
    ) -> None:
        """Construct a GIN encoder.

        Args:
            hidden_dim: number of hidden channels.
            num_convs: number of convolutions.
            activation: activation function.
            short_cut: whether to use short cut.
            concat_hidden: whether to concatenate hidden.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_convs = num_convs
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.node_emb = nn.Embedding(100, hidden_dim)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            self.convs.append(
                GINEConv(
                    MultiLayerPerceptron(
                        hidden_dim, [hidden_dim, hidden_dim], activation=activation
                    ),
                    activation=activation,
                )
            )

    def forward(
        self, z: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        args:
            z: input tensor.
            edge_index: edge indices.
            edge_attr: edge attributes.

        returns:
            graph with node feature.
        """

        node_attr = self.node_emb(z)  # (num_node, hidden)

        hiddens = []
        conv_input = node_attr  # (num_node, hidden)

        for conv_idx, conv in enumerate(self.convs):
            hidden = conv(conv_input, edge_index, edge_attr)
            if conv_idx < len(self.convs) - 1 and self.activation is not None:
                hidden = self.activation(hidden)
            assert hidden.shape == conv_input.shape
            if self.short_cut and hidden.shape == conv_input.shape:
                hidden += conv_input

            hiddens.append(hidden)
            conv_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        return node_feature


class MLPEdgeEncoder(Module):
    """MLP edge encoder."""

    def __init__(self, hidden_dim: int = 100, activation: str = "relu") -> None:
        """Construct a MLP edge encoder.

        Args:
            hidden_dim: number of hidden channels.
            activation: activation function.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = Embedding(100, embedding_dim=self.hidden_dim)
        self.mlp = MultiLayerPerceptron(
            1, [self.hidden_dim, self.hidden_dim], activation=activation
        )

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(
        self, edge_length: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            edge_length: The length of edges.
            edge_type: The type of edges.
        Returns:
            the output representation of edges.
        """
        d_emb = self.mlp(edge_length)  # (num_edge, hidden_dim)
        edge_attr = self.bond_emb(edge_type)  # (num_edge, hidden_dim)
        return d_emb * edge_attr  # (num_edge, hidden)
