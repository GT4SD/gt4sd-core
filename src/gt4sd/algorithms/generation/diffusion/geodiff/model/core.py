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
from typing import Any, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.modeling_utils import ModelMixin
from torch import nn
from torch_geometric.data import Data

from .layers import (
    GINEncoder,
    MLPEdgeEncoder,
    MoleculeGNNOutput,
    MultiLayerPerceptron,
    SchNetEncoder,
)
from .utils import (
    assemble_atom_pair_feature,
    clip_norm,
    extend_graph_order_radius,
    get_distance,
    graph_field_network,
    is_local_edge,
)


class MoleculeGNN(ModelMixin, ConfigMixin):
    """Graph Neural Network Model for molecule conformation.

    Code adapted from colab:
        https://colab.research.google.com/drive/1pLYYWQhdLuv1q-JtEHGZybxp2RBF8gPs#scrollTo=-3-P4w5sXkRU written by Nathan Lambert.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_convs: int = 6,
        num_convs_local: int = 4,
        cutoff: float = 10.0,
        mlp_act: str = "relu",
        edge_order: int = 3,
        edge_encoder: str = "mlp",
        smooth_conv: bool = True,
    ) -> None:
        """Constructor for MoleculeGNN.
        Args:
            hidden_dim: hidden dimension of the GNN.
            num_convs: number of GNN layers.
            num_convs_local: number of GNN layers for local edges.
            cutoff: nutoff radius for the GNN.
            mlp_act: activation function for the MLP.
            edge_order: order of the edge features.
            edge_encoder: type of edge encoder.
            smooth_conv: whether to use smooth convolution.
        """
        super().__init__()
        self.cutoff = cutoff
        self.edge_encoder = edge_encoder
        self.edge_order = edge_order

        # edge_encoder: Takes both edge type and edge length as input and outputs a vector [Note]: node embedding is done in SchNetEncoder
        self.edge_encoder_global = MLPEdgeEncoder(
            hidden_dim, mlp_act
        )  # get_edge_encoder(config)
        self.edge_encoder_local = MLPEdgeEncoder(
            hidden_dim, mlp_act
        )  # get_edge_encoder(config)

        # the graph neural network that extracts node-wise features.
        self.encoder_global = SchNetEncoder(
            hidden_channels=hidden_dim,
            num_filters=hidden_dim,
            num_interactions=num_convs,
            edge_channels=self.edge_encoder_global.out_channels,
            cutoff=cutoff,
            smooth=smooth_conv,
        )
        self.encoder_local = GINEncoder(
            hidden_dim=hidden_dim,
            num_convs=num_convs_local,
        )

        # `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs
        # gradients w.r.t. edge_length (out_dim = 1).
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * hidden_dim, [hidden_dim, hidden_dim // 2, 1], activation=mlp_act
        )

        self.grad_local_dist_mlp = MultiLayerPerceptron(
            2 * hidden_dim, [hidden_dim, hidden_dim // 2, 1], activation=mlp_act
        )

        # incorporate parameters together
        self.model_global = nn.ModuleList(
            [self.edge_encoder_global, self.encoder_global, self.grad_global_dist_mlp]
        )
        self.model_local = nn.ModuleList(
            [self.edge_encoder_local, self.encoder_local, self.grad_local_dist_mlp]
        )

    def _forward(
        self,
        atom_type: torch.Tensor,
        pos: torch.Tensor,
        bond_index: torch.Tensor,
        bond_type: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor = None,
        edge_type: torch.Tensor = None,
        edge_length: int = None,
        return_edges: bool = False,
        extend_order: bool = True,
        extend_radius: bool = True,
        is_sidechain: bool = None,
    ) -> Tuple[Any, ...]:
        """Forward pass for edges features.

        Args:
            atom_type:  types of atoms, (N, ).
            pos:        positions of atoms, (N, 3).
            bond_index: indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  bond types, (E, ).
            batch:      node index to graph index, (N, ).
            edge_index: indices of edges (extended, radius-graph), (2, E'). If None, determined by extend_graph_order_radius.
            edge_type: edge types, (E', ). If None, determined by extend_graph_order_radius.
            edge_length: edge lengths, (E', ). If None, determined by extend_graph_order_radius.
            return_edges: whether to return edge_index, edge_type, edge_length.
            extend_order: whether to extend the graph by bond order.
            extend_radius: whether to extend the graph by radius.
            is_sidechain: whether the atom is a sidechain atom.

        Returns:
            tuple with local and global invariant edge features.
                If return_edges is True, it also returns edge_index, edge_type, edge_length, local_edge_index.

        """
        n = atom_type.size(0)
        if edge_index is None or edge_type is None or edge_length is None:
            edge_index, edge_type = extend_graph_order_radius(
                num_nodes=n,
                pos=pos,
                edge_index=bond_index,
                edge_type=bond_type,
                batch=batch,
                order=self.edge_order,
                cutoff=self.cutoff,
                extend_order=extend_order,
                extend_radius=extend_radius,
                is_sidechain=is_sidechain,
            )
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
        local_edge_mask = is_local_edge(edge_type)  # (E, )

        # with the parameterization of NCSNv2
        # DDPM loss implicit handle the noise variance scale conditioning
        sigma_edge = torch.ones(
            size=(edge_index.size(1), 1), device=pos.device  # type: ignore
        )  # (E, 1)

        # Encoding global
        edge_attr_global = self.edge_encoder_global(
            edge_length=edge_length, edge_type=edge_type
        )  # Embed edges

        # Global
        node_attr_global = self.encoder_global(
            z=atom_type,
            edge_index=edge_index,
            edge_length=edge_length,
            edge_attr=edge_attr_global,
        )
        # Assemble pairwise features
        h_pair_global = assemble_atom_pair_feature(
            node_attr=node_attr_global,
            edge_index=edge_index,
            edge_attr=edge_attr_global,
        )  # (E_global, 2H)
        # Invariant features of edges (radius graph, global)
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global) * (
            1.0 / sigma_edge
        )  # (E_global, 1)

        # Encoding local
        edge_attr_local = self.edge_encoder_global(
            edge_length=edge_length, edge_type=edge_type
        )  # Embed edges
        # edge_attr += temb_edge

        # Local
        node_attr_local = self.encoder_local(
            z=atom_type,
            edge_index=edge_index[:, local_edge_mask],  # type: ignore
            edge_attr=edge_attr_local[local_edge_mask],
        )
        # Assemble pairwise features
        h_pair_local = assemble_atom_pair_feature(
            node_attr=node_attr_local,
            edge_index=edge_index[:, local_edge_mask],  # type: ignore
            edge_attr=edge_attr_local[local_edge_mask],
        )  # (E_local, 2H)

        # Invariant features of edges (bond graph, local)
        if isinstance(sigma_edge, torch.Tensor):
            edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (
                1.0 / sigma_edge[local_edge_mask]
            )  # (E_local, 1)
        else:
            edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (
                1.0 / sigma_edge
            )  # (E_local, 1)

        if return_edges:
            return (
                edge_inv_global,
                edge_inv_local,
                edge_index,
                edge_type,
                edge_length,
                local_edge_mask,
            )
        else:
            return (edge_inv_global, edge_inv_local)

    def forward(
        self,
        sample: Data,
        timestep: Union[float, int],
        return_dict: bool = True,
        sigma: float = 1.0,
        global_start_sigma: float = 0.5,
        w_global: float = 1.0,
        extend_order=False,
        extend_radius=True,
        clip_local=None,
        clip_global: float = 1000.0,
    ) -> Union[MoleculeGNNOutput, Tuple]:
        """Forward pass for the model.

        Args:
            sample: packed torch geometric object.
            timestep timesteps.
            return_dict: whether or not to return a MoleculeGNNOutput instead of a plain tuple.
            sigma: the noise variance scale.
            global_start_sigma: the noise variance scale for global edges.
            w_global: the weight for global edges.
            extend_order: whether to extend the graph by bond order.
            extend_radius: whether to extend the graph by radius.
            clip_local: the clip value for local edges.
            clip_global: the clip value for global edges.

        Returns:
            dict with samples. If return_dict is false, return positions.
        """

        # unpack sample
        atom_type = sample.atom_type
        bond_index = sample.edge_index
        bond_type = sample.edge_type
        pos = sample.pos

        (
            edge_inv_global,
            edge_inv_local,
            edge_index,
            _,
            edge_length,
            local_edge_mask,
        ) = self._forward(
            atom_type=atom_type,
            pos=sample.pos,
            bond_index=bond_index,
            bond_type=bond_type,
            batch=sample.batch,
            return_edges=True,
            extend_order=extend_order,
            extend_radius=extend_radius,
        )  # (E_global, 1), (E_local, 1)

        # Important equation in the paper for equivariant features - eqns 5-7 of GeoDiff
        node_eq_local = graph_field_network(
            edge_inv_local,
            pos,
            edge_index[:, local_edge_mask],
            edge_length[local_edge_mask],
        )
        if clip_local is not None:
            node_eq_local = clip_norm(node_eq_local, limit=clip_local)

        # Global
        if sigma < global_start_sigma:
            edge_inv_global = edge_inv_global * (
                1 - local_edge_mask.view(-1, 1).float()
            )
            node_eq_global = graph_field_network(
                edge_inv_global, pos, edge_index, edge_length
            )
            node_eq_global = clip_norm(node_eq_global, limit=clip_global)
        else:
            node_eq_global = 0

        # Sum
        eps_pos = node_eq_local + node_eq_global * w_global

        if not return_dict:
            return (-eps_pos,)

        return MoleculeGNNOutput(sample=torch.FloatTensor(-eps_pos).to(pos.device))  # type: ignore
