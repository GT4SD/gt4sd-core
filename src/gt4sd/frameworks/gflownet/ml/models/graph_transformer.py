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
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch_geometric.data as gd
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops

from ...envs.graph_building_env import GraphActionCategorical, GraphBuildingEnvContext


def mlp(n_in, n_hid, n_out, n_layer, act=nn.LeakyReLU):
    n = [n_in] + [n_hid] * n_layer + [n_out]
    return nn.Sequential(
        *sum([[nn.Linear(n[i], n[i + 1]), act()] for i in range(n_layer + 1)], [])[:-1]
    )


class GraphTransformer(nn.Module):
    """GraphTransformer."""

    def __init__(
        self,
        configuration: Dict[str, Any],
        context: GraphBuildingEnvContext,
        x_dim=64,
        e_dim=64,
        g_dim=64,
        num_emb=64,
        num_layers=3,
        num_heads=2,
    ) -> None:
        """Construct GraphTransformer.

        Code adapted from: https://github.com/recursionpharma/gflownet/tree/trunk/src/gflownet/models.

        Args:
            configuration: model configuration.
            context: context environment.
            x_dim: dimension of input node features.
            e_dim: dimension of input edge features.
            g_dim: dimension of input graph features.
            num_emb: dimension of embedding layer.
            num_layers: number of layers in the graph transformer.
            num_heads: number of heads in the graph transformer.
        """

        super().__init__()
        self.num_layers = num_layers
        self.name = "graph_transformer"

        self.x2h = mlp(x_dim, num_emb, num_emb, 2)
        self.e2h = mlp(e_dim, num_emb, num_emb, 2)
        self.c2h = mlp(g_dim, num_emb, num_emb, 2)
        self.graph2emb = nn.ModuleList(
            sum(
                [
                    [
                        gnn.GENConv(
                            num_emb, num_emb, num_layers=1, aggr="add", norm=None
                        ),
                        gnn.TransformerConv(
                            num_emb * 2, num_emb, edge_dim=num_emb, heads=num_heads
                        ),
                        nn.Linear(num_heads * num_emb, num_emb),
                        gnn.LayerNorm(num_emb, affine=False),
                        mlp(num_emb, num_emb * 4, num_emb, 1),
                        gnn.LayerNorm(num_emb, affine=False),
                    ]
                    for i in range(self.num_layers)
                ],
                [],
            )
        )

    def forward(
        self, g: gd.Batch, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            g: graph data.
            cond: conditioning.

        Returns:
            embeddings and pooled features.
        """
        o = self.x2h(g.x)
        e = self.e2h(g.edge_attr)
        c = self.c2h(cond)
        num_total_nodes = g.x.shape[0]
        # Augment the edges with a new edge to the conditioning
        # information node. This new node is connected to every node
        # within its graph.
        u, v = torch.arange(num_total_nodes, device=o.device), g.batch + num_total_nodes
        aug_edge_index = torch.cat(
            [g.edge_index, torch.stack([u, v]), torch.stack([v, u])], 1
        )
        e_p = torch.zeros((num_total_nodes * 2, e.shape[1]), device=g.x.device)
        e_p[:, 0] = 1  # Manually create a bias term
        aug_e = torch.cat([e, e_p], 0)
        aug_edge_index, aug_e = add_self_loops(aug_edge_index, aug_e, "mean")
        aug_batch = torch.cat([g.batch, torch.arange(c.shape[0], device=o.device)], 0)

        # Append the conditioning information node embedding to o
        o = torch.cat([o, c], 0)
        for i in range(self.num_layers):
            # Run the graph transformer forward
            gen, trans, linear, norm1, ff, norm2 = self.graph2emb[i * 6 : (i + 1) * 6]
            agg = gen(o, aug_edge_index, aug_e)
            o = norm1(
                o + linear(trans(torch.cat([o, agg], 1), aug_edge_index, aug_e)),
                aug_batch,
            )
            o = norm2(o + ff(o), aug_batch)

        glob = torch.cat(
            [gnn.global_mean_pool(o[: -c.shape[0]], g.batch), o[-c.shape[0] :], c], 1
        )
        o_final = torch.cat([o[: -c.shape[0]], c[g.batch]], 1)
        return o_final, glob


class GraphTransformerGFN(nn.Module):
    """GraphTransformerGFN."""

    def __init__(
        self,
        configuration: Dict[str, Any],
        context: GraphBuildingEnvContext,
        num_emb=64,
        num_layers=3,
        num_heads=2,
    ) -> None:
        """Construct GraphTransformerGFN.

        Args:
            configuration: model configuration.
            context: context environment.
            num_emb: dimension of embedding layer.
            num_layers: number of layers in the graph transformer.
            num_heads: number of heads in the graph transformer.
        """

        super().__init__()
        self.name = "graph_transformer_gfn"
        self.transf = GraphTransformer(
            configuration=configuration,
            context=context,
            x_dim=context.num_node_dim,
            e_dim=context.num_edge_dim,
            g_dim=context.num_cond_dim,
            num_emb=num_emb,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        num_final = num_emb * 2
        num_mlp_layers = 0
        self.emb2add_edge = mlp(num_final, num_emb, 1, num_mlp_layers)
        self.emb2add_node = mlp(
            num_final, num_emb, context.num_new_node_values, num_mlp_layers
        )
        self.emb2set_node_attr = mlp(
            num_final, num_emb, context.num_node_attr_logits, num_mlp_layers
        )
        self.emb2set_edge_attr = mlp(
            num_final, num_emb, context.num_edge_attr_logits, num_mlp_layers
        )
        self.emb2stop = mlp(num_emb * 3, num_emb, 1, num_mlp_layers)
        self.emb2reward = mlp(num_emb * 3, num_emb, 1, num_mlp_layers)
        self.log_z = mlp(context.num_cond_dim, num_emb * 2, 1, 2)
        self.action_type_order = context.action_type_order

    def forward(self, g: gd.Batch, cond: torch.Tensor):
        """Forward pass. Given a graph and a conditioning, return the action logits and rewards.

        Args:
            g: graph data.
            cond: conditioning.

        Returns:
            categorical action and rewards.
        """
        node_embeddings, graph_embeddings = self.transf(g, cond)
        ne_row, ne_col = g.non_edge_index
        # On `::2`, edges are duplicated to make graphs undirected, only take the even ones
        e_row, e_col = g.edge_index[:, ::2]
        cat = GraphActionCategorical(
            g,
            logits=[
                self.emb2stop(graph_embeddings),
                self.emb2add_node(node_embeddings),
                self.emb2set_node_attr(node_embeddings),
                self.emb2add_edge(node_embeddings[ne_row] + node_embeddings[ne_col]),
                self.emb2set_edge_attr(node_embeddings[e_row] + node_embeddings[e_col]),
            ],
            keys=[None, "x", "x", "non_edge_index", "edge_index"],
            types=self.action_type_order,
        )
        return cat, self.emb2reward(graph_embeddings)
