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
import copy
import enum
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch_geometric.data as gd
from rdkit.Chem import Mol
from torch_scatter import scatter, scatter_max


class Graph(nx.Graph):
    """Subclassing nx.Graph for debugging purposes."""

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'<{list(self.nodes)}, {list(self.edges)}, {list(self.nodes[i]["v"] for i in self.nodes)}>'


def graph_without_edge(g: nx.Graph, e: nx.Graph) -> nx.Graph:
    """Build a new graph without an edge.

    Args:
        g: a graph.
        e: an edge to remove.

    Returns:
        a new graph without an edge.
    """

    gp = g.copy()
    gp.remove_edge(*e)
    return gp


def graph_without_node(g: nx.Graph, n: nx.Graph) -> nx.Graph:
    """Build a new graph without a node.

    Args:
        g: a graph.
        n: a node to remove.

    Returns:
        a new graph without a node.
    """

    gp = g.copy()
    gp.remove_node(n)
    return gp


def graph_without_node_attr(g: nx.Graph, n: nx.Graph, a: Any) -> nx.Graph:
    """Build a new graph without a node attribute.

    Args:
        g: a graph.
        n: a node.
        a: a node attribute.

    Returns:
        a new graph without a node attribute.
    """

    gp = g.copy()
    del gp.nodes[n][a]
    return gp


def graph_without_edge_attr(g: nx.Graph, e: nx.Graph, a: Any) -> nx.Graph:
    """Build a new graph without an edge attribute.

    Args:
        g: a graph.
        e: an edge.
        a: an edge attribute.

    Returns:
        a new graph without an edge attribute.
    """

    gp = g.copy()
    del gp.edges[e][a]
    return gp


class GraphActionType(enum.Enum):
    """Action types for graph building environments."""

    # Forward actions
    Stop = enum.auto()
    AddNode = enum.auto()
    AddEdge = enum.auto()
    SetNodeAttr = enum.auto()
    SetEdgeAttr = enum.auto()
    # Backward actions
    RemoveNode = enum.auto()
    RemoveEdge = enum.auto()
    RemoveNodeAttr = enum.auto()
    RemoveEdgeAttr = enum.auto()


class GraphAction:
    """Actions on a graph environment for gflownet."""

    def __init__(
        self,
        action: GraphActionType,
        source: int = None,
        target: int = None,
        value: Any = None,
        attr: str = None,
        relabel: int = None,
    ):
        """Initialize a single graph-building action.

        Args:
            action: the action type.
            source: the source node this action is applied on.
            target: the target node (i.e. if specified this is an edge action).
            value: the value (e.g. new node type) applied.
            attr: the set attribute of a node/edge.
            relabel: for AddNode actions, relabels the new node with that id.
        """
        self.action = action
        self.source = source
        self.target = target
        self.attr = attr
        self.value = value
        self.relabel = relabel  # TODO: deprecate this?

    def __repr__(self):
        attrs = ", ".join(
            str(i)
            for i in [self.source, self.target, self.attr, self.value]
            if i is not None
        )
        return f"<{self.action}, {attrs}>"


class GraphBuildingEnv:
    """A Graph building environment which induces a dag state space compatible with gflownet."""

    def __init__(
        self,
        allow_add_edge: bool = True,
        allow_node_attr: bool = True,
        allow_edge_attr: bool = True,
    ):
        """Initialize a GraphBuildingEnv.

        Supports forward and backward actions, with a `parents` function that list parents of
        forward actions. Edges and nodes can have attributes added to them in a key:value style.
        Edges and nodes are created with _implicit_ default attribute
        values (e.g. chirality, single/double bondness) so that:
            - an agent gets to do an extra action to set that attribute, but only
            if it is still default-valued (DAG property preserved)
            - we can generate a legal action for any attribute that isn't a default one.

        Code adapted from: https://github.com/recursionpharma/gflownet/tree/trunk/src/gflownet/envs

        Args:
            allow_add_edge: if True, allows this action and computes AddEdge parents (i.e. if False, this
                env only allows for tree generation).
            allow_node_attr: if True, allows this action and computes SetNodeAttr parents.
            allow_edge_attr: if True, allows this action and computes SetEdgeAttr parents.
        """

        self.allow_add_edge = allow_add_edge
        self.allow_node_attr = allow_node_attr
        self.allow_edge_attr = allow_edge_attr

    def new(self) -> nx.Graph:
        return Graph()

    def step(self, g: Graph, action: GraphAction) -> Graph:
        """Step forward the given graph state with an action

        Args:
            g: the graph to be modified.
            action: the action taken on the graph, indices must match.

        Returns:
            gp: the new graph.
        """
        gp = g.copy()
        if action.action is GraphActionType.AddEdge:
            a, b = action.source, action.target

            assert self.allow_add_edge
            assert a in g and b in g

            if a > b:  # type: ignore
                a, b = b, a

            assert a != b
            assert not g.has_edge(a, b)

            # Ideally the FA underlying this must only be able to send
            # create_edge actions which respect this a<b property (or
            # its inverse!) , otherwise symmetry will be broken
            # because of the way the parents method is written
            gp.add_edge(a, b)

        elif action.action is GraphActionType.AddNode:
            if len(g) == 0:
                assert action.source == 0  # TODO: this may not be useful
                gp.add_node(0, v=action.value)
            else:
                assert action.source in g.nodes
                e = [action.source, max(g.nodes) + 1]
                if action.relabel is not None:
                    raise ValueError("deprecated")
                # if kw and 'relabel' in kw:
                #     e[1] = kw['relabel']  # for `parent` consistency, allow relabeling
                assert not g.has_edge(*e)
                gp.add_node(e[1], v=action.value)
                gp.add_edge(*e)

        elif action.action is GraphActionType.SetNodeAttr:
            assert self.allow_node_attr
            assert action.source in gp.nodes
            assert action.attr not in gp.nodes[action.source]
            gp.nodes[action.source][action.attr] = action.value

        elif action.action is GraphActionType.SetEdgeAttr:
            assert self.allow_edge_attr
            assert g.has_edge(action.source, action.target)
            assert action.attr not in gp.edges[(action.source, action.target)]
            gp.edges[(action.source, action.target)][action.attr] = action.value
        else:
            # TODO: backward actions if we want to support MCMC-GFN style algorithms
            raise ValueError(f"Unknown action type {action.action}", action.action)

        return gp

    def count_backward_transitions(self, g: Graph) -> int:
        """Counts the number of parents of g without checking for isomorphisms."""
        c = 0
        deg = [g.degree[i] for i in range(len(g.nodes))]
        for a, b in g.edges:
            if deg[a] > 1 and deg[b] > 1 and len(g.edges[(a, b)]) == 0:
                # Can only remove edges connected to non-leaves and without
                # attributes (the agent has to remove the attrs, then remove
                # the edge). Removal cannot disconnect the graph.
                new_g = graph_without_edge(g, (a, b))
                if nx.algorithms.is_connected(new_g):
                    c += 1
            c += len(g.edges[(a, b)])  # One action per edge attr
        for i in g.nodes:
            if (
                deg[i] == 1
                and len(g.nodes[i]) == 1
                and len(g.edges[list(g.edges(i))[0]]) == 0
            ):
                c += 1
            c += len(g.nodes[i]) - 1  # One action per node attr, except 'v'
            if len(g.nodes) == 1 and len(g.nodes[i]) == 1:
                # special case if last node in graph
                c += 1
        return c


def generate_forward_trajectory(
    g: Graph, max_nodes: int = None
) -> List[Tuple[Graph, GraphAction]]:
    """Sample (uniformly) a trajectory that generates g.

    Args:
        g: the graph to be generated.
        max_nodes: the maximum number of nodes to generate.

    Returns:
        a list of (graph, action) pairs.
    """

    gn = Graph()
    # Choose an arbitrary starting point, add to the stack
    stack: List[Tuple[int, ...]] = [(np.random.randint(0, len(g.nodes)),)]
    traj = []
    # This map keeps track of node labels in gn, since we have to start from 0
    relabeling_map: Dict[int, int] = {}
    while len(stack):
        # We pop from the stack until all nodes and edges have been
        # generated and their attributes have been set. Uninserted
        # nodes/edges will be added to the stack as the graph is
        # expanded from the starting point. Nodes/edges that have
        # attributes will be reinserted into the stack until those
        # attributes are "set".
        i = stack.pop(np.random.randint(len(stack)))

        gt = gn.copy()  # This is a shallow copy

        if len(i) > 1:  # i is an edge
            e = relabeling_map.get(i[0], None), relabeling_map.get(i[1], None)
            if e in gn.edges:
                # i exists in the new graph, that means some of its attributes need to be added
                attrs = [j for j in g.edges[i] if j not in gn.edges[e]]
                if len(attrs) == 0:
                    continue  # If nodes are in cycles edges leading to them get stack multiple times, disregard
                attr = attrs[np.random.randint(len(attrs))]
                gn.edges[e][attr] = g.edges[i][attr]
                act = GraphAction(
                    GraphActionType.SetEdgeAttr,
                    source=e[0],
                    target=e[1],
                    attr=attr,
                    value=g.edges[i][attr],
                )
            else:
                # i doesn't exist, add the edge
                if e[1] not in gn.nodes:
                    # The endpoint of the edge is not in the graph, this is a AddNode action
                    assert e[1] is None  # normally we shouldn't have relabeled i[1] yet
                    relabeling_map[i[1]] = len(relabeling_map)
                    e = e[0], relabeling_map[i[1]]
                    gn.add_node(e[1], v=g.nodes[i[1]]["v"])
                    gn.add_edge(*e)
                    for j in g[i[1]]:  # stack unadded edges/neighbours
                        jp = relabeling_map.get(j, None)
                        if jp not in gn or (e[1], jp) not in gn.edges:
                            stack.append((i[1], j))
                    act = GraphAction(
                        GraphActionType.AddNode, source=e[0], value=g.nodes[i[1]]["v"]
                    )
                    if len(gn.nodes[e[1]]) < len(g.nodes[i[1]]):
                        stack.append(
                            (i[1],)
                        )  # we still have attributes to add to node i[1]
                else:
                    # The endpoint is in the graph, this is an AddEdge action
                    assert e[0] in gn.nodes
                    gn.add_edge(*e)
                    act = GraphAction(GraphActionType.AddEdge, source=e[0], target=e[1])

            if len(gn.edges[e]) < len(g.edges[i]):
                stack.append(i)  # we still have attributes to add to edge i
        else:  # i is a node, (u,)
            u = i[0]
            n = relabeling_map.get(u, None)
            if n not in gn.nodes:
                # u doesn't exist yet, this should only happen for the first node
                assert len(gn.nodes) == 0
                act = GraphAction(
                    GraphActionType.AddNode, source=0, value=g.nodes[u]["v"]
                )
                n = relabeling_map[u] = len(relabeling_map)
                gn.add_node(0, v=g.nodes[u]["v"])
                for j in g[u]:  # For every neighbour of node u
                    if relabeling_map.get(j, None) not in gn:
                        stack.append((u, j))  # push the (u,j) edge onto the stack
            else:
                # u exists, meaning we have attributes left to add
                attrs = [j for j in g.nodes[u] if j not in gn.nodes[n]]
                attr = attrs[np.random.randint(len(attrs))]
                gn.nodes[n][attr] = g.nodes[u][attr]
                act = GraphAction(
                    GraphActionType.SetNodeAttr,
                    source=n,
                    attr=attr,
                    value=g.nodes[u][attr],
                )
            if len(gn.nodes[n]) < len(g.nodes[u]):
                stack.append((u,))  # we still have attributes to add to node u
        traj.append((gt, act))
    traj.append((gn, GraphAction(GraphActionType.Stop)))
    return traj


class GraphActionCategorical:
    """A multi-type categorical action class compatible with generating structured actions on a graph."""

    def __init__(
        self,
        graphs: gd.Batch,
        logits: List[torch.Tensor],
        keys: List[Union[str, None]],
        types: List[GraphActionType],
        deduplicate_edge_index=True,
    ) -> None:
        """Initialize a GraphActionCategorical.

        What is meant by multi-type here is that there are multiple types of
        mutually exclusive actions, e.g. AddNode and AddEdge are
        mutually exclusive, but since their logits will be produced by
        different variable-sized tensors (corresponding to different
        elements of the graph, e.g. nodes or edges) it is inconvient
        to stack them all into one single Categorical. This class
        provides this convenient interaction between torch_geometric
        Batch objects and lists of logit tensors.

        Code adapted from: https://github.com/recursionpharma/gflownet/tree/trunk/src/gflownet/envs

        Args:
            graphs: a Batch of graphs to which the logits correspond.
            logits: a list of tensors of shape `(n, m)` representing logits
                over a variable number of graph elements (e.g. nodes) for
                which there are `m` possible actions. `n` should thus be
                equal to the sum of the number of such elements for each
                graph in the Batch object. The length of the `logits` list
                should thus be equal to the number of element types (in
                other words there should be one tensor per type).
            keys: the keys corresponding to the Graph elements for each
                tensor in the logits list. Used to extract the `_batch`
                and slice attributes. For example, if the first logit
                tensor is a per-node action logit, and the second is a
                per-edge, `keys` could be `['x', 'edge_index']`. If
                keys[i] is None, the corresponding logits are assumed to
                be graph-level (i.e. if there are `k` graphs in the Batch
                object, this logit tensor would have shape `(k, m)`).
            types: the action type each logit corresponds to.
            deduplicate_edge_index: if true, this means that the 'edge_index' keys have been reduced
                by e_i[::2] (presumably because the graphs are undirected).
        """

        self.num_graphs = graphs.num_graphs
        # The logits
        self.logits = logits
        self.types = types
        self.keys = keys
        self.dev = dev = graphs.x.device

        # This is the minibatch index of each entry in the logits
        # i.e., if graph i in the Batch has N[i] nodes,
        #    g.batch == [0,0,0, ...,  1,1,1,1,1, ... ]
        #                 N[0] times    N[1] times
        # This generalizes to edges and non-edges.
        # Append '_batch' to keys except for 'x', since TG has a special case (done by default for 'x')
        self.batch = [
            getattr(graphs, f"{k}_batch" if k != "x" else "batch") if k is not None
            # None signals a global logit rather than a per-instance logit
            else torch.arange(graphs.num_graphs, device=dev)
            for k in keys
        ]
        # This is the cumulative sum (prefixed by 0) of N[i]s
        self.slice = [
            graphs._slice_dict[k]
            if k is not None
            else torch.arange(graphs.num_graphs, device=dev)
            for k in keys
        ]
        self.logprobs: Union[List[Any], None] = None

        if deduplicate_edge_index and "edge_index" in keys:
            idx = keys.index("edge_index")
            self.batch[idx] = self.batch[idx][::2]
            self.slice[idx] = self.slice[idx].div(2, rounding_mode="floor")

    def detach(self):
        """Detach the logits from the graph batch."""
        new = copy.copy(self)
        new.logits = [i.detach() for i in new.logits]
        if new.logprobs is not None:
            new.logprobs = [i.detach() for i in new.logprobs]
        return new

    def to(self, device):
        """Move everything to the specified device.

        Args:
            device: the device to move to.
        """
        self.dev = device
        self.logits = [i.to(device) for i in self.logits]
        self.batch = [i.to(device) for i in self.batch]
        self.slice = [i.to(device) for i in self.slice]
        if self.logprobs is not None:
            self.logprobs = [i.to(device) for i in self.logprobs]
        return self

    def logsoftmax(self) -> List[torch.Tensor]:
        """Compute log-probabilities given logits.

        Returns:
            a list of log-probabilities.
        """

        if self.logprobs is not None:
            return self.logprobs
        # Use the `subtract by max` trick to avoid precision errors:
        # compute max
        maxl = (
            torch.cat(
                [
                    scatter(i, b, dim=0, dim_size=self.num_graphs, reduce="max")
                    for i, b in zip(self.logits, self.batch)
                ],
                dim=1,
            )
            .max(1)
            .values.detach()
        )

        # substract by max then take exp
        # x[b, None] indexes by the batch to map back to each node/edge and adds a broadcast dim
        exp_logits = [
            (i - maxl[b, None]).exp() + 1e-40 for i, b in zip(self.logits, self.batch)
        ]

        # sum corrected exponentiated logits, to get log(z - max) = log(sum(exp(logits)) - max)
        log_z = sum(  # type: ignore
            [
                scatter(i, b, dim=0, dim_size=self.num_graphs, reduce="sum").sum(1)
                for i, b in zip(exp_logits, self.batch)
            ]
        ).log()

        # log probabilities is log(exp(logit) / Z)
        self.logprobs = [
            i.log() - log_z[b, None] for i, b in zip(exp_logits, self.batch)
        ]
        return self.logprobs

    def sample(self) -> List[Tuple[int, int, int]]:
        """Use the Gumbel trick to sample categoricals.

        if X ~ argmax(logits - log(-log(uniform(logits.shape))))
        then  p(X = i) = exp(logits[i]) / Z
        Here we have to do the argmax first over the variable number
        of rows of each element type for each graph in the
        minibatch, then over the different types (since they are
        mutually exclusive).

        Returns:
            A list of tuples specifying the action (type, row, column).
        """

        # Uniform noise
        u = [torch.rand(i.shape, device=self.dev) for i in self.logits]
        # Gumbel noise
        gumbel = [logit - (-noise.log()).log() for logit, noise in zip(self.logits, u)]
        # scatter_max and .max create a (values, indices) pair
        # These logits are 2d (num_obj_of_type, num_actions_of_type),
        # first reduce-max over the batch, which preserves the
        # columns, so we get (minibatch_size, num_actions_of_type).
        # First we prefill `out` with very negative values in case
        # there are no corresponding logits (this can happen if e.g. a
        # graph has no edges), we don't want to accidentally take the
        # max of that type.
        mnb_max = [
            torch.zeros(self.num_graphs, i.shape[1], device=self.dev) - 1e6
            for i in self.logits
        ]
        mnb_max = [
            scatter_max(i, b, dim=0, out=out)
            for i, b, out in zip(gumbel, self.batch, mnb_max)
        ]
        # Then over cols, this gets us which col holds the max value,
        # so we get (minibatch_size,)
        col_max = [values.max(1) for values, idx in mnb_max]
        # Now we look up which row in those argmax cols was the max:
        row_pos = [
            idx_mnb[torch.arange(len(idx_col)), idx_col]
            for (_, idx_mnb), (_, idx_col) in zip(mnb_max, col_max)
        ]
        # The maxes themselves
        maxs = [values for values, idx in col_max]
        # Now we need to check which type of logit has the actual max
        type_max_val, type_max_idx = torch.stack(maxs).max(0)
        if torch.isfinite(type_max_val).logical_not_().any():
            raise ValueError(
                "Non finite max value in sample", (type_max_val, self.logits)
            )

        # Now we can return the indices of where the actions occured
        # in the form List[(type, row, column)]
        actions = []
        for i in range(type_max_idx.shape[0]):
            t = type_max_idx[i]
            # Subtract from the slice of that type and index, since the computed
            # row position is batch-wise rather graph-wise
            actions.append(
                (int(t), int(row_pos[t][i] - self.slice[t][i]), int(col_max[t][1][i]))
            )
        # It's now up to the Context class to create GraphBuildingAction instances
        # if it wants to convert these indices to env-compatible actions
        return actions

    def log_prob(self, actions: List[Tuple[int, int, int]]) -> torch.Tensor:
        """The log-probability of a list of action tuples.

        Args:
            actions: A list of action tuples (type, row, column).

        Returns:
            A tensor of shape (minibatch_size,) containing the log-probability of each action.
        """
        logprobs = self.logsoftmax()
        return torch.stack(
            [
                logprobs[t][row + self.slice[t][i], col]
                for i, (t, row, col) in enumerate(actions)
            ]
        )


class GraphBuildingEnvContext:
    """A context environment that defines what the graphs are, how they map to and from data."""

    device: str
    num_node_dim: int
    num_edge_dim: int
    num_cond_dim: int
    num_new_node_values: int
    num_node_attr_logits: int
    num_edge_attr_logits: int
    action_type_order: List[GraphActionType]

    def aidx_to_graph_action(
        self, g: gd.Data, action_idx: Tuple[int, int, int]
    ) -> GraphAction:
        """Translate an action index (e.g. from a GraphActionCategorical) to a GraphAction.

        Args:
            g: the graph to which the action is being applied.
            action_idx: the tensor indices for the corresponding action.

        Returns:
            a graph action that could be applied to the original graph coressponding to g.
        """
        raise NotImplementedError()

    def graph_action_to_aidx(
        self, g: gd.Data, action: GraphAction
    ) -> Tuple[int, int, int]:
        """Translate a GraphAction to an action index (e.g. from a GraphActionCategorical).

        Args:
            g: the graph to which the action is being applied.
            action: a graph action that could be applied to the original graph coressponding to g.

        Returns:
            the tensor indices for the corresponding action.
        """
        raise NotImplementedError()

    def graph_to_mol(self, g: Graph) -> Mol:
        """Translate a graph to a molecule.

        Args:
            g: the graph to translate.

        Returns:
            the molecule corresponding to the graph.
        """
        pass

    def sample_conditional_information(self):
        """Sample conditional information."""
        pass

    def graph_to_data(self, g: Graph) -> gd.Data:
        """Convert a networkx Graph to a torch geometric Data instance.

        Args:
            g: a graph instance.

        Returns:
            the corresponding torch_geometric graph data.
        """
        raise NotImplementedError()

    def collate(self, graphs: List[gd.Data]) -> gd.Batch:
        """Convert a list of torch geometric Data instances to a Batch
        instance.  This exists so that environment contexts can set
        custom batching attributes, e.g. by using `follow_batch`.

        Args:
            graphs: graph instances.

        Returns:
            the corresponding torch_geometric batch.
        """

        return gd.Batch.from_data_list(graphs)

    def is_sane(self, g: Graph) -> bool:
        """Verifies whether a graph is sane according to the context. This can
        catch, e.g. impossible molecules.

        Args:
            g: a graph.

        Returns:
            true if the environment considers g to be sane.
        """

        raise NotImplementedError()

    def mol_to_graph(self, mol: Mol) -> Graph:
        """Verifies whether a graph is sane according to the context. This can
        catch, e.g. impossible molecules.

        Args:
            mol: an RDKit molecule.

        Returns:
            the corresponding Graph representation of that molecule.
        """

        raise NotImplementedError()
