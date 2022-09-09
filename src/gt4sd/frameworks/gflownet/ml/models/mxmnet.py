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
import inspect
import math
from collections import OrderedDict
from math import pi as PI
from operator import itemgetter
from typing import List, Optional, Set, Tuple

import numpy as np
import sympy as sym
import torch
import torch.nn as nn
from rdkit.Chem import AllChem
from scipy import special as sp
from scipy.optimize import brentq
from torch.nn import Linear, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, radius
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter
from torch_sparse import SparseTensor

HAR2EV = 27.2113825435
KCALMOL2EV = 0.04336414


class MXMNetConfig(object):
    """MXMNet configuration."""

    def __init__(self, dim, n_layer, cutoff):
        """Initialize MXMNet configuration.

        Args:
            dim: dimension of the input.
            n_layer: number of layers.
            cutoff: cutoff radius.
        """
        self.dim = dim
        self.n_layer = n_layer
        self.cutoff = cutoff


class MXMNet(nn.Module):
    """MXMNet - Multiplex Molecular Graph Neural Network"""

    def __init__(
        self, config: MXMNetConfig, num_spherical=7, num_radial=6, envelope_exponent=5
    ) -> None:
        """Construct an MXMNet.

        Code adapted from: https://github.com/recursionpharma/gflownet/tree/trunk/src/gflownet/models and https://github.com/zetayue/MXMNet.

        Args:
            config: model configuration
            num_spherical: number of spherical harmonics to use.
            num_radial: number of radial harmonics to use.
            envelope_exponent: exponent of the envelope function.
        """
        super(MXMNet, self).__init__()

        self.name = "mxmnet"

        self.dim = config.dim
        self.n_layer = config.n_layer
        self.cutoff = config.cutoff

        self.embeddings = nn.Parameter(torch.ones((5, self.dim)))

        self.rbf_l = BesselBasisLayer(16, 5, envelope_exponent)
        self.rbf_g = BesselBasisLayer(16, self.cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, 5, envelope_exponent)

        self.rbf_g_mlp = MLP([16, self.dim])
        self.rbf_l_mlp = MLP([16, self.dim])

        self.sbf_1_mlp = MLP([num_spherical * num_radial, self.dim])
        self.sbf_2_mlp = MLP([num_spherical * num_radial, self.dim])

        self.global_layers = torch.nn.ModuleList()
        for layer in range(config.n_layer):
            self.global_layers.append(Global_MP(config))

        self.local_layers = torch.nn.ModuleList()
        for layer in range(config.n_layer):
            self.local_layers.append(Local_MP(config))

        self.init()

    def init(self):
        stdv = math.sqrt(3)
        self.embeddings.data.uniform_(-stdv, stdv)

    def indices(self, edge_index, num_nodes):
        """Compute indices.

        Args:
            edge_index: edge index of the graph.
            num_nodes: number of nodes in the graph.

        Returns:
            tuple of indeces.
        """
        row, col = edge_index

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(
            row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
        )

        # Compute the node indices for two-hop angles
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k
        idx_i_1, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji_1 = adj_t_row.storage.row()[mask]

        # Compute the node indices for one-hop angles
        adj_t_col = adj_t[col]

        num_pairs = adj_t_col.set_value(None).sum(dim=1).to(torch.long)
        idx_i_2 = row.repeat_interleave(num_pairs)
        idx_j1 = col.repeat_interleave(num_pairs)
        idx_j2 = adj_t_col.storage.col()

        idx_ji_2 = adj_t_col.storage.row()
        idx_jj = adj_t_col.storage.value()

        return (
            idx_i_1,
            idx_j,
            idx_k,
            idx_kj,
            idx_ji_1,
            idx_i_2,
            idx_j1,
            idx_j2,
            idx_jj,
            idx_ji_2,
        )

    def forward(self, data):
        """Forward pass.

        Args:
            data: batch of data.

        Returns:
            gloabl pooled features.
        """
        x = data.x
        edge_index = data.edge_index
        pos = data.pos
        batch = data.batch
        # Initialize node embeddings
        h = torch.index_select(self.embeddings, 0, x.long())

        # Get the edges and pairwise distances in the local layer
        edge_index_l, _ = remove_self_loops(edge_index)
        j_l, i_l = edge_index_l
        dist_l = (pos[i_l] - pos[j_l]).pow(2).sum(dim=-1).sqrt()

        # Get the edges pairwise distances in the global layer
        row, col = radius(pos, pos, self.cutoff, batch, batch, max_num_neighbors=500)
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, _ = remove_self_loops(edge_index_g)
        j_g, i_g = edge_index_g
        dist_g = (pos[i_g] - pos[j_g]).pow(2).sum(dim=-1).sqrt()

        # Compute the node indices for defining the angles
        (
            idx_i_1,
            idx_j,
            idx_k,
            idx_kj,
            idx_ji,
            idx_i_2,
            idx_j1,
            idx_j2,
            idx_jj,
            idx_ji_2,
        ) = self.indices(edge_index_l, num_nodes=h.size(0))

        # Compute the two-hop angles
        pos_ji_1, pos_kj = pos[idx_j] - pos[idx_i_1], pos[idx_k] - pos[idx_j]
        a = (pos_ji_1 * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji_1, pos_kj).norm(dim=-1)
        angle_1 = torch.atan2(b, a)

        # Compute the one-hop angles
        pos_ji_2, pos_jj = pos[idx_j1] - pos[idx_i_2], pos[idx_j2] - pos[idx_j1]
        a = (pos_ji_2 * pos_jj).sum(dim=-1)
        b = torch.cross(pos_ji_2, pos_jj).norm(dim=-1)
        angle_2 = torch.atan2(b, a)

        # Get the RBF and SBF embeddings
        rbf_g = self.rbf_g(dist_g)
        rbf_l = self.rbf_l(dist_l)
        sbf_1 = self.sbf(dist_l, angle_1, idx_kj)
        sbf_2 = self.sbf(dist_l, angle_2, idx_jj)

        rbf_g = self.rbf_g_mlp(rbf_g)
        rbf_l = self.rbf_l_mlp(rbf_l)
        sbf_1 = self.sbf_1_mlp(sbf_1)
        sbf_2 = self.sbf_2_mlp(sbf_2)

        # Perform the message passing schemes
        node_sum = 0

        for layer in range(self.n_layer):
            h = self.global_layers[layer](h, rbf_g, edge_index_g)
            h, t = self.local_layers[layer](
                h, rbf_l, sbf_1, sbf_2, idx_kj, idx_ji, idx_jj, idx_ji_2, edge_index_l
            )
            node_sum += t

        # Readout
        output = global_add_pool(node_sum, batch)
        return output.view(-1)


class EMA:
    """EMA - Exponential Moving Average."""

    def __init__(self, model, decay):
        """Initialize ema.

        Args:
            model: model to ema.
            decay: decay rate.
        """
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates=99999):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


def MLP(channels) -> nn.Sequential:
    """multi-layer perceptron.

    Args:
        channels: list of number of channels.

    Returns:
        MLP model.
    """
    return Sequential(
        *[
            Sequential(Linear(channels[i - 1], channels[i]), SiLU())
            for i in range(1, len(channels))
        ]
    )


class Res(nn.Module):
    """Residual Block."""

    def __init__(self, dim):
        """Initialize residual block.

        Args:
            dim: dimension of the layer.
        """
        super(Res, self).__init__()

        self.mlp = MLP([dim, dim, dim])

    def forward(self, m):
        m1 = self.mlp(m)
        m_out = m1 + m
        return m_out


def compute_idx(pos, edge_index):
    """Compute the indices for the edges and angles.

    Args:
        pos: node positions.
        edge_index: edge indices.
    """

    pos_i = pos[edge_index[0]]
    pos_j = pos[edge_index[1]]

    d_ij = torch.norm(abs(pos_j - pos_i), dim=-1, keepdim=False).unsqueeze(-1) + 1e-5
    v_ji = (pos_i - pos_j) / d_ij

    unique, counts = torch.unique(
        edge_index[0], sorted=True, return_counts=True
    )  # Get central values
    full_index = (
        torch.arange(0, edge_index[0].size()[0]).cuda().int()
    )  # init full index

    # Compute 1
    repeat = torch.repeat_interleave(counts, counts)
    counts_repeat1 = torch.repeat_interleave(full_index, repeat)  # 0,...,0,1,...,1,...

    # Compute 2
    split = torch.split(full_index, counts.tolist())  # split full index
    index2 = list(edge_index[0].data.cpu().numpy())  # get repeat index
    counts_repeat2 = torch.cat(itemgetter(*index2)(split), dim=0)  # 0,1,2,...,0,1,2,..

    # Compute angle embeddings
    v1 = v_ji[counts_repeat1.long()]
    v2 = v_ji[counts_repeat2.long()]

    angle = (v1 * v2).sum(-1).unsqueeze(-1)
    angle = torch.clamp(angle, min=-1.0, max=1.0) + 1e-6 + 1.0

    return counts_repeat1.long(), counts_repeat2.long(), angle


def Jn(r, n):
    return np.sqrt(np.pi / (2 * r)) * sp.jv(n + 0.5, r)


def Jn_zeros(n, k):
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n):
    x = sym.symbols("x")
    f = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        f += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return f


def bessel_basis(n, k):
    zeros = Jn_zeros(n, k)
    normalizer = []

    for order in range(n):
        _normalizer_tmp = []
        for i in range(k):
            _normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]

        normalizer_tmp = 1 / np.array(_normalizer_tmp) ** 0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols("x")
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(
                    normalizer[order][i] * f[order].subs(x, zeros[order, i] * x)
                )
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(k, m):
    return (
        (2 * k + 1)
        * math.factorial(k - abs(m))
        / (4 * np.pi * math.factorial(k + abs(m)))
    ) ** 0.5


def associated_legendre_polynomials(k, zero_m_only=True):
    z = sym.symbols("z")
    P_l_m = [[0] * (j + 1) for j in range(k)]

    P_l_m[0][0] = 1
    if k > 0:
        P_l_m[1][0] = z

        for j in range(2, k):
            P_l_m[j][0] = sym.simplify(
                ((2 * j - 1) * z * P_l_m[j - 1][0] - (j - 1) * P_l_m[j - 2][0]) / j
            )
        if not zero_m_only:
            for i in range(1, k):
                P_l_m[i][i] = sym.simplify((1 - 2 * i) * P_l_m[i - 1][i - 1])
                if i + 1 < k:
                    P_l_m[i + 1][i] = sym.simplify((2 * i + 1) * z * P_l_m[i][i])
                for j in range(i + 2, k):
                    P_l_m[j][i] = sym.simplify(
                        (
                            (2 * j - 1) * z * P_l_m[j - 1][i]
                            - (i + j - 1) * P_l_m[j - 2][i]
                        )
                        / (j - i)
                    )

    return P_l_m


def real_sph_harm(k, zero_m_only=True, spherical_coordinates=True):
    if not zero_m_only:
        S_m = [0]
        C_m = [1]
        for i in range(1, k):
            x = sym.symbols("x")
            y = sym.symbols("y")
            S_m += [x * S_m[i - 1] + y * C_m[i - 1]]
            C_m += [x * C_m[i - 1] - y * S_m[i - 1]]

    P_l_m = associated_legendre_polynomials(k, zero_m_only)
    if spherical_coordinates:
        theta = sym.symbols("theta")
        z = sym.symbols("z")
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if type(P_l_m[i][j]) != int:
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
        if not zero_m_only:
            phi = sym.symbols("phi")

            for i in range(len(S_m)):
                S_m[i] = (
                    S_m[i]  # type: ignore
                    .subs(x, sym.sin(theta) * sym.cos(phi))
                    .subs(y, sym.sin(theta) * sym.sin(phi))
                )
            for i in range(len(C_m)):
                C_m[i] = (
                    C_m[i]  # type: ignore
                    .subs(x, sym.sin(theta) * sym.cos(phi))
                    .subs(y, sym.sin(theta) * sym.sin(phi))
                )

    Y_func_l_m = [["0"] * (2 * j + 1) for j in range(k)]
    for i in range(k):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, k):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j]
                )
        for i in range(1, k):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j]
                )

    return Y_func_l_m


class BesselBasisLayer(torch.nn.Module):
    """Bessel Basis Layer."""

    def __init__(self, num_radial, cutoff, envelope_exponent=6) -> None:
        """Initialize Bessel basis layer.

        Args:
            num_radial: number of radial basis functions.
            cutoff: cutoff radius.
            envelope_exponent: envelope exponent.
        """

        super(BesselBasisLayer, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        torch.arange(1, self.freq.numel() + 1, out=self.freq.data).mul_(PI)

    def forward(self, dist):
        """Forward pass.

        Args:
            dist: distance matrix.

        Returns:
            Bessel basis.
        """
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class SiLU(nn.Module):
    """SiLU Activation Function."""

    def __init__(self) -> None:
        """Initialize the SiLU activation function."""
        super().__init__()

    def forward(self, input):
        return silu(input)


def silu(input):
    return input * torch.sigmoid(input)


class Envelope(torch.nn.Module):
    """Envelope."""

    def __init__(self, exponent) -> None:
        """Initialize envelope.

        Args:
            exponent: exponent of the envelope.
        """
        super(Envelope, self).__init__()
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        """Forward pass.

        Args:
            x: input.

        Returns:
            Envelope of x.
        """

        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p)
        x_pow_p1 = x_pow_p0 * x
        env_val = 1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p1 * x

        zero = torch.zeros_like(x)
        return torch.where(x < 1, env_val, zero)


class SphericalBasisLayer(torch.nn.Module):
    """Spherical Basis Layer."""

    def __init__(
        self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5
    ) -> None:
        """Initialize spherical basis layer.

        Args:
            num_spherical: number of spherical harmonics.
            num_radial: number of radial functions.
            cutoff: cutoff radius.
            envelope_exponent: envelope exponent.
        """
        super(SphericalBasisLayer, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols("x theta")
        modules = {"sin": torch.sin, "cos": torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        """Forward pass.

        Args:
            dist: distance matrix.
            angle: angle matrix.
            idx_kj: index matrix.

        Returns:
            Spherical basis.
        """
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


msg_special_args = set(
    ["edge_index", "edge_index_i", "edge_index_j", "size", "size_i", "size_j"]
)

aggr_special_args = set(["index", "dim_size"])

update_special_args: Set = set([])


class MessagePassing(torch.nn.Module):
    r"""Message Passing Layer.

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),
    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    """

    def __init__(self, aggr="add", flow="target_to_source", node_dim=0) -> None:
        """Initialize message passing layer.

        Args:
            aggr: the aggregation scheme to use (add, mean, max).
            flow: the flow direction of message passing (source_to_target, target_to_source).
            node_dim: the axis along which to propagate.
        """
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ["add", "mean", "max"]

        self.flow = flow
        assert self.flow in ["source_to_target", "target_to_source"]

        self.node_dim = node_dim
        assert self.node_dim >= 0

        self.__msg_params__tmp = inspect.signature(self.message).parameters
        self.__msg_params__ = OrderedDict(self.__msg_params__tmp)

        self.__aggr_params__tmp = inspect.signature(self.aggregate).parameters
        self.__aggr_params__ = OrderedDict(self.__aggr_params__tmp)
        self.__aggr_params__.popitem(last=False)

        self.__update_params__tmp = inspect.signature(self.update).parameters
        self.__update_params__ = OrderedDict(self.__update_params__tmp)
        self.__update_params__.popitem(last=False)

        msg_args = set(self.__msg_params__.keys()) - msg_special_args
        aggr_args = set(self.__aggr_params__.keys()) - aggr_special_args
        update_args = set(self.__update_params__.keys()) - update_special_args

        self.__args__ = set().union(msg_args, aggr_args, update_args)

    def __set_size__(self, size, index, tensor):
        if not torch.is_tensor(tensor):
            pass
        elif size[index] is None:
            size[index] = tensor.size(self.node_dim)
        elif size[index] != tensor.size(self.node_dim):
            raise ValueError(
                (
                    f"Encountered node tensor with size {tensor.size(self.node_dim)} in dimension {self.node_dim}, but expected size {size[index]}."
                )
            )

    def __collect__(self, edge_index, size, kwargs):
        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        ij = {"_i": i, "_j": j}

        out = {}
        for arg in self.__args__:
            if arg[-2:] not in ij.keys():
                out[arg] = kwargs.get(arg, inspect.Parameter.empty)
            else:
                idx = ij[arg[-2:]]
                data = kwargs.get(arg[:-2], inspect.Parameter.empty)

                if data is inspect.Parameter.empty:
                    out[arg] = data
                    continue

                if isinstance(data, tuple) or isinstance(data, list):
                    assert len(data) == 2
                    self.__set_size__(size, 1 - idx, data[1 - idx])
                    data = data[idx]

                if not torch.is_tensor(data):
                    out[arg] = data
                    continue

                self.__set_size__(size, idx, data)
                out[arg] = data.index_select(self.node_dim, edge_index[idx])

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        # Add special message arguments.
        out["edge_index"] = edge_index
        out["edge_index_i"] = edge_index[i]
        out["edge_index_j"] = edge_index[j]
        out["size"] = size
        out["size_i"] = size[i]
        out["size_j"] = size[j]

        # Add special aggregate arguments.
        out["index"] = out["edge_index_i"]
        out["dim_size"] = out["size_i"]

        return out

    def __distribute__(self, params, kwargs):
        out = {}
        for key, param in params.items():
            data = kwargs[key]
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f"Required parameter {key} is empty.")
                data = param.default
            out[key] = data
        return out

    def propagate(
        self, edge_index: torch.Tensor, size: Optional[List[Tuple]] = None, **kwargs
    ):
        """The initial call to start propagating messages.

        Args:
            edge_index: the indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size: the size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size will be
                automatically inferred and assumed to be quadratic.
            **kwargs: any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """

        size = [None, None] if size is None else size  # type: ignore
        size = [size, size] if isinstance(size, int) else size  # type: ignore
        size = size.tolist() if torch.is_tensor(size) else size  # type: ignore
        size = list(size) if isinstance(size, tuple) else size
        assert isinstance(size, list)
        assert len(size) == 2

        kwargs = self.__collect__(edge_index, size, kwargs)

        msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)

        m = self.message(**msg_kwargs)
        aggr_kwargs = self.__distribute__(self.__aggr_params__, kwargs)
        m = self.aggregate(m, **aggr_kwargs)

        update_kwargs = self.__distribute__(self.__update_params__, kwargs)
        m = self.update(m, **update_kwargs)

        return m

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages to node :math:`i` in analogy to
        :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :math:`(j,i) \in \mathcal{E}` if :obj:`flow="source_to_target"` and
        :math:`(i,j) \in \mathcal{E}` if :obj:`flow="target_to_source"`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """

        return x_j

    def aggregate(self, inputs, index, dim_size):  # pragma: no cover
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.
        By default, delegates call to scatter functions that support
        "add", "mean" and "max" operations specified in :meth:`__init__` by
        the :obj:`aggr` argument.
        """

        return scatter(
            inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr
        )

    def update(self, inputs):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """
        return inputs


params = AllChem.ETKDGv3()
params.useSmallRingTorsions = True


def rdkit_conformation(mol, n=5, addHs=False):
    """An function that finds the lowest energy conformation of a molecule.

    Args:
        mol: RDKit molecule object.
        n: Number of conformations to find.
        addHs: Whether to add hydrogens to the molecule.

    Returns:
        RDKit molecule object with lowest energy conformation. If none, no conformation is found.
    """
    if addHs:
        mol = AllChem.AddHs(mol)
    confs = AllChem.EmbedMultipleConfs(mol, numConfs=n, params=params)
    minc, aminc = 1000, 0
    for i in range(len(confs)):
        mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=i)
        if ff is None:
            continue
        e = ff.CalcEnergy()
        if e < minc:
            minc = e
            aminc = i
    if len(confs):
        pos = []
        conf = mol.GetConformer(aminc)
        for i in range(mol.GetNumAtoms()):
            pos.append(list(conf.GetAtomPosition(i)))
        return torch.tensor(pos)
    return None


def mol2graph(mol):
    """Converts a RDKit molecule to a graph.

    Args:
        mol: RDKit molecule.

    Returns:
        A graph with node features and edge features.
    """
    mol = AllChem.AddHs(mol)
    N = mol.GetNumAtoms()
    try:
        pos = rdkit_conformation(mol)
        assert pos is not None, "no conformations found"
    except Exception:
        return None
    types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    type_idx = []
    for atom in mol.GetAtoms():
        type_idx.append(types[atom.GetSymbol()])

    row, col = [], []  # ,edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]

    x = torch.tensor(type_idx).to(torch.float)
    data = Data(x=x, pos=pos, edge_index=edge_index)
    return data


class Global_MP(MessagePassing):
    """Global message passing."""

    def __init__(self, config) -> None:
        """Initializes the global message passing.

        Args:
            config: configuration.
        """
        super(Global_MP, self).__init__()
        self.dim = config.dim

        self.h_mlp = MLP([self.dim, self.dim])

        self.res1 = Res(self.dim)
        self.res2 = Res(self.dim)
        self.res3 = Res(self.dim)
        self.mlp = MLP([self.dim, self.dim])

        self.x_edge_mlp = MLP([self.dim * 3, self.dim])
        self.linear = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, h, edge_attr, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))

        res_h = h

        # Integrate the Cross Layer Mapping inside the Global Message Passing
        h = self.h_mlp(h)

        # Message Passing operation
        h = self.propagate(edge_index, x=h, num_nodes=h.size(0), edge_attr=edge_attr)

        # Update function f_u
        h = self.res1(h)
        h = self.mlp(h) + res_h
        h = self.res2(h)
        h = self.res3(h)

        # Message Passing operation
        h = self.propagate(edge_index, x=h, num_nodes=h.size(0), edge_attr=edge_attr)

        return h

    def message(self, x_i, x_j, edge_attr, edge_index, num_nodes):
        num_edge = edge_attr.size()[0]

        x_edge = torch.cat((x_i[:num_edge], x_j[:num_edge], edge_attr), -1)
        x_edge = self.x_edge_mlp(x_edge)

        x_j = torch.cat((self.linear(edge_attr) * x_edge, x_j[num_edge:]), dim=0)

        return x_j

    def update(self, aggr_out):

        return aggr_out


class Local_MP(torch.nn.Module):
    """Local message passing."""

    def __init__(self, config) -> None:
        """Initialize local message passing.

        Args:
            config: configuration.
        """

        super(Local_MP, self).__init__()
        self.dim = config.dim

        self.h_mlp = MLP([self.dim, self.dim])

        self.mlp_kj = MLP([3 * self.dim, self.dim])
        self.mlp_ji_1 = MLP([3 * self.dim, self.dim])
        self.mlp_ji_2 = MLP([self.dim, self.dim])
        self.mlp_jj = MLP([self.dim, self.dim])

        self.mlp_sbf1 = MLP([self.dim, self.dim, self.dim])
        self.mlp_sbf2 = MLP([self.dim, self.dim, self.dim])
        self.lin_rbf1 = nn.Linear(self.dim, self.dim, bias=False)
        self.lin_rbf2 = nn.Linear(self.dim, self.dim, bias=False)

        self.res1 = Res(self.dim)
        self.res2 = Res(self.dim)
        self.res3 = Res(self.dim)

        self.lin_rbf_out = nn.Linear(self.dim, self.dim, bias=False)

        self.h_mlp = MLP([self.dim, self.dim])

        self.y_mlp = MLP([self.dim, self.dim, self.dim, self.dim])
        self.y_W = nn.Linear(self.dim, 1)

    def forward(
        self,
        h,
        rbf,
        sbf1,
        sbf2,
        idx_kj,
        idx_ji_1,
        idx_jj,
        idx_ji_2,
        edge_index,
        num_nodes=None,
    ):
        res_h = h

        # Integrate the Cross Layer Mapping inside the Local Message Passing
        h = self.h_mlp(h)

        # Message Passing 1
        j, i = edge_index
        m = torch.cat([h[i], h[j], rbf], dim=-1)

        m_kj = self.mlp_kj(m)
        m_kj = m_kj * self.lin_rbf1(rbf)
        m_kj = m_kj[idx_kj] * self.mlp_sbf1(sbf1)
        m_kj = scatter(m_kj, idx_ji_1, dim=0, dim_size=m.size(0), reduce="add")

        m_ji_1 = self.mlp_ji_1(m)

        m = m_ji_1 + m_kj

        # Message Passing 2       (index jj denotes j'i in the main paper)
        m_jj = self.mlp_jj(m)
        m_jj = m_jj * self.lin_rbf2(rbf)
        m_jj = m_jj[idx_jj] * self.mlp_sbf2(sbf2)
        m_jj = scatter(m_jj, idx_ji_2, dim=0, dim_size=m.size(0), reduce="add")

        m_ji_2 = self.mlp_ji_2(m)

        m = m_ji_2 + m_jj

        # Aggregation
        m = self.lin_rbf_out(rbf) * m
        h = scatter(m, i, dim=0, dim_size=h.size(0), reduce="add")

        # Update function f_u
        h = self.res1(h)
        h = self.h_mlp(h) + res_h
        h = self.res2(h)
        h = self.res3(h)

        # Output Module
        y = self.y_mlp(h)
        y = self.y_W(y)

        return h, y
