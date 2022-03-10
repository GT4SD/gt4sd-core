"""
Implementation details for TorchDrug generation algorithms.

Parts of the implementation inspired by: https://torchdrug.ai/docs/tutorials/generation.html.
"""
import logging
from pathlib import Path
from typing import List, Optional, Union

import torch

# Disable openmp usage since this raises on MacOS when libomp has the wrong version.
torch._C.has_openmp = False
from torch import optim

from torchdrug import core, models, tasks
from torchdrug.layers import distribution

from ....frameworks.torch import device_claim

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DummyDataset:
    """A helper class to imitate a torchdrug dataset."""

    def __init__(self, atom_types: List[int]):
        self.atom_types = atom_types
        self.transform = None


class Generator:
    """Implementation of a TorchDrug generator."""

    def __init__(
        self,
        resources_path: str,
        atom_types: List[int],
        hidden_dims: List[int],
        input_dim: int,
        num_relation: int,
        batch_norm: bool,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """A TorchDrug generation algorithm.

        Args:
            resources_path: path to the cache.
            atom_types: list of atom types.
            hidden_dims: list of hidden dimensions, one per layer.
            num_relation: number of relations for the graph.
            batch_norm: whether to use batch normalization.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """
        self.device = device_claim(device)
        self.resources_path = resources_path

        self.model = models.RGCN(
            input_dim=input_dim,
            num_relation=num_relation,
            hidden_dims=hidden_dims,
            batch_norm=batch_norm,
        )

        self.dataset = DummyDataset(atom_types)

    def load_model(self, resources_path: str):
        """Load a pretrained TorchDrug model."""
        self.solver.load(Path(resources_path).joinpath("weights.pkl"))

    def sample(self) -> List[str]:
        """Sample a molecule.

        Returns:
            a generated SMILES string wrapped into a list.
        """

        results = self.task.generate(num_sample=16, max_resample=32)
        return [list(results.to_smiles())[-1]]


class GCPNGenerator(Generator):
    """
    Interface for the GCPN model as implemented in TorchDrug.

    For details see:
    You, J. et al. (2018). Graph convolutional policy network for goal-directed
    molecular graph generation. Advances in neural information processing systems, 31.

    """

    input_dim = 18
    num_relation = 3
    batch_norm = False
    atom_types = [6, 7, 8, 9, 15, 16, 17, 35, 53]
    hidden_dims = [256, 256, 256, 256]

    def __init__(self, resources_path: str):
        """
        Args:
            resources_path: path to the cache.
        """

        super().__init__(
            input_dim=self.input_dim,
            num_relation=self.num_relation,
            batch_norm=self.batch_norm,
            atom_types=self.atom_types,
            hidden_dims=self.hidden_dims,
            resources_path=resources_path,
        )

        self.task = tasks.GCPNGeneration(
            self.model,
            self.atom_types,
            max_edge_unroll=12,
            max_node=38,
            criterion="nll",
        )
        optimizer = optim.Adam(self.task.parameters(), lr=1e-3)
        self.solver = core.Engine(self.task, self.dataset, None, None, optimizer)
        self.load_model(resources_path)


class GAFGenerator(Generator):
    """
    Interface for the GraphAF model as implemented in TorchDrug.

    For details see:
    Shi, Chence, et al. "GraphAF: a Flow-based Autoregressive Model for Molecular
    Graph Generation" International Conference on Learning Representations (ICLR), 2020.
    """

    input_dim = 9
    num_relations = 3
    batch_norm = True
    atom_types = [6, 7, 8, 9, 15, 16, 17, 35, 53]
    hidden_dims = [256, 256, 256]

    def __init__(self, resources_path: str):
        """
        Args:
            resources_path (str): path to the cache.
        """
        super().__init__(
            input_dim=self.input_dim,
            num_relation=self.num_relations,
            batch_norm=self.batch_norm,
            atom_types=self.atom_types,
            hidden_dims=self.hidden_dims,
            resources_path=resources_path,
        )

        node_prior = distribution.IndependentGaussian(
            torch.zeros(self.input_dim), torch.ones(self.input_dim)
        )
        edge_prior = distribution.IndependentGaussian(
            torch.zeros(self.num_relations + 1), torch.ones(self.num_relations + 1)
        )
        node_flow = models.GraphAF(self.model, node_prior, num_layer=12)
        edge_flow = models.GraphAF(self.model, edge_prior, use_edge=True, num_layer=12)

        self.task = tasks.AutoregressiveGeneration(
            node_flow, edge_flow, max_node=38, max_edge_unroll=12, criterion="nll"
        )
        optimizer = optim.Adam(self.task.parameters(), lr=1e-3)
        self.solver = core.Engine(self.task, self.dataset, None, None, optimizer)
        self.load_model(resources_path)
