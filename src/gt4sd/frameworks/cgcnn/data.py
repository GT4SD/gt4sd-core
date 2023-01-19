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
"""Data module."""

from __future__ import division, print_function

import csv
import functools
import json
import logging
import os
import random
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import torch
from pymatgen.core.structure import Structure  # type: ignore
from torch import LongTensor, Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_train_val_test_loader(
    dataset: torch.utils.data.Dataset,
    collate_fn: Callable[[List[Any]], Any] = default_collate,
    batch_size: int = 64,
    train_ratio: float = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    return_test: bool = False,
    num_workers: int = 1,
    pin_memory: bool = False,
    **kwargs,
) -> Union[
    Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]],
    Tuple[DataLoader[Any], DataLoader[Any]],
]:
    """Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Args:
        dataset: torch.utils.data.Dataset
          The full dataset to be divided.
        collate_fn: torch.utils.data.DataLoader.
        batch_size: int.
        train_ratio: float.
        val_ratio: float.
        test_ratio: float.
        return_test: bool.
          Whether to return the test dataset loader. If False, the last test_size
          data will be hidden.
        num_workers: int.
        pin_memory: bool.

    Returns:
        train_loader: torch.utils.data.DataLoader
          DataLoader that random samples the training data.
        val_loader: torch.utils.data.DataLoader
          DataLoader that random samples the validation data.
        (test_loader): torch.utils.data.DataLoader
          DataLoader that random samples the test data, Returns if
            return_test=True.
    """
    total_size = len(dataset)  # type: ignore
    if kwargs["train_size"] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            logger.warning(
                f"train_ratio is None, using 1 - val_ratio - "
                f"test_ratio = {train_ratio} as training data."
            )

        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs["train_size"]:
        train_size = kwargs["train_size"]
    else:
        train_size = int(train_ratio * total_size)  # type: ignore
    if kwargs["test_size"]:
        test_size = kwargs["test_size"]
    else:
        test_size = int(test_ratio * total_size)
    if kwargs["val_size"]:
        valid_size = kwargs["val_size"]
    else:
        valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[-(valid_size + test_size) : -test_size])

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    if return_test:

        test_sampler = SubsetRandomSampler(indices[-test_size:])

        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(
    dataset_list: List[Any],
) -> Tuple[Tuple[Tensor, Tensor, Tensor, List[LongTensor]], Tensor, List[Any]]:
    """Collate a list of data and return a batch for predicting crystal properties.

    Args:
        dataset_list: list of tuples for each data point.
          (atom_fea, nbr_fea, nbr_fea_idx, target)

          atom_fea: torch.Tensor shape (n_i, atom_fea_len).
          nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len).
          nbr_fea_idx: torch.LongTensor shape (n_i, M).
          target: torch.Tensor shape (1, ).
          cif_id: str or int.

    Returns:
        N = sum(n_i); N0 = sum(i)
        batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
          Atom features from atom type.
        batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors.
        batch_nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom.
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx.
        target: torch.Tensor shape (N, 1)
          Target value for prediction.
        batch_cif_ids: list.
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id) in enumerate(
        dataset_list
    ):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (
        (
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx,
        ),
        torch.stack(batch_target, dim=0),
        batch_cif_ids,
    )


class GaussianDistance:
    """Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin: float, dmax: float, step: float, var: float = None):
        """
        Args:
            dmin: float
              Minimum interatomic distance.
            dmax: float
              Maximum interatomic distance.
            step: float
              Step size for the Gaussian filter.
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances: np.ndarray) -> np.ndarray:
        """Apply Gaussian disntance filter to a numpy distance array.

        Args:
            distance: np.array shape n-d array
              A distance matrix of any shape.

        Returns:
            expanded_distance: shape (n+1)-d array
              Expanded distance matrix with the last dimension of length
              len(self.filter).
        """
        return np.exp(
            -((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2
        )


class AtomInitializer:
    """Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {
            idx: atom_type for atom_type, idx in self._embedding.items()
        }

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, "_decodedict"):
            self._decodedict = {
                idx: atom_type for atom_type, idx in self._embedding.items()
            }
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    """

    def __init__(self, elem_embedding_file: str):
        """
        Args:
            elem_embedding_file: str
                The path to the .json file.
        """
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.
    """

    def __init__(
        self,
        root_dir: str,
        max_num_nbr: int = 12,
        radius: int = 8,
        dmin: int = 0,
        step: float = 0.2,
        random_seed: int = 123,
    ):
        """
        Args:
            root_dir: str
                The path to the root directory of the dataset.
            max_num_nbr: int
                The maximum number of neighbors while constructing the crystal graph.
            radius: float
                The cutoff radius for searching neighbors.
            dmin: float
                The minimum distance for constructing GaussianDistance.
            step: float
                The step size for constructing GaussianDistance.
            random_seed: int
                Random seed for shuffling the dataset.
        """
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), "root_dir does not exist!"
        id_prop_file = os.path.join(self.root_dir, "id_prop.csv")
        assert os.path.exists(id_prop_file), "id_prop.csv does not exist!"
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, "atom_init.json")
        assert os.path.exists(atom_init_file), "atom_init.json does not exist!"
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:  # type: ignore
        """
        Args:
           idx: index.
        Returns:
            atom_fea: torch.Tensor shape (n_i, atom_fea_len).
            nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len).
            nbr_fea_idx: torch.LongTensor shape (n_i, M).
            target: torch.Tensor shape (1, ).
            cif_id: str or int.
        """
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id + ".cif"))
        atom_fea = np.vstack(
            [
                self.ari.get_atom_fea(crystal[i].specie.number)
                for i in range(len(crystal))
            ]
        )
        atom_fea = torch.Tensor(atom_fea)  # type: ignore
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                logger.warning(
                    "{} not find enough neighbors to build graph. "
                    "If it happens frequently, consider increase "
                    "radius.".format(cif_id)
                )
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr))
                    + [self.radius + 1.0] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[: self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[: self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)  # type: ignore
        nbr_fea = self.gdf.expand(nbr_fea)  # type: ignore
        atom_fea = torch.Tensor(atom_fea)  # type: ignore
        nbr_fea = torch.Tensor(nbr_fea)  # type: ignore
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)  # type: ignore
        target = torch.Tensor([float(target)])  # type: ignore
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
