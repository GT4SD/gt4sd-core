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
import ast
import os

# import tarfile
# import zipfile
from typing import Any, Callable, Dict, List, NewType, Tuple, Union

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor
from torch.utils.data import Dataset

from ..envs.graph_building_env import GraphActionCategorical
from ..util import wrap_model_mp

# This type represents an unprocessed list of reward signals/conditioning information
FlatRewards = NewType("FlatRewards", torch.tensor)  # type: ignore

# This type represents the outcome for a multi-objective task of
# converting FlatRewards to a scalar, e.g. (sum R_i omega_i) ** beta
RewardScalar = NewType("RewardScalar", torch.tensor)  # type: ignore


class GFlowNetTask:
    def __init__(
        self,
        configuration: Dict[str, Any],
        dataset: Dataset,
        reward_model: nn.Module = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ) -> None:

        """Class for a generic task.
        We consider the task as part of the dataset (environment).

        Args:
            reward_model: The model that is used to generate the conditional reward.
            dataset:
            temperature_distribution:
            temperature_parameters:
            wrap_model: a wrapper function that is applied to the model. # TODO: do we need it with lightning?
            device: cpu or cuda
        """
        hps = configuration

        self._wrap_model = wrap_model
        self.num_workers = hps["num_workers"]
        self.device = hps["device"]
        self.rng = hps["rng"]

        # fix this
        if reward_model:
            self.model = {"model_task": reward_model}
        else:
            self.model = self.load_task_models()

        self.dataset = dataset
        self.temperature_sample_dist = hps["temperature_sample_dist"]
        self.temperature_dist_params = ast.literal_eval(hps["temperature_dist_params"])

        self._min, self._max, self._percentile_95 = self.dataset.get_stats(percentile=0.05)  # type: ignore
        self._width = self._max - self._min
        self._rtrans = "unit+95p"

    def load_task_models(self) -> Dict[str, nn.Module]:
        pass

    def sample_conditional_information(self, n) -> Dict[str, Any]:
        pass

    def cond_info_to_reward(
        self, cond_info: Dict[str, Any], flat_reward: FlatRewards
    ) -> RewardScalar:
        """Combines a minibatch of reward signal vectors and conditional information into a scalar reward.

        Args:
            cond_info: a dictionary with various conditional informations (e.g. temperature).
            flat_reward: a 2d tensor where each row represents a series of flat rewards.

        Returns:
            reward: a 1d tensor, a scalar reward for each minibatch entry.
        """
        raise NotImplementedError()

    def compute_flat_rewards(self, x: List[Any]) -> Tuple[RewardScalar, Any]:
        """Compute the flat rewards of mols according the the tasks' proxies.

        Args:
            mols: a list of RDKit molecules.
        Returns:
            reward: a 1d tensor, a scalar reward for each molecule.
            is_valid: a 1d tensor, a boolean indicating whether the molecule is valid.
        """
        raise NotImplementedError()

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        raise NotImplementedError()

    def _wrap_model_mp(self, model):
        """Wraps a nn.Module instance so that it can be shared to `DataLoader` workers."""
        if self.num_workers > 0:
            placeholder = wrap_model_mp(
                model, self.num_workers, cast_types=(gd.Batch, GraphActionCategorical)
            )
            return placeholder
        return model


class GFlowNetDataset(Dataset):
    def __init__(
        self,
        h5_file: str = None,
        xyz_file: str = None,
        target: str = "gap",
        properties: List[str] = [],
    ) -> None:
        """Dataloader for generic dataset. Assuming the dataset is in a format compatible with h5 file.
        Describe dataset structure in the h5 file.

        Args:
            h5_file: data file in h5 format.
            xyz_file: data file in xyz format.
            train: split.
            target: target.
            properties: properties.
        """
        if h5_file is not None:
            import pickle

            # problem with python 3.8 and 3.7
            # pickle on 3.8 uses prot 5
            pickle.HIGHEST_PROTOCOL = 5
            self.df = pd.HDFStore(
                h5_file,
                "r",
            )["df"]
        elif xyz_file is not None:
            pass
            # self.load_tar(xyz_file)

        self.target = target
        self.properties = properties
        self.len = len(self.df)

    def set_indexes(self, ixs):
        self.idcs = ixs

    def get_len(self):
        return self.len

    def get_stats(self, percentile: float = 0.95) -> Tuple[float, float, Any]:
        """Get the stats of the dataset.

        Args:
            percentile: percentile.

        Returns:
            min, max, percentile.
        """
        y = self.df[self.target].astype(float)
        return y.min(), y.max(), np.sort(y)[int(y.shape[0] * percentile)]

    # def load_tar(self, xyz_file: str) -> None:
    #     """Load the data from a tar file.

    #     Args:
    #         xyz_file: name of the tar file.
    #     """
    #     f = tarfile.TarFile(xyz_file, "r")
    #     labels = self.properties
    #     all_mols = []
    #     for _pt in f:
    #         data = f.extractfile(_pt).read().decode().splitlines()
    #         all_mols.append(
    #             data[-2].split()[:1] + list(map(float, data[1].split()[2:]))  # type: ignore
    #         )
    #     self.df = pd.DataFrame(all_mols, columns=["SMILES"] + labels)

    # def load_zip(self, xyz_file) -> None:
    #     """Load the data from a zip file.

    #     Args:
    #         xyz_file: name of the zip file.
    #     """
    #     f = zipfile.ZipFile(xyz_file, "r")
    #     labels = self.properties
    #     all_mols = []
    #     data = f.extractall().read().decode().splitlines()
    #     all_mols.append(
    #         data[-2].split()[:1] + list(map(float, data[1].split()[2:]))  # type: ignore
    #     )
    #     self.df = pd.DataFrame(all_mols, columns=["SMILES"] + labels)

    def convert2h5(
        self,
        xyz_path: str,
        h5_path: str = "qm9.h5",
        property_names: List[str] = [
            "rA",
            "rB",
            "rC",
            "mu",
            "alpha",
            "homo",
            "lumo",
            "gap",
            "r2",
            "zpve",
            "U0",
            "U",
            "H",
            "G",
            "Cv",
        ],
    ) -> None:

        # Reads the xyz files and return the properties, smiles and coordinates
        data = []
        smiles = []
        properties = []
        i = 0

        for file in os.listdir(xyz_path):
            try:
                path = os.path.join(xyz_path, file)
                atoms, coordinates, smile, prop = self._read_xyz(path)
                data.append(
                    (atoms, coordinates)
                )  # A tuple with the atoms and its coordinates
                smiles.append(smile)  # The SMILES representation
                properties.append(prop)  # The molecules properties
            except ValueError:
                print(path)
            i += 1

        # rename relevant properties to match GFN schema
        labels = property_names
        df = pd.DataFrame(properties, columns=labels)
        df["SMILES"] = smiles

        df.to_hdf(h5_path, key="df", mode="w")

    def _read_xyz(self, path: str):
        """Reads the xyz files in the directory on 'path'.
        Code adapted from # https://www.kaggle.com/code/rmonge/predicting-molecule-properties-based-on-its-smiles/notebook

            Args:
                path: the path to the folder to be read.

            Returns:
                atoms: list with the characters representing the atoms of a molecule.
                coordinates: list with the cartesian coordinates of each atom.
                smile: list with the SMILE representation of a molecule.
                prop: list with the scalar properties.
        """
        atoms = []
        coordinates = []

        with open(path, "r") as file:
            lines = file.readlines()
            n_atoms = int(lines[0])  # the number of atoms
            smile = lines[n_atoms + 3].split()[0]  # smiles string
            prop = lines[1].split()[2:]  # scalar properties

            # to retrieve each atmos and its cartesian coordenates
            for atom in lines[2 : n_atoms + 2]:
                line = atom.split()
                # which atom
                atoms.append(line[0])

                # its coordinate
                # Some properties have '*^' indicading exponentiation
                try:
                    coordinates.append((float(line[1]), float(line[2]), float(line[3])))
                except ValueError:
                    coordinates.append(
                        (
                            float(line[1].replace("*^", "e")),
                            float(line[2].replace("*^", "e")),
                            float(line[3].replace("*^", "e")),
                        )
                    )

        return atoms, coordinates, smile, prop

    def __len__(self):
        return len(self.idcs)

    def __getitem__(self, idx):
        return (
            Chem.MolFromSmiles(self.df["SMILES"][self.idcs[idx]]),
            self.df[self.target][self.idcs[idx]],
        )