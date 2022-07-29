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
import tarfile
import zipfile
from typing import List

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from torch.utils.data import Dataset

LABELS: List[str] = [
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
]


class QM9Dataset(Dataset):
    def __init__(
        self,
        h5_file: str = None,
        xyz_file: str = None,
        train: bool = True,
        target: str = "gap",
        split_seed: int = 142857,
        ratio: float = 0.9,
    ) -> None:
        """Dataloader for QM9 dataset.

        Args:
            h5_file: data file in h5 format.
            xyz_file: data file in xyz format.
            train: split.
            target: target.
            split_seed: seed.
            ratio: ratio.
        """
        if h5_file is not None:
            self.df = pd.HDFStore(h5_file, "r")["df"]
        elif xyz_file is not None:
            self.load_tar(xyz_file)

        rng = np.random.default_rng(split_seed)
        idcs = np.arange(
            len(self.df)
        )  # TODO: error if there is no h5_file provided. Should h5 be required
        rng.shuffle(idcs)
        self.target = target
        if train:
            self.idcs = idcs[: int(np.floor(ratio * len(self.df)))]
        else:
            self.idcs = idcs[int(np.floor(ratio * len(self.df))) :]

    def get_stats(self, percentile: float = 0.95):
        """Get the stats of the dataset.

        Args:
            percentile: percentile.

        Returns:
            min, max, percentile.
        """
        y = self.df[self.target].astype(float)
        return y.min(), y.max(), np.sort(y)[int(y.shape[0] * percentile)]

    def load_tar(self, xyz_file: str):
        """Load the data from a tar file.

        Args:
            xyz_file: name of the tar file.
        """
        f = tarfile.TarFile(xyz_file, "r")
        labels = LABELS
        all_mols = []
        for pt in f:
            pt = f.extractfile(pt)
            data = pt.read().decode().splitlines()
            all_mols.append(
                data[-2].split()[:1] + list(map(float, data[1].split()[2:]))
            )
        self.df = pd.DataFrame(all_mols, columns=["SMILES"] + labels)

    def load_zip(self, xyz_file):
        """Load the data from a zip file.

        Args:
            xyz_file: name of the zip file.
        """
        f = zipfile.ZipFile(xyz_file, "r")
        labels = LABELS
        all_mols = []
        for pt in f:
            pt = f.extractall(pt)
            data = pt.read().decode().splitlines()
            all_mols.append(
                data[-2].split()[:1] + list(map(float, data[1].split()[2:]))
            )
        self.df = pd.DataFrame(all_mols, columns=["SMILES"] + labels)

    def __len__(self):
        return len(self.idcs)

    def __getitem__(self, idx):
        return (
            Chem.MolFromSmiles(self.df["SMILES"][self.idcs[idx]]),
            self.df[self.target][self.idcs[idx]],
        )
