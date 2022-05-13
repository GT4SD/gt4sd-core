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
from typing import Callable, List, Optional, Union

from torchdrug import data

# isort: off
from torch import nn

"""
Necessary because torchdrug silently overwrites the default nn.Module. This is quite
invasive and causes significant side-effects in the rest of the code.
See: https://github.com/DeepGraphLearning/torchdrug/issues/77
"""
nn.Module = nn._Module  # type: ignore


class TorchDrugDataset(data.MoleculeDataset):
    """A generic TorchDrug dataset class that can be fed with custom data"""

    def __init__(
        self,
        file_path: str,
        target_fields: str,
        smiles_field: str = "smiles",
        verbose: int = 1,
        lazy: Optional[bool] = False,
        transform: Optional[Callable] = None,
        node_feature: Optional[Union[str, List[str]]] = "default",
        edge_feature: Optional[Union[str, List[str]]] = "default",
        graph_feature: Optional[Union[str, List[str]]] = None,
        with_hydrogen: Optional[bool] = False,
        kekulize: Optional[bool] = False,
    ):
        """
        Constructor of TorchDrugDataset.

        Args:
            file_path: The path to the .csv file containing the data.
            target_fields: The columns containing the property to be optimized.
            smiles_field: The column name containing the SMILES. Defaults to 'smiles'.
            verbose: output verbose level. Defaults to 1.
            lazy: If yes, molecules are processed in the dataloader. This is faster
                for setup, but slower at training time. Defaults to False.
            transform: Optional data transformation function. Defaults to None.
            node_feature: Node features to extract. Defaults to 'default'.
            edge_feature: Edge features to extract. Defaults to 'default'.
            graph_feature: Graph features to extract. Defaults to None.
            with_hydrogen: Whether hydrogens are stored in molecular graph.
                Defaults to False.
            kekulize: Whether aromatic bonds are converted to single/double bonds.
                Defaults to False.
        """

        self.load_csv(
            file_path,
            smiles_field=smiles_field,
            target_fields=target_fields,
            verbose=verbose,
            lazy=lazy,
            transform=transform,
            node_feature=node_feature,
            edge_feature=edge_feature,
            graph_feature=graph_feature,
            with_hydrogen=with_hydrogen,
            kekulize=kekulize,
        )
