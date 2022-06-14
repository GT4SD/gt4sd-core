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
"""Dataset module."""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from torch.utils.data import Dataset

from ..ml.models import ARCHITECTURE_FACTORY, AUTOENCODER_ARCHITECTURES
from ..tokenizer.tokenizer import TOKENIZER_FACTORY, Tokenizer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SCALING_FACTORY_FN: Dict[str, Callable] = {
    "onehot": lambda: OneHotEncoder(handle_unknown="error", sparse=False),
    "min-max": lambda: MinMaxScaler(),
    "standard": lambda: StandardScaler(),
}
MODEL_TYPES = set(ARCHITECTURE_FACTORY.keys())


class GranularDataset(Dataset):
    """A dataset wrapper for granular"""

    def __init__(self, name: str, data: Dict[str, Any]) -> None:
        """Initialize a granular dataset.

        Args:
            name: dataset name.
            data: dataset samples.
        """
        self.dataset: Dict[str, Any] = {"name": name, "data": data}

        self.tokenizer: Tokenizer
        self.set_seq_size: int
        self.input_size: int
        self.target_size: int

    def __len__(self) -> int:
        """Dataset length.

        Returns:
            length of the dataset.
        """
        lengths = {key: len(data) for key, data in self.dataset["data"].items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"mismatching dimensions for the data: {lengths}")
        return list(lengths.values())[0]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Retrieve an item from the dataset by index.

        Args:
            index: index for the item.

        Returns:
            an item.
        """
        result = dict()
        for key in self.dataset["data"]:
            result[self.dataset["name"] + "_" + key] = self.dataset["data"][key][index]
        return result


class CombinedGranularDataset(Dataset):
    """General dataset combining multiple granular datasets."""

    def __init__(self, datasets: List[Dict[str, Any]]) -> None:
        """Initialize a general dataset.

        Args:
            datasets: list of dataset configurations.
        """
        self.datasets = datasets
        self.names = [data["name"] for data in datasets]

    def __len__(self) -> int:
        """Dataset length.

        Returns:
            length of the dataset.
        """
        return len([*self.datasets[0]["data"].values()][0])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Retrieve an item from the dataset by index.

        Args:
            index: index for the item.

        Returns:
            an item.
        """
        result = dict()
        for dataset in self.datasets:
            keys = [*dataset["data"]]
            for key in keys:
                result[dataset["name"] + "_" + key] = dataset["data"][key][index]
        return result


class SmilesTokenizationPreProcessingDataset(GranularDataset):
    """Dataset for SMILES/SELFIES preprocessing."""

    def __init__(
        self,
        name: str,
        data_columns: Dict[str, Any],
        input_smiles: pd.DataFrame,
        target_smiles: pd.DataFrame,
        tokenizer: Tokenizer,
        set_seq_size: Optional[int] = None,
    ) -> None:
        """Construct a SmilesTokenizationPreProcessingDataset.

        Args:
            name: dataset name.
            data_columns: data columns mapping.
            input_smiles: dataframe containing input SMILES.
            target_smiles: dataframe containing target SMILES.
            tokenizer: a tokenizer defining the molecule representation used.
            set_seq_size: sequence size. Defaults to None, a.k.a., define this
                using the input SMILES.
        """
        self.name = name
        self.input_smiles = input_smiles.values.flatten().tolist()
        self.target_smiles = target_smiles.values.flatten().tolist()
        self.tokenizer = tokenizer
        self.input_tokens: List[torch.Tensor] = []
        self.target_tokens: List[torch.Tensor] = []

        tokens_ids = [
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(smile))
            for smile in self.input_smiles
        ]
        if set_seq_size:
            self.set_seq_size = set_seq_size
        else:
            self.set_seq_size = max([len(i) for i in tokens_ids]) + 20

        self.smiles_to_ids(input_smiles=self.input_smiles)
        self.smiles_to_ids(target_smiles=self.target_smiles)

        super().__init__(
            name=name,
            data={
                data_columns["input"]: self.input_tokens,
                data_columns["target"]: self.target_tokens,
            },
        )

    def smiles_to_ids(
        self, input_smiles: List[str] = [], target_smiles: List[str] = []
    ) -> None:
        """Process input SMILES lists generating examples by tokenizing strings and converting them to tensors.

        Args:
            input_smiles: list of input SMILES representations. Defaults to [].
            target_smiles: list of target SMILES representations. Defaults to [].
        """
        if len(input_smiles) > 0 and len(target_smiles) == 0:
            self.input_smiles = input_smiles
            smiles = input_smiles
        elif len(input_smiles) == 0 and len(target_smiles) > 0:
            self.target_smiles = target_smiles
            smiles = target_smiles
        else:
            raise Exception(
                "Either input_smiles or target_smiles needs to be specified"
            )

        tokens_ids = [
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(smile))
            for smile in smiles
        ]
        examples = []
        for token in tokens_ids:
            example_tokens = self.tokenizer.convert_tokens_to_ids(
                [self.tokenizer.sos_token]
            )
            example_tokens.extend(token)
            example_tokens.extend(
                self.tokenizer.convert_tokens_to_ids([self.tokenizer.eos_token])
            )
            examples.append(
                torch.tensor(
                    self.tokenizer.add_padding_tokens(example_tokens, self.set_seq_size)
                )
            )

        if len(input_smiles) > 0 and len(target_smiles) == 0:
            self.input_tokens = examples
        elif len(input_smiles) == 0 and len(target_smiles) > 0:
            self.target_tokens = examples


class LatentModelDataset(GranularDataset):
    """Latent model dataset."""

    def __init__(
        self,
        name: str,
        data_columns: Dict[str, Any],
        target_data: pd.DataFrame,
        scaling: Optional[str] = None,
    ) -> None:
        """Construct a LatentModelDataset.

        Args:
            name: dataset name.
            data_columns: data columns mapping.
            target_data: dataframe for targets.
            scaling: feature scaling process. Defaults to None, a.k.a. no scaling. Currently not supported.

        Raises:
            NotImplementedError: in case a scaling is selected.
        """
        self.name = name
        if scaling:
            raise NotImplementedError("Scaling not yet supported")
        self.target_data = torch.from_numpy(target_data.values)
        self.target_data = self.target_data.type(torch.float)
        self.target_size = target_data.shape[1]
        super().__init__(name=name, data={data_columns["target"]: self.target_data})


class AutoEncoderDataset(GranularDataset):
    """Autoencoder dataset."""

    def __init__(
        self,
        name: str,
        data_columns: Dict[str, Any],
        input_data: pd.DataFrame,
        target_data: pd.DataFrame,
        scaling: Optional[str] = None,
    ) -> None:
        """Construct an AutoEncoderDataset.

        Args:
            name: dataset name.
            data_columns: data columns mapping.
            input_data: dataframe for inputs.
            target_data: dataframe for targets.
            scaling: feature scaling process. Defaults to None, a.k.a. no scaling. Feasible values: "onehot", "min-max" and "standard".

        Raises:
            ValueError: in case requested scaling is not supported.
        """
        self.name = name
        self.data_columns = data_columns

        if scaling is None:
            self.input_data = torch.from_numpy(input_data.values)
            self.target_data = torch.from_numpy(target_data.values)
        else:
            if scaling not in SCALING_FACTORY_FN:
                raise ValueError(
                    f"Scaling={scaling} not supported. Pick a valid one: {sorted(list(SCALING_FACTORY_FN.keys()))}"
                )

            self.input_scaling = ColumnTransformer(
                transformers=[
                    (
                        "InputScaling",
                        SCALING_FACTORY_FN[scaling](),
                        [data_columns["input"]],
                    )
                ]
            )
            self.target_scaling = ColumnTransformer(
                transformers=[
                    (
                        "TargetScaling",
                        SCALING_FACTORY_FN[scaling](),
                        [data_columns["target"]],
                    )
                ]
            )

            self.input_data = torch.from_numpy(
                self.input_scaling.fit_transform(pd.concat([input_data], axis=1))
            )
            self.target_data = torch.from_numpy(
                self.target_scaling.fit_transform(pd.concat([target_data], axis=1))
            )

        self.input_data, self.target_data = (
            self.input_data.type(torch.float),
            self.target_data.type(torch.float),
        )
        self.input_size = self.input_data.shape[1]
        self.target_size = self.target_data.shape[1]

        super().__init__(
            name=name,
            data={
                data_columns["input"]: self.input_data,
                data_columns["target"]: self.target_data,
            },
        )


DATASET_FACTORY: Dict[str, Type[GranularDataset]] = {
    "latentmodel": LatentModelDataset,
    "smiles": SmilesTokenizationPreProcessingDataset,
    "big-smiles": SmilesTokenizationPreProcessingDataset,
    "selfies": SmilesTokenizationPreProcessingDataset,
    "autoencoder": AutoEncoderDataset,
}


def build_data_columns(hparams: Dict[str, Any]) -> Dict[str, Any]:
    """Build data columns from hyper-parameters.

    Args:
        hparams: hyper-parameters for the data columns.

    Returns:
        data columns.
    """
    try:
        input_columns = hparams["input"]
    except KeyError:
        input_columns = None
    try:
        target_columns = hparams["target"]
    except KeyError:
        target_columns = None
    # create dictionary
    if input_columns:
        data_columns = {"input": input_columns, "target": target_columns}
    else:
        data_columns = {"target": target_columns}
    return data_columns


def build_dataset(
    name: str,
    data: pd.DataFrame,
    dataset_type: str,
    data_columns: Dict[str, Any],
    hparams: Dict[str, Any],
) -> GranularDataset:
    """Build a granular dataset.

    Args:
        name: dataset name.
        data: dataframe representing the dataset.
        dataset_type: dataset type. Feasible values: "latentmodel", "smiles", "selfies", "big-smiles" and "autoencoder".
        data_columns: data columns mapping.
        hparams: hyper-parameters for the data columns.

    Raises:
            ValueError: in case requested dataset type is not supported.

    Returns:
        a granular dataset.
    """
    dataset: GranularDataset
    dataset_type = dataset_type.lower()
    if dataset_type not in DATASET_FACTORY:
        raise ValueError(
            f"dataset_type={dataset_type} not supported. Pick a valid one: {sorted(list(DATASET_FACTORY.keys()))}"
        )

    input_columns: List[Any]
    if not dataset_type == "latentmodel":
        if data_columns["input"] == "all":
            input_columns = data.columns.tolist()
        else:
            if isinstance(data_columns["input"], list):
                input_columns = data_columns["input"]
            else:
                input_columns = [data_columns["input"]]

    target_columns: List[Any]
    if data_columns["target"] == "all":
        target_columns = data.columns.tolist()
    else:
        if isinstance(data_columns["target"], list):
            target_columns = data_columns["target"]
        else:
            target_columns = [data_columns["target"]]

    if dataset_type in {"smiles", "selfies", "big-smiles"}:
        try:
            build_vocab = hparams["build_vocab"]
        except KeyError:
            build_vocab = None
        try:
            sequence_size = hparams["sequence_size"]
        except KeyError:
            sequence_size = None
        vocab_file = hparams["vocab_file"]

        # build tokenizer
        if build_vocab:
            tokenizer = TOKENIZER_FACTORY[dataset_type](
                vocab_file, smiles=data[input_columns].squeeze().tolist()
            )
        else:
            tokenizer = TOKENIZER_FACTORY[dataset_type](vocab_file, smiles=[])
        dataset = SmilesTokenizationPreProcessingDataset(
            name=name,
            data_columns=data_columns,
            input_smiles=data[input_columns],
            target_smiles=data[target_columns],
            tokenizer=tokenizer,
            set_seq_size=sequence_size,
        )
    elif dataset_type == "latentmodel":
        dataset = LatentModelDataset(
            name=name,
            data_columns=data_columns,
            target_data=data[target_columns],
            scaling=None,
        )
    elif dataset_type == "autoencoder":
        dataset = AutoEncoderDataset(
            name=name,
            data_columns=data_columns,
            input_data=data[input_columns],
            target_data=data[target_columns],
            scaling=hparams["scaling"],
        )

    return dataset


def build_architecture(
    model_type: str,
    data_columns: Dict[str, Any],
    dataset: GranularDataset,
    hparams: Dict[str, Any],
) -> Dict[str, Any]:
    """Build architecture configuration for the selected model type and dataset.

    Args:
        model_type: model type. Feasible values: "vae_rnn", "vae_trans", "mlp_predictor", "no_encoding", "mlp_autoencoder" and "vae_mlp".
        data_columns: data columns mapping.
        dataset: a granular dataset.
        hparams: hyper-parameters for the data columns.

    Raises:
        ValueError: in case requested model type is not supported.

    Returns:
        architecture configuration.
    """
    model_type = model_type.lower()
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"model_type={model_type} not supported. Pick a valid one: {sorted(list(MODEL_TYPES))}"
        )

    architecture: Dict[str, Any] = {
        "name": hparams["name"],
        "type": hparams["type"],
        "start_from_checkpoint": hparams["start_from_checkpoint"],
        "freeze_weights": hparams["freeze_weights"],
        "data": data_columns,
        "hparams": hparams,
    }

    if model_type in AUTOENCODER_ARCHITECTURES:
        architecture["position"] = hparams["position"]
        if model_type in {"vae_rnn", "vae_trans"}:
            hparams["tokenizer"] = dataset.tokenizer
            hparams["vocab_size"] = dataset.tokenizer.vocab_size
            if model_type == "vae_rnn":
                hparams["embedding_size"] = dataset.set_seq_size
            else:  # "vae_trans"
                hparams["sequence_len"] = dataset.set_seq_size
        elif model_type == "no_encoding":
            hparams["latent_size"] = dataset.input_size
        elif model_type in {"mlp_autoencoder", "vae_mlp"}:
            hparams["input_size_enc"] = dataset.input_size
            hparams["output_size_dec"] = dataset.target_size
    else:  # "mlp_predictor"
        hparams["output_size"] = dataset.target_size
        architecture["from_position"] = hparams["from_position"]

    return architecture


def build_dataset_and_architecture(
    name: str,
    data_path: str,
    data_file: str,
    dataset_type: str,
    model_type: str,
    hparams: Dict[str, Any],
    **kwargs,
) -> Tuple[GranularDataset, Dict[str, Any]]:
    """Build a dataset and an architecture configuration.

    Args:
        name: dataset name.
        data_path: path to the dataset.
        data_file: data file name.
        dataset_type: dataset type. Feasible values: "latentmodel", "smiles", "selfies", "big-smiles" and "autoencoder".
        model_type: model type. Feasible values: "vae_rnn", "vae_trans", "mlp_predictor", "no_encoding", "mlp_autoencoder" and "vae_mlp".
        hparams: hyper-parameters for the data columns.

    Raises:
        ValueError: in case the data file has an unsupported extension/format.

    Returns:
        a tuple containing a granular dataset and a related architecture configuration.
    """
    if data_file.endswith(".csv"):
        data = pd.read_csv(f"{data_path}{os.path.sep}{data_file}")
    elif data_file.endswith(".bz2") or data_file.endswith(".pkl"):
        data = pd.read_pickle(f"{data_path}{os.path.sep}{data_file}")
    else:
        raise ValueError(
            f"data_file={data_file} extension not supported. Use a compatible extension/format: {['.csv', '.bz2', '.pkl']}"
        )
    data_columns = build_data_columns(hparams)
    dataset = build_dataset(name, data, dataset_type, data_columns, hparams)
    architecture = build_architecture(model_type, data_columns, dataset, hparams)
    return dataset, architecture
