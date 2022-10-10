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
import typing
from typing import List

import importlib_metadata
import torch
from packaging import version
from torch.optim.lr_scheduler import (  # type: ignore
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    SequentialLR,
    StepLR,
    _LRScheduler,
)
from torch.utils.data.dataset import (
    ChainDataset,
    ConcatDataset,
    Dataset,
    IterableDataset,
    Subset,
    TensorDataset,
)

sane_datasets = [
    Dataset,
    ChainDataset,
    ConcatDataset,
    IterableDataset,
    Subset,
    TensorDataset,
]

torch_version = version.parse(importlib_metadata.version("torch"))

if torch_version < version.parse("1.12") and torch_version >= version.parse("1.10"):
    from torch.utils.data.dataset import DFIterDataPipe  # type: ignore

    sane_datasets.append(DFIterDataPipe)

if torch_version < version.parse("1.12") and torch_version >= version.parse("1.11"):
    from torch.utils.data.dataset import IterDataPipe, MapDataPipe  # type: ignore

    sane_datasets.extend([IterDataPipe, MapDataPipe])


sane_schedulers = [
    _LRScheduler,
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    MultiStepLR,
    MultiplicativeLR,
    OneCycleLR,
    SequentialLR,
    StepLR,
]


@typing.no_type_check
def fix_datasets(sane_datasets: List[Dataset]) -> None:
    """
    Helper function to revert TorchDrug dataset handling (which breaks core
    pytorch functionalities). For details see:
    https://github.com/DeepGraphLearning/torchdrug/issues/96

    Args:
        sane_datasets: A list of pytorch datasets.

    Raises:
        AttributeError: If a passed dataset was not sane.
    """
    dataset = sane_datasets[0]
    torch.utils.data.dataset.Dataset = dataset  # type: ignore
    torch.utils.data.dataset.ChainDataset = sane_datasets[1]  # type: ignore
    torch.utils.data.dataset.ConcatDataset = sane_datasets[2]  # type: ignore
    torch.utils.data.dataset.IterableDataset = sane_datasets[3]  # type: ignore
    torch.utils.data.dataset.Subset = sane_datasets[4]  # type: ignore
    torch.utils.data.dataset.TensorDataset = sane_datasets[5]  # type: ignore

    if torch_version < version.parse("1.12") and torch_version >= version.parse("1.10"):
        torch.utils.data.dataset.DFIterDataPipe = sane_datasets[6]  # type: ignore

    if torch_version < version.parse("1.12") and torch_version >= version.parse("1.11"):
        torch.utils.data.dataset.IterDataPipe = sane_datasets[7]  # type: ignore
        torch.utils.data.dataset.MapDataPipe = sane_datasets[8]  # type: ignore

    for ds in sane_datasets[1:]:
        if not issubclass(ds, dataset):
            raise AttributeError(
                f"Reverting silent TorchDrug overwriting failed, {ds} is not a subclass"
                f" of {dataset}."
            )


@typing.no_type_check
def fix_schedulers(sane_schedulers: List[_LRScheduler]) -> None:
    """
    Helper function to revert TorchDrug LR scheduler handling (which breaks core
    pytorch functionalities). For details see:
    https://github.com/DeepGraphLearning/torchdrug/issues/96

    Args:
        sane_schedulers: A list of pytorch lr_schedulers.

    Raises:
        AttributeError: If a passed lr_scheduler was not sane.
    """
    scheduler = sane_schedulers[0]
    torch.optim.lr_scheduler._LRScheduler = scheduler  # type: ignore
    torch.optim.lr_scheduler.ChainedScheduler = sane_schedulers[1]  # type: ignore
    torch.optim.lr_scheduler.ConstantLR = sane_schedulers[2]  # type: ignore
    torch.optim.lr_scheduler.CosineAnnealingLR = sane_schedulers[3]  # type: ignore
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts = sane_schedulers[4]  # type: ignore
    torch.optim.lr_scheduler.CyclicLR = sane_schedulers[5]  # type: ignore
    torch.optim.lr_scheduler.ExponentialLR = sane_schedulers[6]  # type: ignore
    torch.optim.lr_scheduler.LambdaLR = sane_schedulers[7]  # type: ignore
    torch.optim.lr_scheduler.LinearLR = sane_schedulers[8]  # type: ignore
    torch.optim.lr_scheduler.MultiStepLR = sane_schedulers[9]  # type: ignore
    torch.optim.lr_scheduler.MultiplicativeLR = sane_schedulers[10]  # type: ignore
    torch.optim.lr_scheduler.OneCycleLR = sane_schedulers[11]  # type: ignore
    torch.optim.lr_scheduler.SequentialLR = sane_schedulers[12]  # type: ignore
    torch.optim.lr_scheduler.StepLR = sane_schedulers[13]  # type: ignore

    for lrs in sane_schedulers[1:]:
        if not issubclass(lrs, scheduler):
            raise AttributeError(
                f"Reverting silent TorchDrug overwriting failed, {lrs} is not a subclass"
                f" of {scheduler}."
            )
