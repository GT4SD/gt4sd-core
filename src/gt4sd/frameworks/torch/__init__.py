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
"""Generic utils for pytorch."""

from typing import Dict, List, Optional, Union

import torch


def get_gpu_device_names() -> List[str]:
    """Get GPU device names as a list.

    Returns:
        names of available GPU devices.
    """
    gpu_device_names = []
    if torch.cuda.is_available():
        gpu_device_names = [
            f"cuda:{index}" for index in range(torch.cuda.device_count())
        ]
    return gpu_device_names


def claim_device_name() -> str:
    """Claim a device name.

    Returns:
        device name, if on GPU is available returns CPU.
    """
    device_name = "cpu"
    gpu_device_names = get_gpu_device_names()
    if len(gpu_device_names) > 0:
        device_name = gpu_device_names[0]
    return device_name


def get_device() -> torch.device:
    """
    Get device dynamically.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_claim(device: Optional[Union[torch.device, str]] = None) -> torch.device:
    """
    Satidfy a device claim.

    Args:
        device: device where the inference is running either as a dedicated class or
            a string. If not provided is inferred.

    Returns:
        torch.device: the claimed device or a default one.
    """
    if isinstance(device, str):
        device = torch.device(device)
    device = (
        get_device()
        if (device is None or not isinstance(device, torch.device))
        else device
    )
    return device


def get_device_from_tensor(tensor: torch.Tensor) -> torch.device:
    """Get the device from a tensor.

    Args:
        tensor: a tensor.

    Returns:
        the device.
    """
    device_id = tensor.get_device()
    device = "cpu" if device_id < 0 else f"cuda:{device_id}"
    return device_claim(device)


def map_tensor_dict(
    tensor_dict: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Maps a dictionary of tensors to a specific device.

    Args:
        tensor_dict: A dictionary of tensors.
        device: The device to map the tensors to.

    Returns:
        A dictionary of tensors mapped to the device.
    """
    return {key: tensor.to(device) for key, tensor in tensor_dict.items()}
