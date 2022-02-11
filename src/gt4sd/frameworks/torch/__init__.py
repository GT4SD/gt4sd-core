"""Generic utils for pytorch."""

from typing import Optional, Union

import torch


def get_device() -> torch.device:
    """
    Get device dynamically.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_claim(device: Optional[Union[torch.device, str]] = None) -> torch.device:
    """
    Satidfy a device claim.

    Args:
        device: device where the inference
            is running either as a dedicated class or a string. If not provided is inferred.

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
