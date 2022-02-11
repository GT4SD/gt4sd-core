"""Activations for granular models."""

from typing import Dict

from torch import nn

ACTIVATION_FACTORY: Dict[str, nn.Module] = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "softmax": nn.Softmax(),
    "relu": nn.ReLU(),
}
