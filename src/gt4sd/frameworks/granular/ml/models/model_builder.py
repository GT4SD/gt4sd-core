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
"""Model builder module."""

import logging
from collections import OrderedDict
from typing import Any, Dict, List
from typing import OrderedDict as OrderedDictType

import torch

from ....torch import device_claim
from . import ARCHITECTURE_FACTORY
from .base_model import GranularBaseModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def build_model(architecture: Dict[str, Any]) -> GranularBaseModel:
    """Build model from architecture configuration.

    Args:
        architecture: architecture configuration.

    Returns:
        built model.
    """
    model_name = architecture["name"]
    model_type = architecture["type"].lower()
    if model_type not in ARCHITECTURE_FACTORY:
        raise ValueError(
            f"model_type={model_type} not supported. Pick a valid one: {sorted(ARCHITECTURE_FACTORY.keys())}"
        )
    model = ARCHITECTURE_FACTORY[model_type](
        data=architecture["data"], **architecture["hparams"]
    )

    if architecture["start_from_checkpoint"]:
        loaded_params = torch.load(
            architecture["hparams"]["checkpoint_path"], map_location=device_claim(None)
        )
        loaded_architecture_latent = loaded_params["hyper_parameters"][
            "architecture_latent_models"
        ]
        loaded_architecture_autoencoder = loaded_params["hyper_parameters"][
            "architecture_autoencoders"
        ]
        for arcihtecture_autoencoder in loaded_architecture_autoencoder:
            if model_name == arcihtecture_autoencoder["name"]:
                architecture = arcihtecture_autoencoder
        for architecture_latent in loaded_architecture_latent:
            if model_name == architecture_latent["name"]:
                architecture = architecture_latent
        loaded_state_dict: OrderedDictType[str, torch.Tensor] = OrderedDict()
        for layer_name in loaded_params["state_dict"]:
            state_model_name, *layer_name_elements = layer_name.split(".")
            state_name = ".".join(layer_name_elements)
            try:
                checkpoint_model_name = architecture["hparams"]["checkpoint_model_name"]
            except Exception:
                checkpoint_model_name = None
            if (
                state_model_name == model_name
                or state_model_name == checkpoint_model_name
            ):
                loaded_state_dict[state_name] = loaded_params["state_dict"][layer_name]
        model.load_state_dict(loaded_state_dict)
        model.name = model_name
        model.data = architecture["data"]
        model.target_key = model_name + "_" + architecture["data"]["target"]
        try:
            freeze_weights = architecture["freeze_weights"]
        except KeyError:
            freeze_weights = None

        if freeze_weights:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        if model_type == "mlp_predictor":
            model.from_position = architecture["from_position"]
        else:
            model.position = architecture["position"]
            model.input_key = model_name + "_" + architecture["data"]["input"]
    return model


def building_models(architectures: List[Dict[str, Any]]) -> List[GranularBaseModel]:
    """Building models given architecture configurations.

    Args:
        architectures: list of architecture configurations.

    Returns:
        a list of models.
    """
    return [build_model(architecture) for architecture in architectures]


def define_latent_models_input_size(
    architecture_autoencoders: List[Dict[str, Any]],
    architecture_latent_models: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Define latent models input size and return the updated configurations.

    Args:
        architecture_autoencoders: list of autoencoder architecture configurations.
        architecture_latent_models: list of latent model architecture configurations.

    Returns:
        list of update latent model architecture configurations.
    """
    size_autoencoder: Dict[str, int] = dict()
    for architecture in architecture_autoencoders:
        if architecture["position"] not in size_autoencoder.keys():
            size_autoencoder[architecture["position"]] = architecture["hparams"][
                "latent_size"
            ]
        else:
            logger.warning(f"position for architecture={architecture} is not unique!")

    updated_architecture_latent_models = []
    for _, architecture in enumerate(architecture_latent_models):
        architecture["hparams"]["input_size"] = sum(
            [size_autoencoder[pos] for pos in architecture["from_position"]]
        )
        updated_architecture_latent_models.append(architecture)

    return updated_architecture_latent_models
