#!/usr/bin/env python
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

"""Run inference pipelines for the GT4SD."""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, cast

from ..algorithms.registry import ApplicationsRegistry
from .algorithms import (
    AVAILABLE_ALGORITHMS,
    AVAILABLE_ALGORITHMS_CATEGORIES,
    filter_algorithm_applications,
)
from .argument_parser import ArgumentParser, DataClassType

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmApplicationArguments:
    """Algorithm application arguments."""

    __name__ = "algorithm_base_args"

    algorithm_type: Optional[str] = field(
        default=None,
        metadata={
            "help": f"Inference algorithm type, supported types: {', '.join(AVAILABLE_ALGORITHMS_CATEGORIES['algorithm_type'])}."
        },
    )
    domain: Optional[str] = field(
        default=None,
        metadata={
            "help": f"Domain of the inference algorithm, supported types: {', '.join(AVAILABLE_ALGORITHMS_CATEGORIES['domain'])}."
        },
    )
    algorithm_name: Optional[str] = field(
        default=None,
        metadata={"help": "Inference algorithm name."},
    )
    algorithm_application: Optional[str] = field(
        default=None,
        metadata={"help": "Inference algorithm application."},
    )
    algorithm_version: Optional[str] = field(
        default=None,
        metadata={"help": "Inference algorithm version."},
    )


@dataclass
class InferenceArguments:
    """Inference arguments."""

    __name__ = "inference_base_args"

    target: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional target for generation represented as a string. Defaults to None, it can be "
                "also provided in the configuration_file as an object, but the commandline takes precendence."
            )
        },
    )
    number_of_samples: int = field(
        default=5,
        metadata={"help": "Number of generated samples, defaults to 5."},
    )
    configuration_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Configuration file for the inference pipeline in JSON format."
        },
    )
    print_info: bool = field(
        default=False,
        metadata={
            "help": "Print info for the selected algorithm, preventing inference run. Defaults to False."
        },
    )


def main() -> None:
    """Run an inference pipeline."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = ArgumentParser(
        cast(
            Iterable[DataClassType], (AlgorithmApplicationArguments, InferenceArguments)
        )
    )
    algorithm_args, inference_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    filters = algorithm_args.__dict__
    matching_algorithms = filter_algorithm_applications(
        algorithms=AVAILABLE_ALGORITHMS, filters=filters
    )
    if len(matching_algorithms) > 1:
        logger.info(
            f"Multiple algorithms matching the parameters:{os.linesep}"
            f"{os.linesep.join(map(str, matching_algorithms))}{os.linesep}"
            f"Select one by specifying additional algorithms parameters: {','.join('--' + key for key, value in filters.items() if not value)}.",
        )
        return
    elif len(matching_algorithms) < 1:
        provided_filters = {key: value for key, value in filters.items() if value}
        logger.error(
            "No algorithms matching the provided parameters, "
            f"please review the supported algorithms:{os.linesep}"
            f"{os.linesep.join(map(str, matching_algorithms))}{os.linesep}"
            f"Please review the parameters provided:{os.linesep}"
            f"{provided_filters}"
        )
    selected_algorithm = matching_algorithms[0]
    logger.info(f"Selected algorithm: {selected_algorithm}")
    target = inference_args.target
    number_of_samples = inference_args.number_of_samples
    print_info = inference_args.print_info
    configuration_filepath = inference_args.configuration_file

    if print_info:
        algorithm_configuration = ApplicationsRegistry.get_configuration_instance(
            **selected_algorithm
        )
        algorithm_configuration_dict = {**algorithm_configuration.to_dict()}
        _ = algorithm_configuration_dict.pop("description", None)
        logger.info(
            f"Selected algorithm support the following configuration parameters:{os.linesep}"
            f"{json.dumps(algorithm_configuration_dict, indent=1)}{os.linesep}"
            f"Target information:{os.linesep}"
            f"{json.dumps({'target': algorithm_configuration.get_target_description()}, indent=1)}"
        )
        return

    configuration: Dict[str, Any] = {}

    if configuration_filepath is not None:
        with open(configuration_filepath) as fp:
            configuration = json.load(fp)
    else:
        logger.info("No configuration file provided, running using default parameters.")

    if target is not None:
        if "target" in configuration:
            logger.info(
                "Target provided both via commandline and configuration file. "
                f"The commandline one will be used: {target}."
            )
        configuration["target"] = target

    algorithm = ApplicationsRegistry.get_application_instance(
        target=configuration.pop("target", None),
        **selected_algorithm,
        **configuration,
    )
    logger.info(
        f"Starting generation with the following configuration:{algorithm.configuration}"
    )
    print(
        f"{os.linesep.join(map(str, algorithm.sample(number_of_items=number_of_samples)))}"
    )


if __name__ == "__main__":
    main()
