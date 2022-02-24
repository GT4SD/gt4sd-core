#!/usr/bin/env python

"""Run inference pipelines for the GT4SD."""


import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, cast

from ..algorithms.registry import ApplicationsRegistry
from .argument_parser import ArgumentParser, DataClassType

logger = logging.getLogger(__name__)

AVAILABLE_ALGORITHMS = ApplicationsRegistry.list_available()
AVAILABLE_ALGORITHMS_CATEGORIES = {
    category: sorted(set([algorithm[category] for algorithm in AVAILABLE_ALGORITHMS]))
    for category in ["domain", "algorithm_type"]
}


@dataclass
class InfereceArguments:
    """Inference arguments."""

    __name__ = "inference_base_args"

    algorithm_type: Optional[str] = field(
        metadata={
            "help": f"Inference algorithm type, supported types: {', '.join(AVAILABLE_ALGORITHMS_CATEGORIES['algorithm_type'])}."
        },
    )
    domain: Optional[str] = field(
        metadata={
            "help": f"Domain of the inference algorithm, supported types: {', '.join(AVAILABLE_ALGORITHMS_CATEGORIES['domain'])}."
        },
    )
    algorithm_name: Optional[str] = field(
        metadata={"help": "Inference algorithm name."},
    )
    algorithm_application: Optional[str] = field(
        metadata={"help": "Inference algorithm application."},
    )
    algorithm_version: Optional[str] = field(
        default=None,
        metadata={"help": "Inference algorithm version."},
    )
    target: Optional[str] = field(
        default=None,
        metadata={"help": "Optional target for generation."},
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


def main() -> None:
    """
    Run an inference pipeline.

    Raises:
        ValueError: isn case the provided training pipeline provided is not supported.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    args = ArgumentParser(
        cast(DataClassType, InfereceArguments)
    ).parse_args_into_dataclasses(return_remaining_strings=True)[0]
    algorithm_type = args.algorithm_type
    domain = args.domain
    algorithm_name = args.algorithm_name
    algorithm_application = args.algorithm_application
    algorithm_version = args.algorithm_version
    target = args.target
    number_of_samples = args.number_of_samples

    configuration: Dict[str, Any] = {}
    configuration_filepath = args.configuration_file
    with open(configuration_filepath) as fp:
        configuration = json.load(fp)

    algorithm = ApplicationsRegistry.get_application_instance(
        target=target,
        algorithm_type=algorithm_type,
        domain=domain,
        algorithm_name=algorithm_name,
        algorithm_application=algorithm_application,
        algorithm_version=algorithm_version,
        **configuration,
    )
    logger.info(f"{algorithm.sample(number_of_items=number_of_samples)}")


if __name__ == "__main__":
    main()
