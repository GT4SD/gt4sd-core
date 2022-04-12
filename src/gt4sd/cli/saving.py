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

"""Run model saving for the GT4SD."""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import IO, Iterable, Optional, cast

from ..algorithms.registry import ApplicationsRegistry
from ..training_pipelines import TRAINING_PIPELINE_ARGUMENTS_FOR_MODEL_SAVING
from ..training_pipelines.core import TrainingPipelineArguments
from .algorithms import (
    AVAILABLE_ALGORITHMS,
    AVAILABLE_ALGORITHMS_CATEGORIES,
    filter_algorithm_applications,
    get_configuration_tuples,
)
from .argument_parser import ArgumentParser, DataClassType

logger = logging.getLogger(__name__)

SUPPORTED_TRAINING_PIPELINES = sorted(
    TRAINING_PIPELINE_ARGUMENTS_FOR_MODEL_SAVING.keys()
)


@dataclass
class SavingArguments:
    """Algorithm saving arguments."""

    __name__ = "saving_base_args"

    training_pipeline_name: str = field(
        metadata={
            "help": f"Training pipeline name, supported pipelines: {', '.join(SUPPORTED_TRAINING_PIPELINES)}."
        },
    )
    target_version: str = field(
        metadata={"help": "Target algorithm version to save."},
    )
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
    source_version: Optional[str] = field(
        default=None,
        metadata={"help": "Source algorithm version to use for missing artifacts."},
    )


class SavingArgumentParser(ArgumentParser):
    """Argument parser using a custom help logic."""

    def print_help(self, file: Optional[IO[str]] = None) -> None:
        """Print help checking dynamically whether a specific pipeline is passed.

        Args:
            file: an optional I/O stream. Defaults to None, a.k.a., stdout and stderr.
        """
        try:
            help_args_set = {"-h", "--help"}
            if (
                len(set(sys.argv).union(help_args_set)) < len(help_args_set) + 2
            ):  # considering filename
                super().print_help()
                return
            args = [arg for arg in sys.argv if arg not in help_args_set]
            parsed_arguments = super().parse_args_into_dataclasses(
                args=args, return_remaining_strings=True
            )
            trainer_arguments = None
            for arguments in parsed_arguments:
                if arguments.__name__ == "trainer_base_args":
                    trainer_arguments = arguments
                    break
            if trainer_arguments:
                trainer_arguments.training_pipeline_name
                training_pipeline_arguments = (
                    TRAINING_PIPELINE_ARGUMENTS_FOR_MODEL_SAVING.get(
                        trainer_arguments.training_pipeline_name,
                        TrainingPipelineArguments,
                    )
                )
                parser = ArgumentParser(
                    tuple(
                        [SavingArguments, *training_pipeline_arguments]  # type:ignore
                    )
                )
                parser.print_help()
        except Exception:
            super().print_help()


def main() -> None:
    """
    Run an algorithm saving pipeline.

    Raises:
        ValueError: in case the provided training pipeline provided is not supported.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    base_args = SavingArgumentParser(
        cast(DataClassType, SavingArguments)
    ).parse_args_into_dataclasses(return_remaining_strings=True)[0]
    training_pipeline_name = base_args.training_pipeline_name
    if training_pipeline_name not in set(SUPPORTED_TRAINING_PIPELINES):
        ValueError(
            f"Training pipeline {training_pipeline_name} is not supported. Supported types: {', '.join(SUPPORTED_TRAINING_PIPELINES)}."
        )
    training_pipeline_saving_arguments = TRAINING_PIPELINE_ARGUMENTS_FOR_MODEL_SAVING[
        training_pipeline_name
    ]
    parser = SavingArgumentParser(
        cast(
            Iterable[DataClassType],
            tuple([SavingArguments, training_pipeline_saving_arguments]),
        )
    )

    saving_args, training_pipeline_saving_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    filters = {
        key: saving_args.__dict__[key]
        for key in [
            "algorithm_type",
            "algorithm_application",
            "domain",
            "algorithm_name",
            "source_version",
        ]
    }
    configuration_tuples = get_configuration_tuples(
        filter_algorithm_applications(algorithms=AVAILABLE_ALGORITHMS, filters=filters)
    )
    if len(configuration_tuples) > 1:
        logger.info(
            f"Multiple configurations matching the parameters:{os.linesep}"
            f"{os.linesep.join(map(str, configuration_tuples))}{os.linesep}"
            f"Select one by specifying additional algorithms parameters: {','.join('--' + key for key, value in filters.items() if not value)}.",
        )
        return
    elif len(configuration_tuples) < 1:
        provided_filters = {key: value for key, value in filters.items() if value}
        logger.error(
            "No configurations matching the provided parameters, "
            f"please review the supported configurations:{os.linesep}"
            f"{os.linesep.join(map(str, configuration_tuples))}{os.linesep}"
            f"Please review the parameters provided:{os.linesep}"
            f"{provided_filters}"
        )
    configuration_tuple = configuration_tuples[0]
    logger.info(f"Selected configuration: {configuration_tuple}")

    algorithm_application = ApplicationsRegistry.applications[configuration_tuple]
    configuration_class = algorithm_application.configuration_class
    logger.info(
        f'Saving model version "{saving_args.target_version}" with the following configuration: {configuration_class}'
    )
    configuration_class.save_version_from_training_pipeline_arguments(
        training_pipeline_arguments=training_pipeline_saving_args,
        target_version=saving_args.target_version,
        source_version=saving_args.source_version,
    )


if __name__ == "__main__":
    main()
