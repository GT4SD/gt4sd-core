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
"""Torchdrug generation algorithm."""

import logging
import os
from typing import ClassVar, Dict, Optional, TypeVar

from ....training_pipelines.core import TrainingPipelineArguments
from ....training_pipelines.torchdrug.core import TorchDrugSavingArguments
from ...core import AlgorithmConfiguration, GeneratorAlgorithm, Untargeted
from ...registry import ApplicationsRegistry
from .implementation import GAFGenerator, GCPNGenerator, Generator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = type(None)
S = TypeVar("S", bound=str)


class TorchDrugGenerator(GeneratorAlgorithm[S, T]):
    def __init__(
        self, configuration: AlgorithmConfiguration, target: Optional[T] = None
    ):
        """TorchDrug generation algorithm.

        Args:
            configuration: domain and application specification, defining types
                and validations.  Currently supported algorithm versions are:
                "zinc250k_v0", "qed_v0" and "plogp_v0".
            target: unused since it is not a conditional generator.

        Example:
            An example for using a generative algorithm from TorchDrug:

                configuration = TorchDrugGCPN(algorithm_version="qed_v0")
                algorithm = TorchDrugGenerator(configuration=configuration)
                items = list(algorithm.sample(1))
                print(items)
        """

        configuration = self.validate_configuration(configuration)
        super().__init__(
            configuration=configuration,
            target=None,  # type:ignore
        )

    def get_generator(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ) -> Untargeted:
        """Get the function to sample batches.

        Args:
            configuration: helps to set up the application.
            target: context or condition for the generation. Unused in the algorithm.

        Returns:
            callable generating a batch of items.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: Generator = configuration.get_conditional_generator(  # type: ignore
            self.local_artifacts
        )
        return implementation.sample

    def validate_configuration(
        self, configuration: AlgorithmConfiguration
    ) -> AlgorithmConfiguration:
        assert isinstance(configuration, AlgorithmConfiguration)
        return configuration


@ApplicationsRegistry.register_algorithm_application(TorchDrugGenerator)
class TorchDrugGCPN(AlgorithmConfiguration[str, None]):
    """
    Interface for TorchDrug Graph-convolutional policy network (GCPN) algorithm.
    Currently supported algorithm versions are "zinc250k_v0", "qed_v0" and "plogp_v0".
    """

    algorithm_type: ClassVar[str] = "generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "zinc250k_v0"

    def get_conditional_generator(self, resources_path: str) -> GCPNGenerator:
        """Instantiate the actual generator implementation.
        Args:
            resources_path: local path to model files.
        Returns:
            instance with :meth:`sample<gt4sd.algorithms.generation.torchdrug.implementation.GCPNGenerator.sample>` method for generation.
        """
        self.generator = GCPNGenerator(resources_path=resources_path)
        return self.generator

    @classmethod
    def get_filepath_mappings_for_training_pipeline_arguments(
        cls, training_pipeline_arguments: TrainingPipelineArguments
    ) -> Dict[str, str]:
        """Get filepath mappings for the given training pipeline arguments.
        Args:
            training_pipeline_arguments: training pipeline arguments.
        Returns:
            a mapping between artifacts' files and training pipeline's output files.
        """
        if isinstance(training_pipeline_arguments, TorchDrugSavingArguments):

            task_name = (
                f"task={training_pipeline_arguments.task}_"
                if training_pipeline_arguments.task
                else ""
            )
            data_name = "data=" + (
                training_pipeline_arguments.dataset_name
                + "_"
                + training_pipeline_arguments.file_path.split(os.sep)[-1].split(".")[0]
                if training_pipeline_arguments.dataset_name == "custom"
                else training_pipeline_arguments.dataset_name
            )

            epochs = training_pipeline_arguments.epochs
            return {
                "weights.pkl": os.path.join(
                    training_pipeline_arguments.model_path,
                    training_pipeline_arguments.training_name,
                    f"gcpn_data={data_name}_{task_name}epoch={epochs}.pkl",
                )
            }
        else:
            return super().get_filepath_mappings_for_training_pipeline_arguments(
                training_pipeline_arguments
            )


@ApplicationsRegistry.register_algorithm_application(TorchDrugGenerator)
class TorchDrugGraphAF(AlgorithmConfiguration[str, None]):
    """
    Interface for TorchDrug flow-based autoregressive graph algorithm (GraphAF).
    Currently supported algorithm versions are "zinc250k_v0", "qed_v0" and "plogp_v0".
    """

    algorithm_type: ClassVar[str] = "generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "zinc250k_v0"

    def get_conditional_generator(self, resources_path: str) -> GAFGenerator:
        """Instantiate the actual generator implementation.
        Args:
            resources_path: local path to model files.
        Returns:
            instance with :meth:`samples<gt4sd.algorithms.generation.torchdrug.implementation.GAFGenerator.sample>` method for generation.
        """
        self.generator = GAFGenerator(resources_path=resources_path)
        return self.generator

    @classmethod
    def get_filepath_mappings_for_training_pipeline_arguments(
        cls, training_pipeline_arguments: TrainingPipelineArguments
    ) -> Dict[str, str]:
        """Get filepath mappings for the given training pipeline arguments.
        Args:
            training_pipeline_arguments: training pipeline arguments.
        Returns:
            a mapping between artifacts' files and training pipeline's output files.
        """
        if isinstance(training_pipeline_arguments, TorchDrugSavingArguments):

            task_name = (
                f"task={training_pipeline_arguments.task}_"
                if training_pipeline_arguments.task
                else ""
            )
            data_name = "data=" + (
                training_pipeline_arguments.dataset_name
                + "_"
                + training_pipeline_arguments.file_path.split(os.sep)[-1].split(".")[0]
                if training_pipeline_arguments.dataset_name == "custom"
                else training_pipeline_arguments.dataset_name
            )

            epochs = training_pipeline_arguments.epochs
            return {
                "weights.pkl": os.path.join(
                    training_pipeline_arguments.model_path,
                    training_pipeline_arguments.training_name,
                    f"graphaf_data={data_name}_{task_name}epoch={epochs}.pkl",
                )
            }
        else:
            return super().get_filepath_mappings_for_training_pipeline_arguments(
                training_pipeline_arguments
            )
