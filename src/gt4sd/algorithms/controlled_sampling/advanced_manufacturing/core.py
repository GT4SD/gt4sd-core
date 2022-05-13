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
"""Advanced manufacturing algorithms."""

import logging
from dataclasses import field
from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, TypeVar

from ....domains.materials import SmallMolecule
from ....training_pipelines.core import TrainingPipelineArguments
from ....training_pipelines.pytorch_lightning.granular.core import (
    GranularSavingArguments,
)
from ...core import AlgorithmConfiguration, GeneratorAlgorithm
from ...registry import ApplicationsRegistry
from .implementation.core import Generator
from .implementation.nccr import CatalystGenerator as NCCRCatalystGenerator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = TypeVar("T", bound=Any)
S = TypeVar("S", bound=Any)
Targeted = Callable[[T], Iterable[Any]]


class AdvancedManufacturing(GeneratorAlgorithm[S, T]):
    """Advance manufacturing generator algorithm."""

    def __init__(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ):
        """Instantiate AdvancedManufacturing ready to generate items.

        Args:
            configuration: domain and application
                specification defining parameters, types and validations.
            target: a target for which to generate items.

        Example:
            An example for generating small molecules (SMILES) with a target binding energy::

                config = CatalystGenerator()
                algorithm = AdvancedManufacturing(configuration=config, target=10.0)
                items = list(algorithm.sample(10))
                print(items)
        """

        configuration = self.validate_configuration(configuration)
        # TODO there might also be a validation/check on the target input

        super().__init__(
            configuration=configuration,  # type:ignore
            target=target,  # type:ignore
        )

    def get_generator(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ) -> Targeted[T]:
        """Get the function to sample batches via AdvancedManufacturing's generator.

        Args:
            configuration: helps to set up specific application of AdvancedManufacturing.
            target: context or condition for the generation.

        Returns:
            callable with target generating a batch of items.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: Generator = configuration.get_conditional_generator(  # type: ignore
            self.local_artifacts
        )
        return implementation.generate_samples


@ApplicationsRegistry.register_algorithm_application(AdvancedManufacturing)
class CatalystGenerator(AlgorithmConfiguration[SmallMolecule, float]):
    """Configuration to generate catalysts with a desired binding energy."""

    algorithm_type: ClassVar[str] = "controlled_sampling"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    number_of_points: int = field(
        default=32,
        metadata=dict(
            description="Number of points to sample with the Gaussian Process."
        ),
    )
    number_of_steps: int = field(
        default=50,
        metadata=dict(
            description="Number of optimization steps in the Gaussian Process optimization."
        ),
    )
    generated_length: int = field(
        default=100,
        metadata=dict(
            description="Maximum length in tokens of the generated molcules (relates to the SMILES length)."
        ),
    )
    primer_smiles: str = field(
        default="",
        metadata=dict(
            description="Primer molecule to initiate the sampling in SMILES format. Defaults to no primer."
        ),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.

        Returns:
            target description.
        """
        return {
            "title": "Target energy",
            "description": "Binding energy target for the catalysts generated.",
            "type": "number",
        }

    def get_conditional_generator(self, resources_path: str) -> Generator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.controlled_sampling.advanced_manufacturing.implementation.core.Generator.generate_samples>` method for targeted generation.
        """
        return NCCRCatalystGenerator(
            resources_path=resources_path,
            generated_length=self.generated_length,
            number_of_points=self.number_of_points,
            number_of_steps=self.number_of_steps,
            primer_smiles=self.primer_smiles,
        )

    @classmethod
    def get_filepath_mappings_for_training_pipeline_arguments(
        cls, training_pipeline_arguments: TrainingPipelineArguments
    ) -> Dict[str, str]:
        """Ger filepath mappings for the given training pipeline arguments.

        Args:
            training_pipeline_arguments: training pipeline arguments.

        Returns:
            a mapping between artifacts' files and training pipeline's output files.
        """
        if isinstance(training_pipeline_arguments, GranularSavingArguments):
            return {
                "epoch=199-step=5799.ckpt": training_pipeline_arguments.model_path,
            }
        else:
            return super().get_filepath_mappings_for_training_pipeline_arguments(
                training_pipeline_arguments
            )
