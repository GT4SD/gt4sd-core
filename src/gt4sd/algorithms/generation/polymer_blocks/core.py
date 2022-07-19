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
"""PaccMann vanilla generator trained on polymer building blocks (catalysts/monomers)."""

import logging
import os
from dataclasses import field
from typing import ClassVar, Dict, Optional, TypeVar

from ....domains.materials import SMILES, validate_molecules
from ....exceptions import InvalidItem
from ....training_pipelines.core import TrainingPipelineArguments
from ....training_pipelines.paccmann.core import PaccMannSavingArguments
from ...core import AlgorithmConfiguration, GeneratorAlgorithm, Untargeted
from ...registry import ApplicationsRegistry
from .implementation import Generator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = type(None)
S = TypeVar("S", bound=SMILES)


class PolymerBlocks(GeneratorAlgorithm[S, T]):
    def __init__(
        self, configuration: AlgorithmConfiguration, target: Optional[T] = None
    ):
        """Polymer blocks generation.

        Args:
            configuration: domain and application
                specification, defining types and validations.
            target: unused since it is not a conditional generator.

        Example:
            An example for generating small molecules (SMILES) that resembles
            monomers/catalysts for polymer synthesis::

                configuration = PolymerBlocksGenerator()
                polymer_blocks = PolymerBlocks(configuration=configuration)
                items = list(polymer_blocks.sample(10))
                print(items)
        """

        configuration = self.validate_configuration(configuration)
        # TODO there might also be a validation/check on the target input

        super().__init__(
            configuration=configuration,
            target=None,  # type:ignore
        )

    def get_generator(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ) -> Untargeted:
        """Get the function to sample batches via the Generator.

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
        # TODO raise InvalidAlgorithmConfiguration
        assert isinstance(configuration, AlgorithmConfiguration)
        return configuration


@ApplicationsRegistry.register_algorithm_application(PolymerBlocks)
class PolymerBlocksGenerator(AlgorithmConfiguration[SMILES, None]):
    """Configuration to generate subunits of polymers."""

    algorithm_type: ClassVar[str] = "generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    batch_size: int = field(
        default=32,
        metadata=dict(description="Batch size used for the generative model sampling."),
    )
    generated_length: int = field(
        default=100,
        metadata=dict(
            description="Maximum length in tokens of the generated molcules (relates to the SMILES length)."
        ),
    )

    def get_target_description(self) -> Optional[Dict[str, str]]:
        """Get description of the target for generation.

        Returns:
            target description, returns None in case no target is used.
        """
        return None

    def get_conditional_generator(self, resources_path: str) -> Generator:
        return Generator(
            resources_path=resources_path,
            generated_length=self.generated_length,
            batch_size=self.batch_size,
        )

    def validate_item(self, item: str) -> SMILES:
        (
            molecules,
            _,
        ) = validate_molecules([item])
        if molecules[0] is None:
            raise InvalidItem(
                title="InvalidSMILES",
                detail=f'rdkit.Chem.MolFromSmiles returned None for "{item}"',
            )
        return SMILES(item)

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
        if isinstance(training_pipeline_arguments, PaccMannSavingArguments):
            return {
                "smiles_language.pkl": os.path.join(
                    training_pipeline_arguments.model_path,
                    f"{training_pipeline_arguments.training_name}.lang",
                ),
                "params.json": os.path.join(
                    training_pipeline_arguments.model_path,
                    training_pipeline_arguments.training_name,
                    "model_params.json",
                ),
                "weights.pt": os.path.join(
                    training_pipeline_arguments.model_path,
                    training_pipeline_arguments.training_name,
                    "weights",
                    "best_rec.pt",
                ),
            }
        else:
            return super().get_filepath_mappings_for_training_pipeline_arguments(
                training_pipeline_arguments
            )
