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
import logging
from dataclasses import field
from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, TypeVar

from ....domains.materials import SMILES, validate_molecules
from ....exceptions import InvalidItem
from ...core import AlgorithmConfiguration, GeneratorAlgorithm
from ...registry import ApplicationsRegistry
from .implementation import ReinventConditionalGenerator

T = TypeVar("T", bound=Any)
S = TypeVar("S", bound=Any)
Targeted = Callable[[T], Iterable[Any]]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Reinvent(GeneratorAlgorithm[S, T]):
    """Reinvent sample generation algorithm."""

    def __init__(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ):
        """Instantiate Reinvent ready to generate samples.

        Args:
            configuration: domain and application
                specification defining parameters, types and validations.
            target: a target for which to generate items.

        Example:
            An example for predicting topics for a given text::

                config = ReinventGenerator()
                algorithm = Reinvent(configuration=config, target="CCO")
                items = list(algorithm.sample(1))
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
        """Get the function to perform the prediction via Reinvent's generator.

        Args:
            configuration: helps to set up specific application of Reinvent.
            target: context or condition for the generation.

        Returns:
            callable with target generating samples.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: ReinventConditionalGenerator = configuration.get_conditional_generator(  # type: ignore
            self.local_artifacts
        )
        return implementation.generate_samples


@ApplicationsRegistry.register_algorithm_application(Reinvent)
class ReinventGenerator(AlgorithmConfiguration[str, str]):
    """Configuration to generate molecules using the REINVENT algorithm. It generates the molecules minimizing the distances between the scaffolds."""

    algorithm_name: ClassVar[str] = Reinvent.__name__
    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    batch_size: int = field(
        default=20,
        metadata=dict(description=("Number of samples to generate per scaffold")),
    )

    randomize: bool = field(
        default=True,
        metadata=dict(description=("Randomize the scaffolds if set to true")),
    )

    sample_uniquely: bool = field(
        default=True,
        metadata=dict(description=("Generate unique sample sequences if set to true")),
    )
    max_sequence_length: int = field(
        default=256,
        metadata=dict(description=("Maximal length of SMILES sequences")),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.

        Returns:
            target description.
        """
        return {
            "title": "SMILES for sample generation",
            "description": "SMILES considered for the samples generation.",
            "type": "string",
        }

    def get_conditional_generator(
        self, resources_path: str
    ) -> ReinventConditionalGenerator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_samples<gt4sd.algorithms.conditional_generation.reinvent.implementation.ReinventConditionalGenerator.generate_samples>` method for targeted generation.
        """
        return ReinventConditionalGenerator(
            resources_path=resources_path,
            batch_size=self.batch_size,
            randomize=self.randomize,
            sample_uniquely=self.sample_uniquely,
            max_sequence_length=self.max_sequence_length,
        )

    def validate_item(self, item: str) -> SMILES:
        """Check that item is a valid SMILES.

        Args:
            item: a generated item that is possibly not valid.

        Raises:
            InvalidItem: in case the item can not be validated.

        Returns:
            the validated SMILES.
        """
        molecules, _ = validate_molecules(smiles_list=[item])

        if molecules[0] is None:
            raise InvalidItem(
                title="InvalidSMILES",
                detail=f'rdkit.Chem.MolFromSmiles returned None for "{item}"',
            )
        return SMILES(item)
