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
"""MoLeR Algorithm.

MoLeR generation algorithm.
"""

import logging
from dataclasses import field
from typing import Any, ClassVar, Dict, Optional, TypeVar

from ....domains.materials import SMILES, validate_molecules
from ....exceptions import InvalidItem
from ...core import AlgorithmConfiguration, GeneratorAlgorithm, Untargeted
from ...registry import ApplicationsRegistry
from .implementation import MoLeRGenerator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = type(None)
S = TypeVar("S", bound=SMILES)


class MoLeR(GeneratorAlgorithm[S, T]):
    """MoLeR Algorithm."""

    def __init__(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T] = None,
    ):
        """Instantiate MoLeR ready to generate items.

        Args:
            configuration: domain and application
                specification defining parameters, types and validations.
            target: a target for which to generate items.

        Example:
            An example for generating small molecules (SMILES) with the default configuration:

                configuration = MoLeRDefaultGenerator()
                MoLeR = MoLeR(configuration=configuration, target=target)
                items = list(MoLeR.sample(10))
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
    ) -> Untargeted:
        """Get the function to sample batches via the MoLeRGenerator.

        Args:
            configuration: helps to set up the application.
            target: context or condition for the generation. Unused in the algorithm.

        Returns:
            callable generating a batch of items.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: MoLeRGenerator = configuration.get_conditional_generator(  # type: ignore
            self.local_artifacts
        )
        return implementation.generate

    def validate_configuration(
        self, configuration: AlgorithmConfiguration[S, T]
    ) -> AlgorithmConfiguration[S, T]:
        # TODO raise InvalidAlgorithmConfiguration
        assert isinstance(configuration, AlgorithmConfiguration)
        return configuration


@ApplicationsRegistry.register_algorithm_application(MoLeR)
class MoLeRDefaultGenerator(AlgorithmConfiguration[SMILES, Any]):
    """Configuration to generate compounds using default parameters of MoLeR."""

    algorithm_type: ClassVar[str] = "generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    scaffolds: str = field(
        default="",
        metadata=dict(
            description="Scaffolds as '.'-separated SMILES. If empty, no scaffolds are used."
        ),
    )
    num_samples: int = field(
        default=32,
        metadata=dict(description="Number of molecules to sample per call."),
    )
    beam_size: int = field(
        default=1,
        metadata=dict(description="Beam size to use during decoding."),
    )
    seed: int = field(
        default=0,
        metadata=dict(description="Seed used for random number generation."),
    )
    num_workers: int = field(
        default=6,
        metadata=dict(description="Number of workers used for generation."),
    )

    def get_target_description(self) -> Optional[Dict[str, str]]:
        """Get description of the target for generation.

        Returns:
            target description, returns None in case no target is used.
        """
        return None

    def get_conditional_generator(self, resources_path: str) -> MoLeRGenerator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate<gt4sd.algorithms.generation.MoLeR.implementation.MoLeRGenerator.generate>` for generation.
        """
        return MoLeRGenerator(
            resources_path=resources_path,
            scaffolds=self.scaffolds,
            num_samples=self.num_samples,
            beam_size=self.beam_size,
            seed=self.seed,
            num_workers=self.num_workers,
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
