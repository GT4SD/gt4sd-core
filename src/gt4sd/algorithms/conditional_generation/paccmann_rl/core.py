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
"""PaccMann\\ :superscript:`RL` Algorithm.

PaccMann\\ :superscript:`RL` generation is conditioned via reinforcement learning.
"""

import logging
from dataclasses import field
from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, TypeVar

from typing_extensions import Protocol, runtime_checkable

from ....domains.materials import SMILES, Omics, Protein
from ....exceptions import InvalidItem
from ...core import AlgorithmConfiguration, GeneratorAlgorithm
from ...registry import ApplicationsRegistry
from .implementation import (
    ConditionalGenerator,
    ProteinSequenceConditionalGenerator,
    TranscriptomicConditionalGenerator,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = TypeVar("T", Protein, Omics)
S = TypeVar("S", bound=SMILES)
Targeted = Callable[[T], Iterable[Any]]


class PaccMannRL(GeneratorAlgorithm[S, T]):
    """PaccMann\\ :superscript:`RL` Algorithm."""

    def __init__(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ):
        """Instantiate PaccMannRL ready to generate items.

        Args:
            configuration: domain and application
                specification defining parameters, types and validations.
            target: a target for which to generate items.

        Example:
            An example for generating small molecules (SMILES) with high affinity
            for a target protein::

                affinity_config = PaccMannRLProteinBasedGenerator()
                target = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"
                paccmann_affinity = PaccMannRL(configuration=affinity_config, target=target)
                items = list(paccmann_affinity.sample(10))
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
        """Get the function to sample batches via PaccMannRL's ConditionalGenerator.

        Args:
            configuration: helps to set up specific application of PaccMannRL.
            target: context or condition for the generation.

        Returns:
            callable with target generating a batch of items.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: ConditionalGenerator = configuration.get_conditional_generator(  # type: ignore
            self.local_artifacts
        )
        return implementation.generate_batch

    def validate_configuration(
        self, configuration: AlgorithmConfiguration[S, T]
    ) -> AlgorithmConfiguration[S, T]:
        @runtime_checkable
        class AnyPaccMannRLConfiguration(Protocol):
            """Protocol for PaccMannRL configurations."""

            def get_conditional_generator(
                self, resources_path: str
            ) -> ConditionalGenerator:
                ...

            def validate_item(self, item: Any) -> S:
                ...

        # TODO raise InvalidAlgorithmConfiguration
        assert isinstance(configuration, AnyPaccMannRLConfiguration)
        assert isinstance(configuration, AlgorithmConfiguration)
        return configuration


@ApplicationsRegistry.register_algorithm_application(PaccMannRL)
class PaccMannRLProteinBasedGenerator(AlgorithmConfiguration[SMILES, Protein]):
    """
    Configuration to generate compounds with high affinity to a target protein.

    Implementation from the paper: https://doi.org/10.1088/2632-2153/abe808.
    """

    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    batch_size: int = field(
        default=32,
        metadata=dict(description="Batch size used for the generative model sampling."),
    )
    temperature: float = field(
        default=1.4,
        metadata=dict(
            description="Temperature parameter for the softmax sampling in decoding."
        ),
    )
    generated_length: int = field(
        default=100,
        metadata=dict(
            description="Maximum length in tokens of the generated molcules (relates to the SMILES length)."
        ),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.

        Returns:
            target description.
        """
        return {
            "title": "Target protein sequence",
            "description": "AA sequence of the protein target to generate non-toxic ligands against.",
            "type": "string",
        }

    def get_conditional_generator(
        self, resources_path: str
    ) -> ProteinSequenceConditionalGenerator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.paccmann_rl.implementation.ConditionalGenerator.generate_batch>` method for targeted generation.
        """
        return ProteinSequenceConditionalGenerator(
            resources_path=resources_path,
            temperature=self.temperature,
            generated_length=self.generated_length,
            samples_per_protein=self.batch_size,
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
        ) = ProteinSequenceConditionalGenerator.validate_molecules([item])
        if molecules[0] is None:
            raise InvalidItem(
                title="InvalidSMILES",
                detail=f'rdkit.Chem.MolFromSmiles returned None for "{item}"',
            )
        return SMILES(item)


@ApplicationsRegistry.register_algorithm_application(PaccMannRL)
class PaccMannRLOmicBasedGenerator(AlgorithmConfiguration[SMILES, Omics]):
    """
    Configuration to generate compounds with low IC50 for a target omics profile.

    Implementation from the paper: https://doi.org/10.1016/j.isci.2021.102269.
    """

    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    batch_size: int = field(
        default=32,
        metadata=dict(description="Batch size used for the generative model sampling."),
    )
    temperature: float = field(
        default=1.4,
        metadata=dict(
            description="Temperature parameter for the softmax sampling in decoding."
        ),
    )
    generated_length: int = field(
        default=100,
        metadata=dict(
            description="Maximum length in tokens of the generated molcules (relates to the SMILES length)."
        ),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.

        Returns:
            target description.
        """
        return {
            "title": "Gene expression profile",
            "description": "A gene expression profile to generate effective molecules against.",
            "type": "list",
        }

    def get_conditional_generator(
        self, resources_path: str
    ) -> TranscriptomicConditionalGenerator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.paccmann_rl.implementation.ConditionalGenerator.generate_batch>` method for targeted generation.
        """
        return TranscriptomicConditionalGenerator(
            resources_path=resources_path,
            temperature=self.temperature,
            generated_length=self.generated_length,
            samples_per_profile=self.batch_size,
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
        ) = TranscriptomicConditionalGenerator.validate_molecules([item])
        if molecules[0] is None:
            raise InvalidItem(
                title="InvalidSMILES",
                detail=f'rdkit.Chem.MolFromSmiles returned None for "{item}"',
            )
        return SMILES(item)
