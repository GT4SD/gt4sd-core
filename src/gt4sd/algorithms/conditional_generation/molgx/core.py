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
"""MolGX Algorithm.

MolGX generation algorithm.
"""

import logging
from dataclasses import field
from typing import Any, ClassVar, Dict, Iterator, Optional, TypeVar

from ....domains.materials import SMILES, validate_molecules
from ....exceptions import InvalidItem
from ...core import AlgorithmConfiguration, GeneratorAlgorithm, Untargeted
from ...registry import ApplicationsRegistry
from .implementation import MolGXGenerator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = type(None)
S = TypeVar("S", bound=SMILES)


class MolGX(GeneratorAlgorithm[S, T]):
    """MolGX Algorithm."""

    def __init__(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T] = None,
    ):
        """Instantiate MolGX ready to generate items.

        Args:
            configuration: domain and application
                specification defining parameters, types and validations.
            target: a target for which to generate items.

        Example:
            An example for generating small molecules (SMILES) with given HOMO and LUMO energies:

                configuration = MolGXQM9Generator()
                molgx = MolGX(configuration=configuration, target=target)
                items = list(molgx.sample(10))
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
        """Get the function to sample batches via the ConditionalGenerator.

        Args:
                configuration: helps to set up the application.
                target: context or condition for the generation. Unused in the algorithm.

        Returns:
                callable generating a batch of items.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: MolGXGenerator = configuration.get_conditional_generator(  # type: ignore
            self.local_artifacts
        )
        return implementation.generate

    def validate_configuration(
        self, configuration: AlgorithmConfiguration[S, T]
    ) -> AlgorithmConfiguration[S, T]:
        # TODO raise InvalidAlgorithmConfiguration
        assert isinstance(configuration, AlgorithmConfiguration)
        return configuration

    def sample(self, number_of_items: int = 100) -> Iterator[S]:
        """Generate a number of unique and valid items.

        Args:
            number_of_items: number of items to generate. Defaults to 100.

        Yields:
            the items.
        """
        if hasattr(self.configuration, "maximum_number_of_solutions"):
            maxiumum_number_of_molecules = int(
                getattr(self.configuration, "maximum_number_of_solutions")
            )
            if number_of_items > maxiumum_number_of_molecules:
                logger.warning(
                    f"current MolGX configuration can not support generation of {number_of_items} molecules..."
                )
                logger.warning(
                    f"to enable generation of {number_of_items} molecules, increase 'maximum_number_of_solutions' (currently set to {maxiumum_number_of_molecules})"
                )
                number_of_items = maxiumum_number_of_molecules
                logger.warning(f"generating at most: {maxiumum_number_of_molecules}...")

        return super().sample(number_of_items=number_of_items)


@ApplicationsRegistry.register_algorithm_application(MolGX)
class MolGXQM9Generator(AlgorithmConfiguration[SMILES, Any]):
    """Configuration to generate compounds with given HOMO and LUMO energies."""

    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"
    homo_energy_value: float = field(
        default=-0.25,
        metadata=dict(description="Target HOMO energy value."),
    )
    lumo_energy_value: float = field(
        default=0.08,
        metadata=dict(description="Target LUMO energy value."),
    )
    use_linear_model: bool = field(
        default=True,
        metadata=dict(description="Linear model usage."),
    )
    number_of_candidates: int = field(
        default=2,
        metadata=dict(description="Number of candidates to consider."),
    )
    maximum_number_of_candidates: int = field(
        default=5,
        metadata=dict(description="Maximum number of candidates to consider."),
    )
    maximum_number_of_solutions: int = field(
        default=10,
        metadata=dict(description="Maximum number of solutions."),
    )
    maximum_number_of_nodes: int = field(
        default=50000,
        metadata=dict(description="Maximum number of nodes in the graph exploration."),
    )
    beam_size: int = field(
        default=2000,
        metadata=dict(description="Size of the beam during search."),
    )
    without_estimate: bool = field(
        default=True,
        metadata=dict(description="Disable estimates."),
    )
    use_specific_rings: bool = field(
        default=True,
        metadata=dict(description="Flag to indicate whether specific rings are used."),
    )
    use_fragment_const: bool = field(
        default=False,
        metadata=dict(description="Using constant fragments."),
    )

    def get_target_description(self) -> Optional[Dict[str, str]]:
        """Get description of the target for generation.

        Returns:
                target description, returns None in case no target is used.
        """
        return None

    def get_conditional_generator(self, resources_path: str) -> MolGXGenerator:
        """Instantiate the actual generator implementation.

        Args:
                resources_path: local path to model files.

        Returns:
                instance with :meth:`generate<gt4sd.algorithms.conditional_generation.molgx.implementation.MolGXGenerator.generate>` for generation.
        """
        return MolGXGenerator(
            resources_path=resources_path,
            homo_energy_value=self.homo_energy_value,
            lumo_energy_value=self.lumo_energy_value,
            use_linear_model=self.use_linear_model,
            number_of_candidates=self.number_of_candidates,
            maximum_number_of_candidates=self.maximum_number_of_candidates,
            maximum_number_of_solutions=self.maximum_number_of_solutions,
            maximum_number_of_nodes=self.maximum_number_of_nodes,
            beam_size=self.beam_size,
            without_estimate=self.without_estimate,
            use_specific_rings=self.use_specific_rings,
            use_fragment_const=self.use_fragment_const,
            tag_name="qm9_sample_pretrained_model.pickle",
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
