import logging
from dataclasses import field
from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, TypeVar

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
                algorithm = Reinvent(configuration=config, target="")
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
        )
