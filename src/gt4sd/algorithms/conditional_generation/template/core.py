"""Template Algorithm"""

import logging
from dataclasses import field
from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, TypeVar

from ...core import AlgorithmConfiguration, GeneratorAlgorithm  # type: ignore
from ...registry import ApplicationsRegistry  # type: ignore
from .implementation import Generator  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = TypeVar("T")
S = TypeVar("S")
Targeted = Callable[[T], Iterable[Any]]


class Template(GeneratorAlgorithm[S, T]):
    """Template Algorithm."""

    def __init__(
        self, configuration: AlgorithmConfiguration[S, T], target: Optional[T] = None
    ):
        """Template Generation

        Args:
            configuration: domain and application
                specification, defining types and validations.
            target: Optional depending on the type of generative model. In this template
                we will convert the target to a string.

        Example:
            An example for using this template::

            target = 'World'
            configuration = TemplateGenerator()
            algorithm = Template(configuration=configuration, target=target)
            items = list(algorithm.sample(1))
            print(items)
        """

        configuration = self.validate_configuration(configuration)
        # TODO there might also be a validation/check on the target input

        super().__init__(
            configuration=configuration,
            target=target,  # type:ignore
        )

    def get_generator(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ) -> Targeted[T]:
        """Get the function to from generator.

        Args:
            configuration: helps to set up the application.
            target: context or condition for the generation. Just an optional string here.

        Returns:
            callable generating a list of 1 item containing salutation and temperature converted to fahrenheit.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: Generator = configuration.get_conditional_generator(  # type: ignore
            self.local_artifacts
        )
        return implementation.hello_name  # type:ignore

    def validate_configuration(
        self, configuration: AlgorithmConfiguration
    ) -> AlgorithmConfiguration:
        # TODO raise InvalidAlgorithmConfiguration
        assert isinstance(configuration, AlgorithmConfiguration)
        return configuration


@ApplicationsRegistry.register_algorithm_application(Template)
class TemplateGenerator(AlgorithmConfiguration[str, str]):
    """Configuration for specific generator."""

    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    temperature: int = field(
        default=36,
        metadata=dict(description="Temperature parameter ( in celsius )"),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.
        Returns:
            target description.
        """
        return {
            "title": "Target name",
            "description": "A simple string to define the name in the output [Hello name].",
            "type": "string",
        }

    def get_conditional_generator(self, resources_path: str) -> Generator:
        return Generator(resources_path=resources_path, temperature=self.temperature)
