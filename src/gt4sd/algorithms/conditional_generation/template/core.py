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
