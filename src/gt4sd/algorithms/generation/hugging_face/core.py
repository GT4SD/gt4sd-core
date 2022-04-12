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
"""HuggingFace generation algorithm."""

import logging
from dataclasses import field
from typing import ClassVar, Dict, Optional, Set, TypeVar

from ...core import (
    AlgorithmConfiguration,
    GeneratorAlgorithm,
    Untargeted,
    get_configuration_class_with_attributes,
)
from ...registry import ApplicationsRegistry
from .implementation import MODEL_TYPES, Generator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = type(None)
S = TypeVar("S", bound=str)


class HuggingFaceGenerationAlgorithm(GeneratorAlgorithm[S, T]):
    def __init__(
        self, configuration: AlgorithmConfiguration, target: Optional[T] = None
    ):
        """HuggingFace generation algorithm.

        Args:
            configuration: domain and application
                specification, defining types and validations.
            target: unused since it is not a conditional generator.

        Example:
            An example for using a generative algorithm from HuggingFace::

                configuration = HuggingFaceXLMGenerator()
                algorithm = HuggingFaceGenerationAlgorithm(configuration=configuration)
                items = list(algorithm.sample(1))
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
        # TODO raise InvalidAlgorithmConfiguration
        assert isinstance(configuration, AlgorithmConfiguration)
        return configuration


@ApplicationsRegistry.register_algorithm_application(HuggingFaceGenerationAlgorithm)
class HuggingFaceConfiguration(AlgorithmConfiguration[str, None]):
    """Basic configuration for an hugging face algorithm."""

    algorithm_type: ClassVar[str] = "generation"
    domain: ClassVar[str] = "nlp"

    model_type: str = field(
        default="",
        metadata=dict(
            description=f"Type of the model. Supported: {', '.join(MODEL_TYPES.keys())}"
        ),
    )
    prompt: str = field(
        default="I'm a stochastic parrot.",
        metadata=dict(description="Prompt for text generation."),
    )
    length: int = field(
        default=20, metadata=dict(description="Length of the generated text.")
    )
    stop_token: str = field(
        default="", metadata=dict(description="Stop token for text generation.")
    )
    temperature: float = field(
        default=1.0,
        metadata=dict(
            description="Temperature for sampling, the lower the greedier the sampling."
        ),
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata=dict(
            description="Primarily useful for CTRL model, where 1.2 should be used."
        ),
    )
    k: int = field(
        default=50,
        metadata=dict(description="Number of top-k probability tokens to keep."),
    )
    p: float = field(
        default=1.0,
        metadata=dict(
            description="Only tokens with cumulative probabilities summing up to this value are kept."
        ),
    )
    prefix: str = field(
        default="",
        metadata=dict(
            description="Text defining context provided prior to the prompt."
        ),
    )
    number_of_sequences: int = field(
        default=8,
        metadata=dict(description="Number of text sequences to generate."),
    )

    def get_target_description(self) -> Optional[Dict[str, str]]:
        """Get description of the target for generation.

        Returns:
            target description, returns None in case no target is used.
        """
        return None

    def get_conditional_generator(self, resources_path: str, **kwargs) -> Generator:
        return Generator(
            resources_path=resources_path,
            model_type=self.model_type,
            model_name=self.algorithm_version,
            prompt=self.prompt,
            length=self.length,
            stop_token=self.stop_token,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            k=self.k,
            p=self.p,
            prefix=self.prefix,
            number_of_sequences=self.number_of_sequences,
        )


@ApplicationsRegistry.register_algorithm_application(HuggingFaceGenerationAlgorithm)
class HuggingFaceXLMGenerator(HuggingFaceConfiguration):
    """Configuration to generate text using XLM."""

    algorithm_version: str = "xlm-mlm-en-2048"
    model_type: str = "xlm"

    @classmethod
    def list_versions(cls) -> Set[str]:
        """Get possible algorithm versions.

        Standard S3 and cache search adding the version used in the configuration.

        Returns:
            viable values as :attr:`algorithm_version` for the environment.
        """
        logger.warning(
            "more algorithm versions can be found on https://huggingface.co/models"
        )
        return (
            get_configuration_class_with_attributes(cls)
            .list_versions()
            .union({cls.algorithm_version})
        )


@ApplicationsRegistry.register_algorithm_application(HuggingFaceGenerationAlgorithm)
class HuggingFaceCTRLGenerator(HuggingFaceConfiguration):
    """Configuration to generate text using CTRL."""

    algorithm_version: str = "ctrl"
    model_type: str = "ctrl"

    @classmethod
    def list_versions(cls) -> Set[str]:
        """Get possible algorithm versions.

        Standard S3 and cache search adding the version used in the configuration.

        Returns:
            viable values as :attr:`algorithm_version` for the environment.
        """
        logger.warning(
            "more algorithm versions can be found on https://huggingface.co/models"
        )
        return (
            get_configuration_class_with_attributes(cls)
            .list_versions()
            .union({cls.algorithm_version})
        )


@ApplicationsRegistry.register_algorithm_application(HuggingFaceGenerationAlgorithm)
class HuggingFaceGPT2Generator(HuggingFaceConfiguration):
    """Configuration to generate text using GPT2."""

    algorithm_version: str = "gpt2"
    model_type: str = "gpt2"

    @classmethod
    def list_versions(cls) -> Set[str]:
        """Get possible algorithm versions.

        Standard S3 and cache search adding the version used in the configuration.

        Returns:
            viable values as :attr:`algorithm_version` for the environment.
        """
        logger.warning(
            "more algorithm versions can be found on https://huggingface.co/models"
        )
        return (
            get_configuration_class_with_attributes(cls)
            .list_versions()
            .union({cls.algorithm_version})
        )


@ApplicationsRegistry.register_algorithm_application(HuggingFaceGenerationAlgorithm)
class HuggingFaceOpenAIGPTGenerator(HuggingFaceConfiguration):
    """Configuration to generate text using OpenAIGPT."""

    algorithm_version: str = "openai-gpt"
    model_type: str = "openai-gpt"

    @classmethod
    def list_versions(cls) -> Set[str]:
        """Get possible algorithm versions.

        Standard S3 and cache search adding the version used in the configuration.

        Returns:
            viable values as :attr:`algorithm_version` for the environment.
        """
        logger.warning(
            "more algorithm versions can be found on https://huggingface.co/models"
        )
        return (
            get_configuration_class_with_attributes(cls)
            .list_versions()
            .union({cls.algorithm_version})
        )


@ApplicationsRegistry.register_algorithm_application(HuggingFaceGenerationAlgorithm)
class HuggingFaceXLNetGenerator(HuggingFaceConfiguration):
    """Configuration to generate text using XLNet."""

    algorithm_version: str = "xlnet-large-cased"
    model_type: str = "xlnet"

    @classmethod
    def list_versions(cls) -> Set[str]:
        """Get possible algorithm versions.

        Standard S3 and cache search adding the version used in the configuration.

        Returns:
            viable values as :attr:`algorithm_version` for the environment.
        """
        logger.warning(
            "more algorithm versions can be found on https://huggingface.co/models"
        )
        return (
            get_configuration_class_with_attributes(cls)
            .list_versions()
            .union({cls.algorithm_version})
        )


@ApplicationsRegistry.register_algorithm_application(HuggingFaceGenerationAlgorithm)
class HuggingFaceTransfoXLGenerator(HuggingFaceConfiguration):
    """Configuration to generate text using TransfoXL."""

    algorithm_version: str = "transfo-xl-wt103"
    model_type: str = "transfo-xl"

    @classmethod
    def list_versions(cls) -> Set[str]:
        """Get possible algorithm versions.

        Standard S3 and cache search adding the version used in the configuration.

        Returns:
            viable values as :attr:`algorithm_version` for the environment.
        """
        logger.warning(
            "more algorithm versions can be found on https://huggingface.co/models"
        )
        return (
            get_configuration_class_with_attributes(cls)
            .list_versions()
            .union({cls.algorithm_version})
        )
