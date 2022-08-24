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
"""HuggingFace Diffusers generation algorithm."""

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
from .implementation import MODEL_TYPES, SCHEDULER_TYPES, Generator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = type(None)
S = TypeVar("S", bound=str)


class DiffusersGenerationAlgorithm(GeneratorAlgorithm[S, T]):
    def __init__(
        self, configuration: AlgorithmConfiguration, target: Optional[T] = None
    ):
        """Diffusers generation algorithm.

        Args:
            configuration: domain and application
                specification, defining types and validations.
            target: unused since it is not a conditional generator.

        Example:
            An example for using a generative algorithm from Diffusers::

                configuration = GeneratorConfiguration()
                algorithm = DiffusersGenerationAlgorithm(configuration=configuration)
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
        assert isinstance(configuration, AlgorithmConfiguration)
        return configuration


@ApplicationsRegistry.register_algorithm_application(DiffusersGenerationAlgorithm)
class DiffusersConfiguration(AlgorithmConfiguration[str, None]):
    """Basic configuration for a diffusion algorithm."""

    algorithm_type: ClassVar[str] = "generation"
    domain: ClassVar[str] = "image"

    model_type: str = field(
        default="diffusion",
        metadata=dict(
            description=f"Type of the model. Supported: {', '.join(MODEL_TYPES.keys())}"
        ),
    )

    scheduler_type: str = field(
        default="discrete",
        metadata=dict(
            description=f"Type of the noise scheduler. Supported: {', '.join(SCHEDULER_TYPES.keys())}"
        ),
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
            scheduler_type=self.scheduler_type,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            k=self.k,
            p=self.p,
            prefix=self.prefix,
            number_of_sequences=self.number_of_sequences,
        )


@ApplicationsRegistry.register_algorithm_application(DiffusersGenerationAlgorithm)
class DDPMGenerator(DiffusersConfiguration):
    """DDPM - Configuration to generate using unconditional denoising diffusion models."""

    algorithm_version: str = "google/ddpm-celebahq-256"
    model_type: str = "ddpm"
    scheduler_type: str = "ddpm"

    @classmethod
    def list_versions(cls) -> Set[str]:
        """Get possible algorithm versions.

        Standard S3 and cache search adding the version used in the configuration.

        Returns:
            viable values as :attr:`algorithm_version` for the environment.
        """
        logger.warning(
            "more algorithm versions can be found on https://github.com/huggingface/diffusers"
        )
        return (
            get_configuration_class_with_attributes(cls)
            .list_versions()
            .union({cls.algorithm_version})
        )


@ApplicationsRegistry.register_algorithm_application(DiffusersGenerationAlgorithm)
class DDIMGenerator(DiffusersConfiguration):
    """DDIM - Configuration to generate using a denoising diffusion implicit model."""

    algorithm_version: str = "google/ddim-celebahq-256"
    model_type: str = "ddim"
    scheduler_type: str = "ddim"

    @classmethod
    def list_versions(cls) -> Set[str]:
        """Get possible algorithm versions.

        Standard S3 and cache search adding the version used in the configuration.

        Returns:
            viable values as :attr:`algorithm_version` for the environment.
        """
        logger.warning(
            "more algorithm versions can be found on https://github.com/huggingface/diffusers"
        )
        return (
            get_configuration_class_with_attributes(cls)
            .list_versions()
            .union({cls.algorithm_version})
        )


@ApplicationsRegistry.register_algorithm_application(DiffusersGenerationAlgorithm)
class LDMGenerator(DiffusersConfiguration):
    """Unconditional Latent Diffusion Model - Configuration to generate using a latent diffusion model."""

    algorithm_version: str = ""
    model_type: str = "latent_diffusion"
    scheduler_type: str = "discrete"

    @classmethod
    def list_versions(cls) -> Set[str]:
        """Get possible algorithm versions.

        Standard S3 and cache search adding the version used in the configuration.

        Returns:
            viable values as :attr:`algorithm_version` for the environment.
        """
        logger.warning(
            "more algorithm versions can be found on https://github.com/huggingface/diffusers"
        )
        return (
            get_configuration_class_with_attributes(cls)
            .list_versions()
            .union({cls.algorithm_version})
        )


@ApplicationsRegistry.register_algorithm_application(DiffusersGenerationAlgorithm)
class ScoreSdeGenerator(DiffusersConfiguration):
    """Score SDE Generative Model - Configuration to generate using a score-based diffusion generative model."""

    algorithm_version: str = ""
    model_type: str = "score_sde"
    scheduler_type: str = "continuous"

    @classmethod
    def list_versions(cls) -> Set[str]:
        """Get possible algorithm versions.

        Standard S3 and cache search adding the version used in the configuration.

        Returns:
            viable values as :attr:`algorithm_version` for the environment.
        """
        logger.warning(
            "more algorithm versions can be found on https://github.com/huggingface/diffusers"
        )
        return (
            get_configuration_class_with_attributes(cls)
            .list_versions()
            .union({cls.algorithm_version})
        )


@ApplicationsRegistry.register_algorithm_application(DiffusersGenerationAlgorithm)
class LDMTextToImageGenerator(DiffusersConfiguration):
    """Conditional Latent Diffusion Model - Configuration for conditional text2image generation using a latent diffusion model."""

    algorithm_version: str = "CompVis/ldm-text2im-large-256"
    model_type: str = "latent_diffusion_conditional"
    scheduler_type: str = "discrete"

    @classmethod
    def list_versions(cls) -> Set[str]:
        """Get possible algorithm versions.

        Standard S3 and cache search adding the version used in the configuration.

        Returns:
            viable values as :attr:`algorithm_version` for the environment.
        """
        logger.warning(
            "more algorithm versions can be found on https://github.com/huggingface/diffusers"
        )
        return (
            get_configuration_class_with_attributes(cls)
            .list_versions()
            .union({cls.algorithm_version})
        )


@ApplicationsRegistry.register_algorithm_application(DiffusersGenerationAlgorithm)
class StableDiffusionGenerator(DiffusersConfiguration):
    """Stable Diffusion Model - Configuration for conditional text2image generation using a stable diffusion model."""

    algorithm_version: str = "CompVis/stable-diffusion-v1-3"
    model_type: str = "stable_diffusion"
    scheduler_type: str = "discrete"

    @classmethod
    def list_versions(cls) -> Set[str]:
        """Get possible algorithm versions.

        Standard S3 and cache search adding the version used in the configuration.

        Returns:
            viable values as :attr:`algorithm_version` for the environment.
        """
        logger.warning(
            "more algorithm versions can be found on https://github.com/huggingface/diffusers"
        )
        return (
            get_configuration_class_with_attributes(cls)
            .list_versions()
            .union({cls.algorithm_version})
        )
