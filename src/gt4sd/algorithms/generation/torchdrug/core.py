"""Torchdrug generation algorithm."""

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
from .implementation import Generator, GCPNGenerator, GAFGenerator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = type(None)
S = TypeVar("S", bound=str)


class TorchDrugGCPN(GeneratorAlgorithm[S, T]):
    def __init__(
        self, configuration: AlgorithmConfiguration, target: Optional[T] = None
    ):
        """TorchDrug generation algorithm using a GCPN model.

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


@ApplicationsRegistry.register_algorithm_application(TorchDrugGCPN)
class TorchDrugGCPNConfiguration(AlgorithmConfiguration[str, None]):
    pass


@ApplicationsRegistry.register_algorithm_application(TorchDrugGCPN)
class TorchDrugZincGCPN(TorchDrugGCPNConfiguration):
    pass


@ApplicationsRegistry.register_algorithm_application(TorchDrugGCPN)
class TorchDrugQedGCPN(TorchDrugGCPNConfiguration):
    pass


@ApplicationsRegistry.register_algorithm_application(TorchDrugGCPN)
class TorchDrugPlogpGCPN(TorchDrugGCPNConfiguration):
    pass


# GraphAF models
class TorchDrugGAF(GeneratorAlgorithm[S, T]):
    def __init__(
        self, configuration: AlgorithmConfiguration, target: Optional[T] = None
    ):
        """TorchDrug generation algorithm using a GCPN model.

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
        pass


@ApplicationsRegistry.register_algorithm_application(TorchDrugGAF)
class TorchDrugGAFConfiguration(AlgorithmConfiguration[str, None]):
    pass


@ApplicationsRegistry.register_algorithm_application(TorchDrugGAF)
class TorchDrugZincGCPN(TorchDrugGAFConfiguration):
    pass


@ApplicationsRegistry.register_algorithm_application(TorchDrugGAF)
class TorchDrugQedGCPN(TorchDrugGAFConfiguration):
    pass


@ApplicationsRegistry.register_algorithm_application(TorchDrugGAF)
class TorchDrugPlogpGCPN(TorchDrugGAFConfiguration):
    pass
