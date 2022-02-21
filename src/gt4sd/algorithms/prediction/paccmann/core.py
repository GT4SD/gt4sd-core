"""Prediction algorithms based on Paccmann"""

import logging
from dataclasses import field
from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, TypeVar

from ...core import AlgorithmConfiguration, GeneratorAlgorithm
from ...registry import ApplicationsRegistry
from .implementation import BimodalMCAAffinityPredictor

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = TypeVar("T", bound=Any)
S = TypeVar("S", bound=Any)
Targeted = Callable[[T], Iterable[Any]]

class Paccmann(GeneratorAlgorithm[S, T]):
    """
    Paccmann based prediction
    currently the only supported prediction is binding affinity between a ligand and a target
    """

    def __init__(
        self,
        configuration: AlgorithmConfiguration[S, T],
    ):
        """Instantiate BimodalAffinityPredictor ready.

        Args:
            configuration: domain and application
                specification defining parameters, types and validations.
            target: a target for which to generate items.

        Example:
            An example for predicting affinity for a given ligand+target::

                TODO: add this
        """

        configuration = self.validate_configuration(configuration)
        # TODO there might also be a validation/check on the target input

        super().__init__(
            configuration=configuration,  # type:ignore
            target='dummy',
        )

    def get_generator(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target='dummy',
    ) -> Targeted[T]:
        """Get the function to perform the prediction via TopicsZeroShot's generator.

        Args:
            configuration: helps to set up specific application of TopicsZeroShot.
            target: dummy 

        Returns:
            callable with target predicting topics sorted by relevance.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: BimodalMCAAffinityPredictor = configuration.get_conditional_generator(  # type: ignore
            self.local_artifacts
        )
        return implementation.predict

    


@ApplicationsRegistry.register_algorithm_application(Paccmann)
class BimodalMCAAffinityPredictorConfiguation(AlgorithmConfiguration[str, str]):
    """Configuration to predict affinity."""

    algorithm_type: ClassVar[str] = "prediction"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"
    

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.

        Returns:
            target description.
        """
        return {
            "title": "ligand and target",
            "description": "input ligand and target will be used for affinity prediction",
            "type": "obj",
        }

    def get_conditional_generator(self, resources_path: str) -> BimodalMCAAffinityPredictor:
        """Instantiate the actual predictor implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`gt4sd.algorithms.prediction.affinity._predicto.implementation.BimodalMCAAffinityPredictor.predict` method for predicting affinity.
        """
        return BimodalMCAAffinityPredictor(
            resources_path=resources_path
        )
