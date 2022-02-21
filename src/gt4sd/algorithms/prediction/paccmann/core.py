"""Prediction algorithms based on PaccMann"""

import logging
from dataclasses import field
from typing import Any, ClassVar, List, Optional, TypeVar

from ...core import AlgorithmConfiguration, GeneratorAlgorithm, Untargeted
from ...registry import ApplicationsRegistry
from .implementation import BimodalMCAAffinityPredictor, MCAPredictor

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = TypeVar("T", bound=Any)
S = TypeVar("S", bound=Any)


class PaccMann(GeneratorAlgorithm[S, T]):
    """PaccMann predictor."""

    def __init__(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T] = None,
    ):
        """Instantiate PaccMann for prediction.

        Args:
            configuration: domain and application
                specification defining parameters, types and validations.
            target: a target for which to generate items.

        Example:
            An example for predicting affinity for a given ligand and target protein pair::

                config = AffinityPredictor()
                algorithm = TopicsZeroShot(configuration=config, target="This is a text I want to understand better")
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
    ) -> Untargeted:
        """Get the function to perform the prediction via PaccMann's generator.

        Args:
            configuration: helps to set up specific application of PaccMann.
            target: context or condition for the generation.

        Returns:
            callable with target predicting properties using PaccMann.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: MCAPredictor = configuration.get_conditional_generator(  # type: ignore
            self.local_artifacts
        )
        return implementation.predict_values


@ApplicationsRegistry.register_algorithm_application(PaccMann)
class AffinityPredictor(AlgorithmConfiguration[str, str]):
    """Configuration to predict affinity for a given ligand/protrin target pair."""

    algorithm_type: ClassVar[str] = "prediction"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    protein_targets: List[str] = field(
        default_factory=list,
        metadata=dict(description="List of protein targets as AA sequences."),
    )
    ligands: List[str] = field(
        default_factory=list,
        metadata=dict(description="List of ligands in SMILES format."),
    )
    confidence: bool = field(
        default=False,
        metadata=dict(
            description="Whether the confidence for the prediction should be returned."
        ),
    )

    def get_conditional_generator(
        self, resources_path: str
    ) -> BimodalMCAAffinityPredictor:
        """Instantiate the actual predictor implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`gt4sd.algorithms.prediction.affinity._predicto.implementation.BimodalMCAAffinityPredictor.predict` method for predicting affinity.
        """
        return BimodalMCAAffinityPredictor(
            resources_path=resources_path,
            protein_targets=self.protein_targets,
            ligands=self.ligands,
            confidence=self.confidence,
        )
