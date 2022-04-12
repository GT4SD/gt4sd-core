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
"""Algortihms for topic modelling using zero-shot learning via MLNI models."""

import logging
from dataclasses import field
from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, TypeVar

from ...core import AlgorithmConfiguration, GeneratorAlgorithm
from ...registry import ApplicationsRegistry
from .implementation import ZeroShotClassifier

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = TypeVar("T", bound=Any)
S = TypeVar("S", bound=Any)
Targeted = Callable[[T], Iterable[Any]]


class TopicsZeroShot(GeneratorAlgorithm[S, T]):
    """Topics prediction algorithm."""

    def __init__(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ):
        """Instantiate TopicsZeroShot ready to predict topics.

        Args:
            configuration: domain and application
                specification defining parameters, types and validations.
            target: a target for which to generate items.

        Example:
            An example for predicting topics for a given text::

                config = TopicsPredictor()
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
    ) -> Targeted[T]:
        """Get the function to perform the prediction via TopicsZeroShot's generator.

        Args:
            configuration: helps to set up specific application of TopicsZeroShot.
            target: context or condition for the generation.

        Returns:
            callable with target predicting topics sorted by relevance.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: ZeroShotClassifier = configuration.get_conditional_generator(  # type: ignore
            self.local_artifacts
        )
        return implementation.predict


@ApplicationsRegistry.register_algorithm_application(TopicsZeroShot)
class TopicsPredictor(AlgorithmConfiguration[str, str]):
    """Configuration to generate topics."""

    algorithm_type: ClassVar[str] = "prediction"
    domain: ClassVar[str] = "nlp"
    algorithm_version: str = "dbpedia"

    model_name: str = field(
        default="facebook/bart-large-mnli",
        metadata=dict(
            description="MLNI model name to use. If the  model is not found in the cache, a download from HuggingFace will be attempted."
        ),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.

        Returns:
            target description.
        """
        return {
            "title": "Text to analyze",
            "description": "Text considered for the topics prediction task.",
            "type": "string",
        }

    def get_conditional_generator(self, resources_path: str) -> ZeroShotClassifier:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.prediction.topics_zero_shot.implementation.ZeroShotClassifier.predict>` method for targeted generation.
        """
        return ZeroShotClassifier(
            resources_path=resources_path, model_name=self.model_name
        )
