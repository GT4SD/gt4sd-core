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
"""Algortihms for keyword generation using BERT models."""

import logging
from dataclasses import field
from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, Set, TypeVar

from ...core import (
    AlgorithmConfiguration,
    GeneratorAlgorithm,
    get_configuration_class_with_attributes,
)
from ...registry import ApplicationsRegistry
from .implementation import KeyBERT

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = TypeVar("T", bound=Any)
S = TypeVar("S", bound=Any)
Targeted = Callable[[T], Iterable[Any]]


class KeywordBERTGenerationAlgorithm(GeneratorAlgorithm[S, T]):
    """Topics prediction algorithm."""

    def __init__(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ):
        """Instantiate KeywordBERTGenerationAlgorithm ready to predict topics.

        Args:
            configuration: domain and application
                specification defining parameters, types and validations.
            target: a target for which to generate items.

        Example:
            An example for predicting topics for a given text::

                config = KeyBERTGenerator()
                algorithm = KeywordBERTGenerationAlgorithm(configuration=config, target="This is a text I want to understand better")
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
        """Get the function to perform the prediction via KeywordBERTGenerationAlgorithm's generator.

        Args:
            configuration: helps to set up specific application of KeywordBERTGenerationAlgorithm.
            target: context or condition for the generation.

        Returns:
            callable with target generating keywords sorted by relevance.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: Any = configuration.get_conditional_generator(  # type: ignore
            self.local_artifacts
        )
        return implementation.predict


@ApplicationsRegistry.register_algorithm_application(KeywordBERTGenerationAlgorithm)
class KeyBERTGenerator(AlgorithmConfiguration[str, str]):
    """Configuration to generate keywords.

    If the  model is not found in the cache, models are collected from https://www.sbert.net/docs/pretrained_models.html.
    distilbert-base-nli-stsb-mean-tokens is recommended for english, while xlm-r-bert-base-nli-stsb-mean-tokens for all
    other languages as it support 100+ languages.
    """

    algorithm_name: ClassVar[str] = KeywordBERTGenerationAlgorithm.__name__
    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "nlp"
    algorithm_version: str = "distilbert-base-nli-mean-tokens"

    minimum_keyphrase_ngram: int = field(
        default=1,
        metadata=dict(description=("Lower bound for phrase size.")),
    )
    maximum_keyphrase_ngram: int = field(
        default=2,
        metadata=dict(description=("Upper bound for phrase size.")),
    )
    stop_words: str = field(
        default="english",
        metadata=dict(description=("Language for the stop words removal.")),
    )
    top_n: int = field(
        default=10,
        metadata=dict(description=("Number of keywords to extract.")),
    )
    use_maxsum: bool = field(
        default=False,
        metadata=dict(
            description=("Control usage of max sum similarity for keywords generated.")
        ),
    )
    use_mmr: bool = field(
        default=False,
        metadata=dict(
            description=(
                "Control usage of max marginal relevance for keywords generated."
            )
        ),
    )
    diversity: float = field(
        default=0.5,
        metadata=dict(description=("Diversity for the results when enabling use_mmr.")),
    )
    number_of_candidates: int = field(
        default=20,
        metadata=dict(description=("Candidates considered when enabling use_maxsum.")),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.

        Returns:
            target description.
        """
        return {
            "title": "Text to analyze",
            "description": "Text considered for the keyword generation task.",
            "type": "string",
        }

    def get_conditional_generator(self, resources_path: str) -> KeyBERT:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.key_bert.implementation.KeyBERT.predict>` method for targeted generation.
        """
        return KeyBERT(
            resources_path=resources_path,
            minimum_keyphrase_ngram=self.minimum_keyphrase_ngram,
            maximum_keyphrase_ngram=self.maximum_keyphrase_ngram,
            stop_words=self.stop_words,
            top_n=self.top_n,
            use_maxsum=self.use_maxsum,
            use_mmr=self.use_mmr,
            diversity=self.diversity,
            number_of_candidates=self.number_of_candidates,
            model_name=self.algorithm_version,
        )

    @classmethod
    def list_versions(cls) -> Set[str]:
        """Get possible algorithm versions.

        Standard S3 and cache search adding the version used in the configuration.

        Returns:
            viable values as :attr:`algorithm_version` for the environment.
        """
        logger.warning(
            "more algorithm versions can be found on https://www.sbert.net/docs/pretrained_models.html"
        )
        return (
            get_configuration_class_with_attributes(cls)
            .list_versions()
            .union({cls.algorithm_version})
        )
