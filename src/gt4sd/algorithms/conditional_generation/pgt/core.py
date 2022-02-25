"""Patent Generative Transformer (PGT) generation algorithm."""

import logging
from dataclasses import field
from typing import Any, Callable, ClassVar, Iterable, Optional, Tuple, TypeVar

from typing_extensions import Protocol, runtime_checkable

from ...core import AlgorithmConfiguration, GeneratorAlgorithm
from ...registry import ApplicationsRegistry
from .implementation import CoherenceCheckGenerator, EditGenerator, Generator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = TypeVar("T", bound=Any)
S = TypeVar("S", bound=Any)
Targeted = Callable[[T], Iterable[Any]]

prompts = {
    "title_to_abstract": "{} <|sep|> Given the above title, suggest an abstract <|sep|>",
    "abstract_to_claim": "{} <|sep|> Given the above abstract, suggest a claim <|sep|>",
    "claim_to_abstract": "{} <|sep|> Given the above claim, suggest an abstract <|sep|>",
    "abstract_to_title": "{} <|sep|> Given the above abstract, suggest a title <|sep|>",
    "patent_provenance": "{} <|sep|> {} <|sep|> Do the above TYPE_A and TYPE_B belong to the same patent? <|sep|>",
    "text_infilling": "{} <|sep|> Replace the [MASK] tokens in the above TYPE <|sep|>",
}


class PGT(GeneratorAlgorithm[S, T]):
    """PGT Algorithm."""

    def __init__(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ) -> None:
        """Instantiate PGT ready to generate items.

        Args:
            configuration: domain and application
                specification defining parameters, types and validations.
            target: a target for which to generate items.

        Example:
            An example for generating abstract from a given claim:

                config = PGTGenerator(task="claim_to_abstract")

                generator = PGT(configuration=config, target="My amazing claim")
                print(list(generator.sample()))
        """

        configuration = self.validate_configuration(configuration)

        self.max_samples = configuration.num_return_sequences  # type: ignore

        # No validation/check on the target input here, since model is not yet loaded.
        super().__init__(
            configuration=configuration,  # type:ignore
            target=target,  # type:ignore
        )

    def get_generator(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ) -> Targeted[T]:
        """Get the function to sample with the given configuration.

        Args:
            configuration: helps to set up specific application of PGT.
            target: context or condition for the generation.

        Returns:
            callable with target generating a batch of items.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: Generator = configuration.get_conditional_generator(  # type: ignore
            resources_path=self.local_artifacts, context=target
        )

        return implementation.generate_case  # type: ignore

    def validate_configuration(
        self, configuration: AlgorithmConfiguration[S, T]
    ) -> AlgorithmConfiguration[S, T]:
        @runtime_checkable
        class AnyPGTConfiguration(Protocol):
            """Protocol for PGT configurations."""

            def get_conditional_generator(self, resources_path: str) -> Generator:
                ...

            def validate_item(self, item: Any) -> S:
                ...

        # TODO raise InvalidAlgorithmConfiguration
        assert isinstance(configuration, AnyPGTConfiguration)
        assert isinstance(configuration, AlgorithmConfiguration)
        return configuration


@ApplicationsRegistry.register_algorithm_application(PGT)
class PGTAlgorithm(AlgorithmConfiguration[str, None]):
    """Basic configuration for a PGT algorithm"""

    prompt: str = prompts["title_to_abstract"]  # default prompt

    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "nlp"
    algorithm_version: str = "v0"

    model_type: str = field(
        default="",
        metadata=dict(description="Type of the model."),
    )
    max_length: int = field(
        default=512, metadata=dict(description="Maximum length of the generated text.")
    )
    top_k: int = field(
        default=50,
        metadata=dict(description="Number of top-k probability tokens to keep."),
    )
    top_p: float = field(
        default=1.0,
        metadata=dict(
            description="Only tokens with cumulative probabilities summing up to this value are kept."
        ),
    )
    num_return_sequences: int = field(
        default=3,
        metadata=dict(description="Number of alternatives to be generated."),
    )

    def get_conditional_generator(self, resources_path: str, **kwargs) -> Generator:
        """Instantiate the actual PGT implementation.

        Args:
               resources_path: local path to model files.

        Returns:
               instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.pgt.implementation.Generator.generate_case>` method for targeted generation.
        """
        return Generator(
            resources_path=resources_path,
            model_type=self.model_type,
            model_name=self.algorithm_version,
            prompt=self.prompt,
            max_length=self.max_length,
            top_k=self.top_k,
            top_p=self.top_p,
            num_return_sequences=self.num_return_sequences,
        )


@ApplicationsRegistry.register_algorithm_application(PGT)
class PGTGenerator(PGTAlgorithm):
    """Configuration for a PGT Generator algorithm"""

    task: str = field(
        default="title_to_abstract",
        metadata=dict(
            description="Generation task. Options:"
            "title_to_abstract, abstract_to_title,"
            "abstract_to_claim and claim_to_abstract."
        ),
    )

    def get_conditional_generator(self, resources_path: str, **kwargs) -> Generator:
        """Instantiate the actual PGT implementation for part of patent generation.

        Args:
           resources_path: local path to model files.

        Returns:
           instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.pgt.implementation.Generator.generate_case>` method for targeted generation.
        """

        if self.task not in prompts:
            raise ValueError(f"{self.task} is not a valid option for task.")

        self.prompt = prompts[self.task]

        return super().get_conditional_generator(resources_path, **kwargs)


@ApplicationsRegistry.register_algorithm_application(PGT)
class PGTEditor(PGTAlgorithm):
    """Configuration for a PGT Editor algorithm"""

    type: str = field(
        default="abstract",
        metadata=dict(
            description="In which part of a patent the input text belongs. Options: abstract and title."
        ),
    )

    def get_conditional_generator(self, resources_path: str, **kwargs) -> Generator:
        """Instantiate the actual PGT implementation for part of patent editing.

        Args:
           resources_path: local path to model files.

        Returns:
           instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.pgt.implementation.Generator.generate_case>` method for targeted generation.
        """

        self.prompt = prompts["text_infilling"]
        self.prompt = self.prompt.replace("TYPE", self.type)

        return EditGenerator(
            resources_path=resources_path,
            model_type=self.model_type,
            model_name=self.algorithm_version,
            prompt=self.prompt,
            max_length=self.max_length,
            top_k=self.top_k,
            top_p=self.top_p,
            num_return_sequences=self.num_return_sequences,
        )


@ApplicationsRegistry.register_algorithm_application(PGT)
class PGTCoherenceChecker(PGTAlgorithm):
    """Configuration for a PGT coherence check algorithm"""

    num_return_sequences: int = 1

    input_a: str = field(
        default="I'm a stochastic parrot.",
        metadata=dict(description="First input for coherence check."),
    )

    input_b: str = field(
        default="I'm a stochastic parrot.",
        metadata=dict(description="Second input for coherence check."),
    )

    coherence_type: str = field(
        default="title-abstract",
        metadata=dict(
            description="Combination of inputs for the check. Options:"
            "title-abstract, title-claim and abstract-claim."
        ),
    )

    valid_types = ["title-abstract", "abstract-claim", "title-claim"]

    def extract_coherence_types(self) -> Tuple[str, str]:
        """Check the validity and extract coherence types of input text

        Returns:
            Tuple containing the type of the input.
        """

        if self.coherence_type in self.valid_types:
            type_a, type_b = self.coherence_type.split("-")

            return type_a, type_b

        else:
            raise ValueError(f"{self.coherence_type} is not a valid coherence type")

    def get_conditional_generator(self, resources_path: str, **kwargs) -> Generator:
        """Instantiate the actual PGT implementation for patent coherence check.

        Args:
           resources_path: local path to model files.

        Returns:
           instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.pgt.implementation.Generator.generate_case>` method for targeted generation.
        """

        type_a, type_b = self.coherence_type.split("-")

        self.prompt = prompts["patent_provenance"]
        self.prompt = self.prompt.replace("TYPE_A", type_a)
        self.prompt = self.prompt.replace("TYPE_B", type_b)

        return CoherenceCheckGenerator(
            resources_path=resources_path,
            model_type=self.model_type,
            model_name=self.algorithm_version,
            prompt=self.prompt,
            max_length=self.max_length,
            top_k=self.top_k,
            top_p=self.top_p,
            num_return_sequences=self.num_return_sequences,
        )
