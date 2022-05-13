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
"""Patent Generative Transformer (PGT) generation algorithm."""

import logging
import os
import shutil
from dataclasses import field
from typing import Any, ClassVar, Dict, Optional, TypeVar

from typing_extensions import Protocol, runtime_checkable

from ....cli.pl_to_hf_converter import convert_pl_to_hf
from ....training_pipelines.core import TrainingPipelineArguments
from ....training_pipelines.pytorch_lightning.language_modeling.core import (
    LanguageModelingSavingArguments,
)
from ...core import AlgorithmConfiguration, GeneratorAlgorithm, Untargeted
from ...registry import ApplicationsRegistry
from .implementation import (
    COHERENCE_TYPES,
    EDITING_TYPES,
    GENERATION_PROMPTS,
    CoherenceCheckGenerator,
    EditGenerator,
    Generator,
    PartGenerator,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = type(None)
S = TypeVar("S", bound=str)


class PGT(GeneratorAlgorithm[S, T]):
    """PGT Algorithm."""

    def __init__(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T] = None,
    ) -> None:
        """Instantiate PGT ready to generate items.

        Args:
            configuration: domain and application
                specification defining parameters, types and validations.
            target: unused since it is not a conditional generator.

        Example:
            An example for generating abstract from a given claim:

                config = PGTGenerator(task="claim_to_abstract", input_text="My interesting claim")
                generator = PGT(configuration=config)
                print(list(generator.sample(1)))
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
    ) -> Untargeted:
        """Get the function to sample with the given configuration.

        Args:
            configuration: helps to set up specific application of PGT.
            target: context or condition for the generation. Unused in the algorithm.

        Returns:
            callable with target generating a batch of items.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: Generator = configuration.get_generator(  # type: ignore
            self.local_artifacts
        )

        return implementation.generate_case  # type: ignore

    def validate_configuration(
        self, configuration: AlgorithmConfiguration[S, T]
    ) -> AlgorithmConfiguration[S, T]:
        @runtime_checkable
        class AnyPGTConfiguration(Protocol):
            """Protocol for PGT configurations."""

            def get_generator(self, resources_path: str) -> Generator:
                ...

            def validate_item(self, item: Any) -> S:
                ...

        # TODO raise InvalidAlgorithmConfiguration
        assert isinstance(configuration, AnyPGTConfiguration)
        assert isinstance(configuration, AlgorithmConfiguration)
        return configuration


@ApplicationsRegistry.register_algorithm_application(PGT)
class PGTAlgorithmConfiguration(AlgorithmConfiguration[str, None]):
    """Basic configuration for a PGT algorithm"""

    algorithm_type: ClassVar[str] = "generation"
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
    no_repeat_ngram_size: int = field(
        default=2,
        metadata=dict(description="Size of n-gram to not appear twice."),
    )

    def get_target_description(self) -> Optional[Dict[str, str]]:
        """Get description of the target for generation.

        Returns:
            target description, returns None in case no target is used.
        """
        return None

    def get_generator(self, resources_path: str, **kwargs) -> Generator:
        """Instantiate the actual PGT implementation.

        Args:
               resources_path: local path to model files.

        Returns:
               instance with
                :meth:`generate_batch<gt4sd.algorithms.generation.pgt.implementation.Generator.generate_case>`
                 method for targeted generation.
        """
        return Generator(
            resources_path=resources_path,
            model_type=self.model_type,
            model_name=self.algorithm_version,
            max_length=self.max_length,
            top_k=self.top_k,
            top_p=self.top_p,
            num_return_sequences=self.num_return_sequences,
        )

    @classmethod
    def save_version_from_training_pipeline_arguments_postprocess(
        cls,
        training_pipeline_arguments: TrainingPipelineArguments,
    ):
        """Postprocess after saving. Remove temporarily converted hf model
           if pytorch-lightning checkpoint is given.

        Args:
            training_pipeline_arguments: training pipeline arguments.
        """

        if isinstance(training_pipeline_arguments, LanguageModelingSavingArguments):
            if training_pipeline_arguments.ckpt is not None:
                shutil.rmtree(training_pipeline_arguments.hf_model_path)

                logger.info(
                    f"Cleaning up temporary files from {training_pipeline_arguments.hf_model_path}"
                )
        else:
            return super().save_version_from_training_pipeline_arguments_postprocess(
                training_pipeline_arguments
            )

    @classmethod
    def get_filepath_mappings_for_training_pipeline_arguments(
        cls, training_pipeline_arguments: TrainingPipelineArguments
    ) -> Dict[str, str]:
        """Ger filepath mappings for the given training pipeline arguments.

        Args:
            training_pipeline_arguments: training pipeline arguments.

        Returns:
            a mapping between artifacts' files and training pipeline's output files.
        """

        if isinstance(training_pipeline_arguments, LanguageModelingSavingArguments):

            if training_pipeline_arguments.ckpt is not None:

                convert_pl_to_hf(training_pipeline_arguments)

            model_files = os.listdir(training_pipeline_arguments.hf_model_path)

            model_files_dict = {
                file: os.path.join(training_pipeline_arguments.hf_model_path, file)
                for file in model_files
            }
            return model_files_dict

        else:
            return super().get_filepath_mappings_for_training_pipeline_arguments(
                training_pipeline_arguments
            )


@ApplicationsRegistry.register_algorithm_application(PGT)
class PGTGenerator(PGTAlgorithmConfiguration):
    """Configuration for a PGT Generator algorithm"""

    input_text: str = field(
        default="This is my input",
        metadata=dict(description="Input text."),
    )
    task: str = field(
        default="title-to-abstract",
        metadata=dict(
            description=f"Generation tasks. Supported: {', '.join(GENERATION_PROMPTS.keys())}"
        ),
    )

    def get_generator(self, resources_path: str, **kwargs) -> Generator:
        """Instantiate the actual PGT implementation for part of patent generation.

        Args:
           resources_path: local path to model files.

        Returns:
           instance with
            :meth:`generate_batch<gt4sd.algorithms.generation.pgt.implementation.Generator.generate_case>`
             method for targeted generation.
        """

        return PartGenerator(
            resources_path=resources_path,
            input_text=self.input_text,
            model_type=self.model_type,
            model_name=self.algorithm_version,
            max_length=self.max_length,
            top_k=self.top_k,
            top_p=self.top_p,
            num_return_sequences=self.num_return_sequences,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            task=self.task,
        )


@ApplicationsRegistry.register_algorithm_application(PGT)
class PGTEditor(PGTAlgorithmConfiguration):
    """Configuration for a PGT Editor algorithm."""

    input_text: str = field(
        default="This is my input",
        metadata=dict(description="Input text."),
    )
    input_type: str = field(
        default="abstract",
        metadata=dict(
            description=f"Part of a patent the input text belongs. Supported: {', '.join(EDITING_TYPES)}"
        ),
    )

    def get_generator(self, resources_path: str, **kwargs) -> Generator:
        """Instantiate the actual PGT implementation for part of patent editing.

        Args:
           resources_path: local path to model files.

        Returns:
           instance with
            :meth:`generate_batch<gt4sd.algorithms.generation.pgt.implementation.Generator.generate_case>`
             method for targeted generation.
        """

        return EditGenerator(
            resources_path=resources_path,
            input_text=self.input_text,
            model_type=self.model_type,
            model_name=self.algorithm_version,
            max_length=self.max_length,
            top_k=self.top_k,
            top_p=self.top_p,
            num_return_sequences=self.num_return_sequences,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            input_type=self.input_type,
        )


@ApplicationsRegistry.register_algorithm_application(PGT)
class PGTCoherenceChecker(PGTAlgorithmConfiguration):
    """Configuration for a PGT coherence check algorithm"""

    num_return_sequences: int = field(
        default=1,
        metadata=dict(
            description="Number of alternatives should be always 1 for coherence check."
        ),
    )

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
            description=f"Input types for the check. Supported: {', '.join(COHERENCE_TYPES)}"
        ),
    )

    def get_generator(self, resources_path: str, **kwargs) -> Generator:
        """Instantiate the actual PGT implementation for patent coherence check.

        Args:
           resources_path: local path to model files.

        Returns:
           instance with
            :meth:`generate_batch<gt4sd.algorithms.generation.pgt.implementation.Generator.generate_case>`
            method for targeted generation.
        """

        return CoherenceCheckGenerator(
            resources_path=resources_path,
            input_a=self.input_a,
            input_b=self.input_b,
            model_type=self.model_type,
            model_name=self.algorithm_version,
            max_length=self.max_length,
            top_k=self.top_k,
            top_p=self.top_p,
            num_return_sequences=self.num_return_sequences,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            coherence_type=self.coherence_type,
        )
