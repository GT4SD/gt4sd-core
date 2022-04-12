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
import logging
from dataclasses import field
from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, TypeVar

from ....training_pipelines.core import TrainingPipelineArguments
from ....training_pipelines.guacamol_baselines.core import GuacaMolSavingArguments
from ....training_pipelines.moses.core import MosesSavingArguments
from ...core import AlgorithmConfiguration, GeneratorAlgorithm
from ...registry import ApplicationsRegistry
from .implementation import (
    AaeIterator,
    Generator,
    GraphGAIterator,
    GraphMCTSIterator,
    OrganIterator,
    SMILESGAIterator,
    SMILESLSTMHCIterator,
    SMILESLSTMPPOIterator,
    VaeIterator,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = TypeVar("T", bound=Any)
S = TypeVar("S", bound=Any)
Targeted = Callable[[T], Iterable[Any]]


class GuacaMolGenerator(GeneratorAlgorithm[S, T]):
    """GuacaMol generation algorithm."""

    def __init__(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ):
        """
        Instantiate GuacaMolGenerator ready to generate samples.

        Args:
            configuration: domain and application
                specification defining parameters, types and validations.
            target: a target for which to generate items.

        Example:
            An example for generating molecules given a scoring function and a score::

                config = SMILESGAGenerator()
                target = {"scoring_function_name": {"target": 0.0}}
                algorithm = GuacaMolGenerator(configuration=config, target=target)
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
        """Get the function to perform the prediction via GuacaMol's generator.

        Args:
            configuration: helps to set up specific application of GuacaMol.

        Returns:
            callable with target generating samples.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: Generator = configuration.get_conditional_generator(  # type: ignore
            self.local_artifacts
        )
        return implementation.generate_batch  # type: ignore


@ApplicationsRegistry.register_algorithm_application(GuacaMolGenerator)
class SMILESGAGenerator(AlgorithmConfiguration[str, str]):
    """Configuration to generate optimizied molecules using SMILES Genetic algorithm"""

    algorithm_name: ClassVar[str] = GuacaMolGenerator.__name__
    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    batch_size: int = field(
        default=32,
        metadata=dict(description="Batch size used for the generative model sampling."),
    )
    population_size: int = field(
        default=100,
        metadata=dict(
            description="it is used with n_mutations for the initial generation of smiles within the population"
        ),
    )
    n_mutations: int = field(
        default=200,
        metadata=dict(
            description="it is used with population size for the initial generation of smiles within the population"
        ),
    )
    n_jobs: int = field(
        default=-1,
        metadata=dict(description="number of concurrently running jobs"),
    )
    gene_size: int = field(
        default=2,
        metadata=dict(
            description="size of the gene which is used in creation of genes"
        ),
    )
    random_start: bool = field(
        default=False,
        metadata=dict(
            description="set to True to randomly choose list of SMILES for generating optimizied molecules"
        ),
    )
    generations: int = field(
        default=2,
        metadata=dict(description="number of evolutionary generations"),
    )
    patience: int = field(
        default=4,
        metadata=dict(
            description="it is used for early stopping if population scores remains the same after generating molecules"
        ),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.

        Returns:
            target description.
        """
        return {
            "title": "Scoring functions with parameters",
            "description": "Scoring functions will be used to generate a score for SMILES.",
            "type": "object",
        }

    def get_conditional_generator(self, resources_path: str) -> SMILESGAIterator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.guacamol.implementation.smiles_ga.SMILESGAIterator.generate_batch>` method for targeted generation.
        """
        return SMILESGAIterator(
            resource_path=resources_path,
            population_size=self.population_size,
            n_mutations=self.n_mutations,
            n_jobs=self.n_jobs,
            random_start=self.random_start,
            gene_size=self.gene_size,
            generations=self.generations,
            patience=self.patience,
            batch_size=self.batch_size,
        )


@ApplicationsRegistry.register_algorithm_application(GuacaMolGenerator)
class GraphGAGenerator(AlgorithmConfiguration[str, str]):
    """Configuration to generate optimizied molecules using Graph-Based Genetic algorithm"""

    algorithm_name: ClassVar[str] = GuacaMolGenerator.__name__
    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    batch_size: int = field(
        default=1,
        metadata=dict(description="Batch size used for the generative model sampling."),
    )
    population_size: int = field(
        default=100,
        metadata=dict(
            description="it is used with n_mutations for the initial generation of smiles within the population"
        ),
    )
    mutation_rate: float = field(
        default=0.01,
        metadata=dict(
            description="frequency of the new mutations in a single gene or organism over time"
        ),
    )
    offspring_size: int = field(
        default=200,
        metadata=dict(description="number of molecules to select for new population"),
    )
    n_jobs: int = field(
        default=-1,
        metadata=dict(description="number of concurrently running jobs"),
    )
    random_start: bool = field(
        default=False,
        metadata=dict(
            description="set to True to randomly choose list of SMILES for generating optimizied molecules"
        ),
    )
    generations: int = field(
        default=2,
        metadata=dict(description="number of evolutionary generations"),
    )
    patience: int = field(
        default=4,
        metadata=dict(
            description="it is used for early stopping if population scores remains the same after generating molecules"
        ),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.

        Returns:
            target description.
        """
        return {
            "title": "Scoring functions with parameters",
            "description": "Scoring functions will be used to generate a score for SMILES.",
            "type": "object",
        }

    def get_conditional_generator(self, resources_path: str) -> GraphGAIterator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.guacamol.implementation.graph_ga.GraphGAIterator.generate_batch>` method for targeted generation.
        """
        return GraphGAIterator(
            resource_path=resources_path,
            batch_size=self.batch_size,
            offspring_size=self.offspring_size,
            population_size=self.population_size,
            mutation_rate=self.mutation_rate,
            n_jobs=self.n_jobs,
            random_start=self.random_start,
            generations=self.generations,
            patience=self.patience,
        )


@ApplicationsRegistry.register_algorithm_application(GuacaMolGenerator)
class GraphMCTSGenerator(AlgorithmConfiguration[str, str]):
    """Configuration to generate optimizied molecules using Graph-based Genetic Algorithm and Generative Model/Monte Carlo Tree Search for the Exploration of Chemical Space"""

    algorithm_name: ClassVar[str] = GuacaMolGenerator.__name__
    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    batch_size: int = field(
        default=1,
        metadata=dict(description="Batch size used for the generative model sampling."),
    )
    init_smiles: str = field(
        default="",
        metadata=dict(description="initial SMILES used for generation of states."),
    )
    population_size: int = field(
        default=100,
        metadata=dict(
            description="it is used with n_mutations for the initial generation of smiles within the population"
        ),
    )
    n_jobs: int = field(
        default=-1,
        metadata=dict(description="number of concurrently running jobs"),
    )
    generations: int = field(
        default=1000,
        metadata=dict(description="number of evolutionary generations"),
    )
    patience: int = field(
        default=4,
        metadata=dict(
            description="it is used for early stopping if population scores remains the same after generating molecules"
        ),
    )
    num_sims: float = field(
        default=40,
        metadata=dict(description="number of times to traverse the tree"),
    )
    max_children: int = field(
        default=25,
        metadata=dict(description="maximum number of childerns a node could have"),
    )
    max_atoms: int = field(
        default=60,
        metadata=dict(
            description="maximum number of atoms to explore to terminal the node state"
        ),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.

        Returns:
            target description.
        """
        return {
            "title": "Scoring functions with parameters",
            "description": "Scoring functions will be used to generate a score for SMILES.",
            "type": "object",
        }

    def get_conditional_generator(self, resources_path: str) -> GraphMCTSIterator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.guacamol.implementation.graph_mcts.GraphMCTSIterator.generate_batch>` method for targeted generation.
        """
        return GraphMCTSIterator(
            init_smiles=self.init_smiles,
            batch_size=self.batch_size,
            population_size=self.population_size,
            max_children=self.max_children,
            num_sims=self.num_sims,
            generations=self.generations,
            n_jobs=self.n_jobs,
            max_atoms=self.max_atoms,
            patience=self.patience,
        )


@ApplicationsRegistry.register_algorithm_application(GuacaMolGenerator)
class SMILESLSTMHCGenerator(AlgorithmConfiguration[str, str]):
    """Configuration to generate optimized molecules using recurrent neural networks with hill climbing algorithm."""

    algorithm_name: ClassVar[str] = GuacaMolGenerator.__name__
    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    batch_size: int = field(
        default=1,
        metadata=dict(description="Batch size used for the generative model sampling."),
    )
    n_jobs: int = field(
        default=-1,
        metadata=dict(description="number of concurrently running jobs"),
    )
    n_epochs: int = field(
        default=20,
        metadata=dict(description="number of epochs to sample"),
    )
    mols_to_sample: int = field(
        default=1024,
        metadata=dict(description="molecules sampled at each step"),
    )
    keep_top: int = field(
        default=512,
        metadata=dict(description="maximum length of a SMILES string"),
    )
    optimize_n_epochs: int = field(
        default=2,
        metadata=dict(description="number of epochs for the optimization"),
    )
    max_len: int = field(
        default=100,
        metadata=dict(description="maximum length of a SMILES string"),
    )
    optimize_batch_size: int = field(
        default=256,
        metadata=dict(description="batch size for the optimization"),
    )
    benchmark_num_samples: int = field(
        default=4096,
        metadata=dict(
            description="number of molecules to generate from final model for the benchmark"
        ),
    )
    random_start: bool = field(
        default=False,
        metadata=dict(
            description="set to True to randomly choose list of SMILES for generating optimizied molecules"
        ),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.

        Returns:
            target description.
        """
        return {
            "title": "Scoring functions with parameters",
            "description": "Scoring functions will be used to generate a score for SMILES.",
            "type": "object",
        }

    def get_conditional_generator(self, resources_path: str) -> SMILESLSTMHCIterator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.guacamol.implementation.smiles_lstm_hc.SMILESLSTMHCIterator.generate_batch>` method for targeted generation.
        """
        return SMILESLSTMHCIterator(
            resource_path=resources_path,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            mols_to_sample=self.mols_to_sample,
            keep_top=self.keep_top,
            optimize_n_epochs=self.optimize_n_epochs,
            max_len=self.max_len,
            optimize_batch_size=self.optimize_batch_size,
            benchmark_num_samples=self.benchmark_num_samples,
            random_start=self.random_start,
            n_jobs=self.n_jobs,
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
        if isinstance(training_pipeline_arguments, GuacaMolSavingArguments):
            return {
                "model_final_0.473.pt": training_pipeline_arguments.model_filepath,
                "model_final_0.473.json": training_pipeline_arguments.model_config_filepath,
                "guacamol_v1_all.smiles": "",
            }
        else:
            return super().get_filepath_mappings_for_training_pipeline_arguments(
                training_pipeline_arguments
            )


@ApplicationsRegistry.register_algorithm_application(GuacaMolGenerator)
class SMILESLSTMPPOGenerator(AlgorithmConfiguration[str, str]):
    """Configuration to generate optimizied molecules using recurrent neural networks with hill climbing algorithm"""

    algorithm_name: ClassVar[str] = GuacaMolGenerator.__name__
    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    batch_size: int = field(
        default=1,
        metadata=dict(description="Batch size used for the generative model sampling."),
    )
    num_epochs: int = field(
        default=20,
        metadata=dict(description="number of epochs to sample"),
    )
    episode_size: int = field(
        default=8192,
        metadata=dict(
            description="number of molecules sampled by the policy at the start of a series of ppo updates"
        ),
    )
    optimize_batch_size: int = field(
        default=1024,
        metadata=dict(description="batch size for the optimization"),
    )
    entropy_weight: int = field(
        default=1,
        metadata=dict(description="used for calculating entropy loss"),
    )
    kl_div_weight: int = field(
        default=10,
        metadata=dict(
            description="used for calculating Kullback-Leibler divergence loss"
        ),
    )
    clip_param: float = field(
        default=0.2,
        metadata=dict(
            description="used for determining how far the new policy is from the old one"
        ),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.

        Returns:
            target description.
        """
        return {
            "title": "Scoring functions with parameters",
            "description": "Scoring functions will be used to generate a score for SMILES.",
            "type": "object",
        }

    def get_conditional_generator(self, resources_path: str) -> SMILESLSTMPPOIterator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.guacamol.implementation.smiles_lstm_ppo.SMILESLSTMPPOIterator.generate_batch>` method for targeted generation.
        """
        return SMILESLSTMPPOIterator(
            resource_path=resources_path,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            episode_size=self.episode_size,
            optimize_batch_size=self.optimize_batch_size,
            entropy_weight=self.entropy_weight,
            kl_div_weight=self.kl_div_weight,
            clip_param=self.clip_param,
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
        if isinstance(training_pipeline_arguments, GuacaMolSavingArguments):
            return {
                "model_final_0.473.pt": training_pipeline_arguments.model_filepath,
                "model_final_0.473.json": training_pipeline_arguments.model_config_filepath,
            }
        else:
            return super().get_filepath_mappings_for_training_pipeline_arguments(
                training_pipeline_arguments
            )


class MosesGenerator(GeneratorAlgorithm[S, T]):
    """Moses generation algorithm."""

    def __init__(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ):
        """
        Instantiate GuacaMolGenerator ready to generate samples.

        Args:
            configuration: domain and application
                specification defining parameters, types and validations.
            target: a target for which to generate items.

        Example:
            An example for generating molecules given a scoring function and a score:

                config = AaeGenerator()
                algorithm = MosesGenerator(configuration=config, target="")
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
        """Get the function to perform the prediction via GuacaMol's generator.

        Args:
            configuration: helps to set up specific application of GuacaMol.

        Returns:
            callable with target generating samples.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: Generator = configuration.get_conditional_generator(  # type: ignore
            self.local_artifacts
        )
        return implementation.generate_batch  # type: ignore


@ApplicationsRegistry.register_algorithm_application(MosesGenerator)
class AaeGenerator(AlgorithmConfiguration[str, str]):
    """Configuration to generate molecules using an adversarial autoencoder."""

    algorithm_name: ClassVar[str] = MosesGenerator.__name__
    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    n_samples: int = field(
        default=20,
        metadata=dict(description="Number of SMILES to generate"),
    )
    n_batch: int = field(
        default=1024,
        metadata=dict(description="Batch size for the optimization"),
    )
    max_len: int = field(
        default=100,
        metadata=dict(description="Maximum length of the generated SMILES"),
    )

    def get_conditional_generator(self, resources_path: str) -> AaeIterator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.guacamol.implementation.AaeIterator.generate_batch>` method for targeted generation.
        """
        return AaeIterator(
            resource_path=resources_path,
            n_samples=self.n_samples,
            n_batch=self.n_batch,
            max_len=self.max_len,
        )


@ApplicationsRegistry.register_algorithm_application(MosesGenerator)
class VaeGenerator(AlgorithmConfiguration[str, str]):
    """Configuration to generate molecules using a variational autoencoder."""

    algorithm_name: ClassVar[str] = MosesGenerator.__name__
    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    n_samples: int = field(
        default=20,
        metadata=dict(description="Number of SMILES to generate"),
    )
    n_batch: int = field(
        default=1024,
        metadata=dict(description="Batch size for the optimization"),
    )
    max_len: int = field(
        default=100,
        metadata=dict(description="Maximum length of the generated SMILES"),
    )

    def get_conditional_generator(self, resources_path: str) -> VaeIterator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.guacamol.implementation.VaeIterator.generate_batch>` method for targeted generation.
        """
        return VaeIterator(
            resource_path=resources_path,
            n_samples=self.n_samples,
            n_batch=self.n_batch,
            max_len=self.max_len,
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
        if isinstance(training_pipeline_arguments, MosesSavingArguments):
            return {
                "model.pt": training_pipeline_arguments.model_path,
                "config.pt": training_pipeline_arguments.config_path,
                "vocab.pt": training_pipeline_arguments.vocab_path,
            }
        else:
            return super().get_filepath_mappings_for_training_pipeline_arguments(
                training_pipeline_arguments
            )


@ApplicationsRegistry.register_algorithm_application(MosesGenerator)
class OrganGenerator(AlgorithmConfiguration[str, str]):
    """Configuration to generate molecules using Objective-Reinforced Generative Adversarial Network"""

    algorithm_name: ClassVar[str] = MosesGenerator.__name__
    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    n_samples: int = field(
        default=20,
        metadata=dict(description="Number of SMILES to generate"),
    )
    n_batch: int = field(
        default=1024,
        metadata=dict(description="Batch size for the optimization"),
    )
    max_len: int = field(
        default=100,
        metadata=dict(description="Maximum length of the generated SMILES"),
    )

    def get_conditional_generator(self, resources_path: str) -> OrganIterator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.guacamol.implementation.OrganIterator.generate_batch>` method for targeted generation.
        """
        return OrganIterator(
            resource_path=resources_path,
            n_samples=self.n_samples,
            n_batch=self.n_batch,
            max_len=self.max_len,
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
        if isinstance(training_pipeline_arguments, MosesSavingArguments):
            return {
                "model.pt": training_pipeline_arguments.model_path,
                "config.pt": training_pipeline_arguments.config_path,
                "vocab.pt": training_pipeline_arguments.vocab_path,
            }
        else:
            return super().get_filepath_mappings_for_training_pipeline_arguments(
                training_pipeline_arguments
            )
