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
"""GuacaMol algorithms implementation module."""

import logging
import os
from typing import Any, List

from guacamol_baselines.graph_ga.goal_directed_generation import GB_GA_Generator
from guacamol_baselines.graph_mcts.goal_directed_generation import GB_MCTS_Generator
from guacamol_baselines.moses_baselines.aae_distribution_learning import AaeGenerator
from guacamol_baselines.moses_baselines.organ_distribution_learning import (
    OrganGenerator,
)
from guacamol_baselines.moses_baselines.vae_distribution_learning import VaeGenerator
from guacamol_baselines.smiles_ga.goal_directed_generation import ChemGEGenerator
from guacamol_baselines.smiles_lstm_hc.goal_directed_generation import (
    SmilesRnnDirectedGenerator,
)
from guacamol_baselines.smiles_lstm_ppo.goal_directed_generation import (
    PPODirectedGenerator,
)

from .....frameworks.torch import claim_device_name
from .....properties.scores import CombinedScorer
from .....properties.utils import get_target_parameters
from .graph_ga import GraphGA
from .graph_mcts import GraphMCTS
from .moses_aae import AAE
from .moses_organ import Organ
from .moses_vae import VAE
from .smiles_ga import SMILESGA
from .smiles_lstm_hc import SMILESLSTMHC
from .smiles_lstm_ppo import SMILESLSTMPPO

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Generator:
    """Abstract interface for a conditional generator."""

    def generate_batch(self, target) -> List[Any]:
        """Generate a batch of molecules.

        Args:
            target: condition used for generation.

        Returns:
            the generated molecules.
        """
        raise NotImplementedError(
            "Implementation not found for generation of molecules."
        )


class SMILESGAIterator(Generator):
    def __init__(
        self,
        resource_path,
        batch_size: int,
        population_size: int,
        n_mutations: int,
        n_jobs: int,
        random_start: bool,
        gene_size: int,
        generations: int,
        patience: int,
    ):
        """Initialize SMILESGAIterator.

        Args:
            resource_path: path to load the hypothesis, candidate labels and, optionally, the smiles file.
            batch_size: number of molecules to generate.
            population_size: used with n_mutations for the initial generation of smiles within the population.
            n_mutations: used with population size for the initial generation of smiles within the population.
            n_jobs: number of concurrently running jobs.
            random_start: set to True to randomly choose list of SMILES for generating optimizied molecules.
            gene_size: size of the gene which is used in creation of genes.
            generations: number of evolutionary generations.
            patience: used for early stopping if population scores remains the same after generating molecules.
        """
        self.resource_path = resource_path
        self.batch_size = batch_size
        self.population_size = population_size
        self.n_mutations = n_mutations
        self.n_jobs = n_jobs
        self.random_start = random_start
        self.gene_size = gene_size
        self.generations = generations
        self.patience = patience
        self.chemGenerator: ChemGEGenerator = None

    def generate_batch(self, target) -> List[Any]:
        """Generate a batch of molecules.

        Args:
            target: condition used for generation.

        Returns:
            the generated molecules.
        """
        score_list, weights = get_target_parameters(target)

        self.scoring_function = CombinedScorer(
            scorer_list=score_list,
            weights=weights,
        )
        if self.chemGenerator is None:
            optimiser = SMILESGA(
                smi_file=os.path.join(self.resource_path, "guacamol_v1_all.smiles"),
                population_size=self.population_size,
                n_mutations=self.n_mutations,
                gene_size=self.gene_size,
                generations=self.generations,
                n_jobs=self.n_jobs,
                random_start=self.random_start,
                patience=self.patience,
            )
            logger.info("Initialization of the Generator")
            self.chemGenerator = optimiser.get_generator()

        logger.info("generating molecules")
        molecules = self.chemGenerator.generate_optimized_molecules(
            self.scoring_function, self.batch_size
        )
        return molecules


class GraphGAIterator(Generator):
    def __init__(
        self,
        resource_path,
        batch_size: int,
        population_size: int,
        offspring_size: int,
        n_jobs: int,
        mutation_rate: float,
        random_start: bool,
        generations: int,
        patience: int,
    ):
        """Initialize GraphGAIterator.

        Args:
            resource_path: path to load the hypothesis, candidate labels and, optionally, the smiles file.
            batch_size: number of molecules to generate.
            population_size: used for the initial generation of smiles within the population.
            n_jobs: number of concurrently running jobs.
            random_start: set to True to randomly choose list of SMILES for generating optimizied molecules.
            offspring_size: number of molecules to select for new population.
            mutation_rate: frequency of the new mutations in a single gene or organism over time.
            generations: number of evolutionary generations.
            patience: used for early stopping if population scores remains the same after generating molecules.
        """
        self.resource_path = resource_path
        self.batch_size = batch_size
        self.population_size = population_size
        self.n_jobs = n_jobs
        self.random_start = random_start
        self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.patience = patience
        self.gb_ga_generator: GB_GA_Generator = None

    def generate_batch(self, target) -> List[Any]:
        """Generate a batch of molecules.

        Args:
            target: condition used for generation.

        Returns:
            the generated molecules.
        """
        score_list, weights = get_target_parameters(target)

        self.scoring_function = CombinedScorer(
            scorer_list=score_list,
            weights=weights,
        )
        if self.gb_ga_generator is None:
            optimiser = GraphGA(
                smi_file=os.path.join(self.resource_path, "guacamol_v1_all.smiles"),
                population_size=self.population_size,
                mutation_rate=self.mutation_rate,
                offspring_size=self.offspring_size,
                generations=self.generations,
                n_jobs=self.n_jobs,
                random_start=self.random_start,
                patience=self.patience,
            )
            logger.info("Initialization of the Generator")
            self.gb_ga_generator = optimiser.get_generator()

        logger.info("generating molecules")
        molecules = self.gb_ga_generator.generate_optimized_molecules(
            self.scoring_function, self.batch_size
        )
        return molecules


class GraphMCTSIterator(Generator):
    def __init__(
        self,
        init_smiles: str,
        batch_size: int,
        population_size: int,
        max_children: int,
        n_jobs: int,
        num_sims: float,
        max_atoms: int,
        generations: int,
        patience: int,
    ):
        """Initialize GraphMCTSIterator.

        Args:
            init_smiles: path where to load hypothesis, candidate labels and, optionally, the smiles file.
            batch_size: number of molecules to generate.
            population_size: used for the initial generation of smiles within the population.
            max_children: maximum number of childerns a node could have.
            n_jobs: number of concurrently running jobs.
            num_sims: number of times to traverse the tree.
            max_atoms: maximum number of atoms to explore to terminal the node state.
            generations: number of evolutionary generations.
            patience: used for early stopping if population scores remains the same after generating molecules.
        """
        self.init_smiles = init_smiles
        self.batch_size = batch_size
        self.population_size = population_size
        self.max_children = max_children
        self.n_jobs = n_jobs
        self.num_sims = num_sims
        self.max_atoms = max_atoms
        self.generations = generations
        self.patience = patience
        self.grah_mcts_generator: GB_MCTS_Generator = None

    def generate_batch(self, target) -> List[Any]:
        """Generate a batch of molecules.

        Args:
            target: condition used for generation.

        Returns:
            the generated molecules.
        """
        score_list, weights = get_target_parameters(target)

        self.scoring_function = CombinedScorer(
            scorer_list=score_list,
            weights=weights,
        )
        if self.grah_mcts_generator is None:
            optimiser = GraphMCTS(
                init_smiles=self.init_smiles,
                population_size=self.population_size,
                max_children=self.max_children,
                num_sims=self.num_sims,
                generations=self.generations,
                n_jobs=self.n_jobs,
                max_atoms=self.max_atoms,
                patience=self.patience,
            )
            logger.info("Initialization of the Generator")
            self.grah_mcts_generator = optimiser.get_generator()

        logger.info("generating molecules")
        molecules = self.grah_mcts_generator.generate_optimized_molecules(
            self.scoring_function, self.batch_size
        )
        return molecules


class SMILESLSTMHCIterator(Generator):
    def __init__(
        self,
        resource_path,
        batch_size: int,
        n_epochs: int,
        mols_to_sample: int,
        n_jobs: int,
        random_start: bool,
        optimize_n_epochs: int,
        benchmark_num_samples: int,
        keep_top: int,
        max_len: int,
        optimize_batch_size: int,
    ):
        """Initialize SMILESLSTMHCIterator.

        Args:
            resource_path: path to load the hypothesis, candidate labels and, optionally, the smiles file.
            batch_size: number of molecules to generate.
            n_epochs: number of epochs to sample.
            mols_to_sample: molecules sampled at each step.
            keep_top: molecules kept each step.
            optimize_n_epochs: number of epochs for the optimization.
            benchmark_num_samples: number of molecules to generate from final model for the benchmark.
            random_start: set to True to randomly choose list of SMILES for generating optimizied molecules.
            n_jobs: number of concurrently running jobs.
            max_len: maximum length of a SMILES string.
            optimize_batch_size: batch size for the optimization.
        """
        self.resource_path = resource_path
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.mols_to_sample = mols_to_sample
        self.keep_top = keep_top
        self.optimize_n_epochs = optimize_n_epochs
        self.benchmark_num_samples = benchmark_num_samples
        self.random_start = random_start
        self.n_jobs = n_jobs
        self.max_len = max_len
        self.optimize_batch_size = optimize_batch_size
        self.smiles_lstm_hc_generator: SmilesRnnDirectedGenerator = None

    def generate_batch(self, target) -> List[Any]:
        """Generate a batch of molecules.

        Args:
            target: condition used for generation.

        Returns:
            the generated molecules.
        """
        score_list, weights = get_target_parameters(target)

        self.scoring_function = CombinedScorer(
            scorer_list=score_list,
            weights=weights,
        )
        if self.smiles_lstm_hc_generator is None:
            optimiser = SMILESLSTMHC(
                model_path=os.path.join(self.resource_path, "model_final_0.473.pt"),
                smi_file=os.path.join(self.resource_path, "guacamol_v1_all.smiles"),
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
            logger.info("Initialization of the Generator")
            self.smiles_lstm_hc_generator = optimiser.get_generator()

        logger.info("generating molecules")
        molecules = self.smiles_lstm_hc_generator.generate_optimized_molecules(
            self.scoring_function, self.batch_size
        )
        return molecules


class SMILESLSTMPPOIterator(Generator):
    def __init__(
        self,
        resource_path,
        batch_size: int,
        episode_size: int,
        num_epochs: int,
        optimize_batch_size: int,
        entropy_weight: int,
        kl_div_weight: int,
        clip_param: float,
    ):
        """Initialize SMILESLSTMPPOIterator.

        Args:
            resource_path: path to load the hypothesis, candidate labels and, optionally, the smiles file.
            batch_size: number of molecules to generate.
            episode_size: number of molecules sampled by the policy at the start of a series of ppo updates.
            num_epochs: number of epochs to sample.
            optimize_batch_size: batch size for the optimization.
            entropy_weight: used for calculating entropy loss.
            kl_div_weight: used for calculating Kullback-Leibler divergence loss.
            clip_param: used for determining how far the new policy is from the old one.
        """
        self.resource_path = resource_path
        self.batch_size = batch_size
        self.episode_size = episode_size
        self.num_epochs = num_epochs
        self.optimize_batch_size = optimize_batch_size
        self.entropy_weight = entropy_weight
        self.kl_div_weight = kl_div_weight
        self.clip_param = clip_param
        self.smiles_lstm_ppo_generator: PPODirectedGenerator = None

    def generate_batch(self, target) -> List[Any]:
        """Generate a batch of molecules.

        Args:
            target: condition used for generation.

        Returns:
            the generated molecules.
        """
        score_list, weights = get_target_parameters(target)

        self.scoring_function = CombinedScorer(
            scorer_list=score_list,
            weights=weights,
        )
        if self.smiles_lstm_ppo_generator is None:
            optimiser = SMILESLSTMPPO(
                model_path=os.path.join(self.resource_path, "model_final_0.473.pt"),
                num_epochs=self.num_epochs,
                episode_size=self.episode_size,
                optimize_batch_size=self.optimize_batch_size,
                entropy_weight=self.entropy_weight,
                kl_div_weight=self.kl_div_weight,
                clip_param=self.clip_param,
            )
            logger.info("initialization of the generator")
            self.smiles_lstm_ppo_generator = optimiser.get_generator()

        logger.info("generating molecules")
        molecules = self.smiles_lstm_ppo_generator.generate_optimized_molecules(
            self.scoring_function, self.batch_size
        )
        return molecules


class AaeIterator:
    def __init__(
        self,
        resource_path: str,
        n_samples: int,
        n_batch: int,
        max_len: int,
    ):
        """Initialize AAE.

        Args:
            resource_path: path to load the hypothesis, candidate labels and, optionally, the smiles file.
            n_samples: number of samples to sample.
            n_batch: size of the batch.
            max_len: max length of SMILES.
        """
        self.resource_path = resource_path
        self.model_path = os.path.join(self.resource_path, "model.pt")
        self.config_path = os.path.join(self.resource_path, "config.pt")
        self.vocab_path = os.path.join(self.resource_path, "vocab.pt")
        self.n_samples = n_samples
        self.n_batch = n_batch
        self.max_len = max_len
        self.aae_generator: AaeGenerator = None
        self.device_name = claim_device_name()

    def generate_batch(self, target=None) -> List[Any]:
        """Generate a batch of molecules.

        Args:
            target: condition used for generation.

        Returns:
            the generated molecules.
        """
        if self.aae_generator is None:
            optimiser = AAE(
                model_path=self.model_path,
                model_config_path=self.config_path,
                vocab_path=self.vocab_path,
                n_samples=self.n_samples,
                n_batch=self.n_batch,
                max_len=self.max_len,
                device=self.device_name,
            )
            logger.info("Initialization of the Generator")
            self.aae_generator = optimiser.get_generator()
        molecules = self.aae_generator.generate(self.n_samples)
        return molecules


class VaeIterator:
    def __init__(
        self,
        resource_path: str,
        n_samples: int,
        n_batch: int,
        max_len: int,
    ):
        """Initialize VaeIterator.

        Args:
            resource_path: path to load the hypothesis, candidate labels and, optionally, the smiles file.
            n_samples: number of samples to sample.
            n_batch: size of the batch.
            max_len: max length of SMILES.
        """
        self.resource_path = resource_path
        self.model_path = os.path.join(self.resource_path, "model.pt")
        self.config_path = os.path.join(self.resource_path, "config.pt")
        self.vocab_path = os.path.join(self.resource_path, "vocab.pt")
        self.n_samples = n_samples
        self.n_batch = n_batch
        self.max_len = max_len
        self.vae_generator: VaeGenerator = None
        self.device_name = claim_device_name()

    def generate_batch(self, target=None) -> List[Any]:
        """Generate a batch of molecules.

        Args:
            target: condition used for generation.

        Returns:
            the generated molecules.
        """
        if self.vae_generator is None:
            optimiser = VAE(
                model_path=self.model_path,
                model_config_path=self.config_path,
                vocab_path=self.vocab_path,
                n_samples=self.n_samples,
                n_batch=self.n_batch,
                max_len=self.max_len,
                device=self.device_name,
            )
            logger.info("Initialization of the Generator")
            self.vae_generator = optimiser.get_generator()
        molecules = self.vae_generator.generate(self.n_samples)
        return molecules


class OrganIterator:
    def __init__(
        self,
        resource_path: str,
        n_samples: int,
        n_batch: int,
        max_len: int,
    ):
        """Initialize OrganIterator.

        Args:
            resource_path: path to load the hypothesis, candidate labels and, optionally, the smiles file.
            n_samples: number of samples to sample.
            n_batch: size of the batch.
            max_len: max length of SMILES.
        """
        self.resource_path = resource_path
        self.model_path = os.path.join(self.resource_path, "model.pt")
        self.config_path = os.path.join(self.resource_path, "config.pt")
        self.vocab_path = os.path.join(self.resource_path, "vocab.pt")
        self.n_samples = n_samples
        self.n_batch = n_batch
        self.max_len = max_len
        self.organ_generator: OrganGenerator = None
        self.device_name = claim_device_name()

    def generate_batch(self, target=None) -> List[Any]:
        """Generate a batch of molecules.

        Args:
            target: condition used for generation.

        Returns:
            the generated molecules.
        """
        if self.organ_generator is None:
            optimiser = Organ(
                model_path=self.model_path,
                model_config_path=self.config_path,
                vocab_path=self.vocab_path,
                n_samples=self.n_samples,
                n_batch=self.n_batch,
                max_len=self.max_len,
                device=self.device_name,
            )
            logger.info("Initialization of the Generator")
            self.organ_generator = optimiser.get_generator()
        molecules = self.organ_generator.generate(self.n_samples)
        return molecules
