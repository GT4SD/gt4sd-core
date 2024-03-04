#
# MIT License
#
# Copyright (c) 2024 GT4SD team
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
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import random
import logging
from itertools import product as iter_product
import time
from joblib import load
from .processing import (
    HFandTAPEModelUtility,
    SelectionGenerator,
    CrossoverGenerator,
    sanitize_intervals,
    sanitize_intervals_with_padding,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MutationModelManager:
    """
    Manages and caches mutation models for efficient reuse.
    """

    _models_cache: Dict[Any, Any] = {}

    @staticmethod
    def load_model(embedding_model_path, tokenizer_path, **kwargs):
        """
        Loads or retrieves a model from the cache based on the given paths.

        Args:
            embedding_model_path (str): Path to the embedding model.
            tokenizer_path (str): Path to the tokenizer.
            **kwargs: Additional arguments for model loading.

        Returns:
            An instance of the loaded model.
        """
        model_key = (embedding_model_path, tokenizer_path)

        if model_key in MutationModelManager._models_cache:
            return MutationModelManager._models_cache[model_key]

        model = HFandTAPEModelUtility(embedding_model_path, tokenizer_path, **kwargs)
        MutationModelManager._models_cache[model_key] = model
        return model

    @staticmethod
    def clear_cache():
        """
        Clears the cached models.
        """
        MutationModelManager._models_cache.clear()


class MutationStrategy(ABC):
    """
    Abstract base class for defining mutation strategies.
    """

    @abstractmethod
    def mutate(
        self, sequence: str, num_mutations: int, intervals: List[List[int]]
    ) -> List[str]:
        """Abstract method for mutating a sequence.

        Args:
            sequence (str): The original sequence to be mutated.
            num_mutations (int): The number of mutations to apply.

        Returns:
            List[str]: The mutated sequence.
        """
        pass


class LanguageModelMutationStrategy(MutationStrategy):
    """
    Mutation strategy using a language model.
    """

    def __init__(self, mutation_model):
        """Initializes the mutation strategy with a given model.

        Args:
            mutation_model: The model to be used for mutation.
        """
        self.mutation_model = mutation_model
        self.top_k = 2

    def set_top_k(self, top_k: int):
        """Sets the top k mutations to consider during mutation.

        Args:
            top_k (int): The number of top mutations to consider.
        """
        self.top_k = top_k

    def mutate(
        self, sequence: str, num_mutations: int, intervals: List[List[int]]
    ) -> List[str]:
        """Mutates a sequence within specified intervals using the model.

        Args:
            sequence (str): The original sequence to be mutated.
            num_mutations (int): The number of mutations to introduce.
            intervals (List[List[int]]): Intervals within the sequence
            where mutations are allowed.

        Returns:
            List[str]: A list of mutated sequences.
        """

        flat_intervals = [
            i
            for interval in intervals
            for i in range(interval[0], interval[1] + 1)
            if i < len(sequence)
        ]

        num_mutations = random.randint(1, num_mutations)

        chosen_positions = random.sample(
            flat_intervals, min(num_mutations, len(flat_intervals))
        )
        sequence_list = list(sequence)

        for pos in chosen_positions:
            sequence_list[pos] = self.mutation_model.tokenizer.mask_token

        masked_sequence = " ".join(sequence_list)

        return self.mutation_model.unmask(masked_sequence, self.top_k)


class TransitionMatrixMutationStrategy(MutationStrategy):
    """
    Mutation strategy based on a transition matrix.
    """

    def __init__(self, transition_matrix: str):
        """Initializes the mutation strategy with a transition matrix.

        Args:
            transition_matrix (str): Path to the CSV file containing
            the transition matrix.
        """
        logger.info(" USING TRNASITION MATRIX  ")
        self.transition_matrix = pd.read_csv(
            transition_matrix, index_col=None, header=0
        )
        self.top_k = 2

    def set_top_k(self, top_k: int):
        """Sets the top k mutations to consider during mutation.

        Args:
            top_k (int): The number of top mutations to consider.
        """

        self.top_k = top_k

    def mutate(
        self, sequence: str, num_mutations: int, intervals: List[List[int]]
    ) -> List[str]:
        """Mutates a sequence based on the transition matrix within
        specified intervals.

        Args:
            sequence (str): The original sequence to be mutated.
            num_mutations (int): The number of mutations to introduce.
            intervals (List[List[int]]): Intervals within the sequence
            where mutations are allowed.

        Returns:
            List[str]: A list of mutated sequences.
        """

        flat_intervals = [
            i
            for interval in intervals
            for i in range(interval[0], interval[1] + 1)
            if i < len(sequence)
        ]

        num_mutations = random.randint(1, num_mutations)

        chosen_positions = random.sample(
            flat_intervals, min(num_mutations, len(flat_intervals))
        )

        mutated_sequences = []

        mutation_options = []
        for pos in chosen_positions:
            aa_probabilities = self.transition_matrix.iloc[pos]
            top_mutations = aa_probabilities.nlargest(self.top_k).index.tolist()
            mutation_options.append([(pos, aa) for aa in top_mutations])

        for mutation_combination in iter_product(*mutation_options):
            temp_sequence = list(sequence)
            for pos, new_aa in mutation_combination:
                temp_sequence[pos] = new_aa
            mutated_sequences.append("".join(temp_sequence))

        return mutated_sequences


class MutationFactory:
    """
    Factory class for creating mutation strategies based on configuration.
    """

    @staticmethod
    def get_mutation_strategy(mutation_config: Dict[str, Any]):
        """Retrieves a mutation strategy based on the provided configuration.

        Args:
            mutation_config (Dict[str, Any]): Configuration specifying
            the type of mutation strategy and its parameters.

        Raises:
            KeyError: If required configuration parameters are missing.
            ValueError: If the mutation type is unsupported.

        Returns:
            _type_: An instance of the specified mutation strategy
        """
        if mutation_config["type"] == "language-modeling":
            mutation_model = MutationModelManager.load_model(
                embedding_model_path=mutation_config["embedding_model_path"],
                tokenizer_path=mutation_config["tokenizer_path"],
                unmasking_model_path=mutation_config.get("unmasking_model_path"),
            )
            return LanguageModelMutationStrategy(mutation_model)
        elif mutation_config["type"] == "transition-matrix":
            transition_matrix = mutation_config.get("transition_matrix")
            if transition_matrix is None:
                raise KeyError(
                    "Transition matrix not provided in mutation configuration."
                )
            return TransitionMatrixMutationStrategy(transition_matrix)
        else:
            raise ValueError("Unsupported mutation type")


class SequenceMutator:
    """
    Class for mutating sequences using a specified strategy.
    """

    def __init__(self, sequence: str, mutation_config: Dict[str, Any]):
        """Initializes the mutator with a sequence and a mutation strategy.

        Args:
            sequence (str): The sequence to be mutated.
            mutation_config (Dict[str, Any]): Configuration for
            the mutation strategy.
        """
        self.sequence = sequence
        self.mutation_strategy = MutationFactory.get_mutation_strategy(mutation_config)
        self.top_k = 2

    def set_top_k(self, top_k: int):
        """Sets the number of top mutations to consider in the mutation strategy.

        Args:
            top_k (int): The number of top mutations to consider.
        """
        self.top_k = top_k
        if isinstance(
            self.mutation_strategy,
            (LanguageModelMutationStrategy, TransitionMatrixMutationStrategy),
        ):
            self.mutation_strategy.set_top_k(top_k)

    def get_mutations(
        self,
        num_sequences: int,
        number_of_mutations: int,
        intervals: List[Tuple[int, int]],
        current_population: List[str],
        already_evaluated_sequences: List[str],
    ) -> List[str]:
        """Generates a set of mutated sequences.

        Args:
            num_sequences (int): Number of mutated sequences to generate.
            number_of_mutations (int): Number of mutations to apply to
            each sequence.
            intervals (List[Tuple[int]]): Intervals within the sequence
            where mutations are allowed.
            already_evaluated_sequences (List[str]): List of sequences
            that have already been evaluated.

        Returns:
            List[str]: A list of mutated sequences.
        """
        max_mutations = min(len(self.sequence), number_of_mutations)
        if len(current_population) < 1:
            current_population.append(self.sequence)

        random.shuffle(current_population)
        mutated_sequences_set: List[str] = []

        while len(mutated_sequences_set) < num_sequences:
            for temp_sequence in current_population:
                new_mutations = self.mutation_strategy.mutate(
                    temp_sequence, max_mutations, intervals
                )
                mutated_sequences_set.extend(new_mutations)
                if len(mutated_sequences_set) >= num_sequences:
                    break
        return random.sample(mutated_sequences_set, num_sequences)


class EnzymeOptimizer:
    """
    Optimizes protein sequences based on interaction with
    substrates and products.
    """

    def __init__(
        self,
        sequence: str,
        protein_model: HFandTAPEModelUtility,
        substrate_smiles: str,
        product_smiles: str,
        chem_model_path: str,
        chem_tokenizer_path: str,
        scorer_filepath: str,
        mutator: SequenceMutator,
        intervals: List[Tuple[int, int]],
        batch_size: int = 2,
        seed: int = 123,
        top_k: int = 2,
        selection_ratio: float = 0.5,
        perform_crossover: bool = False,
        crossover_type: str = "uniform",
        minimum_interval_length: int = 8,
        pad_intervals: bool = False,
        concat_order=["sequence", "substrate", "product"],
    ):
        """Initializes the optimizer with models, sequences, and
        optimization parameters.


        Args:
            sequence (str): The initial protein sequence.
            protein_model (HFandTAPEModelUtility): Model for protein embeddings.
            substrate_smiles (str): SMILES representation of the substrate.
            product_smiles (str): SMILES representation of the product.
            chem_model_path (str): Path to the chemical model.
            chem_tokenizer_path (str): Path to the chemical tokenizer.
            scorer_filepath (str): Path to the scoring model.
            mutator (SequenceMutator): The mutator for generating sequence variants.
            intervals (List[Tuple[int, int]]): Intervals for mutation.
            batch_size (int, optional): The number of sequences to process in one batch. Defaults to 2.
            seed (int, optional): Random seed. Defaults to 123.
            top_k (int, optional): Number of top mutations to consider. Defaults to 2.
            selection_ratio (float, optional): Ratio of sequences to select after scoring. Defaults to 0.5.
            perform_crossover (bool, optional): Flag to perform crossover operation. Defaults to False.
            crossover_type (str, optional): Type of crossover operation. Defaults to "uniform".
            minimum_interval_length (int, optional): Minimum length of mutation intervals. Defaults to 8.
            pad_intervals (bool, optional): Flag to pad the intervals. Defaults to False.
            concat_order (list, optional): Order of concatenating embeddings. Defaults to ["sequence", "substrate", "product"].
        """
        self.sequence = sequence
        self.protein_model = protein_model
        self.mutator = mutator
        self.intervals = intervals
        self.batch_size = batch_size
        self.top_k = top_k
        self.selection_ratio = selection_ratio
        self.perform_crossover = perform_crossover
        self.crossover_type = crossover_type
        self.concat_order = concat_order
        self.minimum_interval_length = minimum_interval_length
        self.pad_intervals = pad_intervals
        self.mutator.set_top_k(top_k)
        self.concat_order = concat_order
        self.scorer = load(scorer_filepath)
        self.seed = seed

        self.chem_model = HFandTAPEModelUtility(chem_model_path, chem_tokenizer_path)
        self.substrate_embedding = self.chem_model.embed([substrate_smiles])[0]
        self.product_embedding = self.chem_model.embed([product_smiles])[0]

        self.selection_generator = SelectionGenerator()
        self.crossover_generator = CrossoverGenerator()

        if intervals is None:
            self.intervals = [(0, len(sequence))]
        else:
            self.intervals = sanitize_intervals(intervals)
            if pad_intervals:
                self.intervals = sanitize_intervals_with_padding(
                    self.intervals, minimum_interval_length, len(sequence)
                )

        random.seed(self.seed)

    def optimize(
        self,
        num_iterations: int,
        num_sequences: int,
        num_mutations: int,
        time_budget: Optional[int] = 360,
    ):
        """Runs the optimization process over a specified number
        of iterations.

        Args:
            num_iterations (int): Number of iterations to run
            the optimization.
            num_sequences (int): Number of sequences to generate
            per iteration.
            num_mutations (int): Max number of mutations to apply.
            time_budget (Optional[int]): Time budget for
            optimizer (in seconds). Defaults to 360.

        Returns:
            A tuple containing the list of all sequences and
            iteration information.
        """

        iteration_info = {}

        scored_original_sequence = self.score_sequence(self.sequence)
        original_sequence_score_ = scored_original_sequence["score"]

        logger.info(f"Original sequence score: {original_sequence_score_}")

        all_mutated_sequences: List[str] = [scored_original_sequence["sequence"]]
        current_best_score = original_sequence_score_

        all_scored_sequences: List[Dict[str, Any]] = []

        for iteration in range(num_iterations):
            start_time = time.time()

            scored_sequences: List[Dict[str, Any]] = [scored_original_sequence]

            if iteration == 0:
                current_population: List[str] = [self.sequence]
                if len(current_population) < num_sequences:
                    while len(current_population) < num_sequences:
                        new_mutants = self.mutator.mutation_strategy.mutate(
                            self.sequence, num_mutations, self.intervals
                        )
                        for mut in new_mutants:
                            if mut not in all_mutated_sequences:
                                current_population.append(mut)
                            else:
                                continue
                        if len(current_population) >= num_sequences:
                            break

                if len(current_population) >= num_sequences:
                    random.shuffle(current_population)
                    current_population = random.sample(
                        current_population, k=num_sequences
                    )

            logger.info(
                f"Number of sequences in current population: {len(current_population)}"
            )

            iteration_scored_sequences = []
            for _ in range(0, len(current_population), self.batch_size):
                scored_sequences = self.score_sequences(
                    current_population[_ : _ + self.batch_size]
                )
                all_mutated_sequences.extend(
                    current_population[_ : _ + self.batch_size]
                )
                all_scored_sequences.extend(scored_sequences)
                iteration_scored_sequences.extend(scored_sequences)

            if self.selection_ratio < 1.0:

                samples_with_higher_score = [
                    i
                    for i in iteration_scored_sequences
                    if i["score"] > original_sequence_score_
                ]
                selected_sequences = self.selection_generator.selection(
                    samples_with_higher_score, self.selection_ratio
                )
            else:
                selected_sequences = iteration_scored_sequences

            offspring_sequences = []
            if self.perform_crossover and len(selected_sequences) > 1:
                for i in range(0, len(selected_sequences), 2):
                    if i + 1 < len(selected_sequences):
                        parent1 = selected_sequences[i]["sequence"]
                        parent2 = selected_sequences[i + 1]["sequence"]
                        if self.crossover_type == "single_point":
                            (
                                offspring1,
                                offspring2,
                            ) = self.crossover_generator.sp_crossover(parent1, parent2)
                        else:
                            (
                                offspring1,
                                offspring2,
                            ) = self.crossover_generator.uniform_crossover(
                                parent1, parent2
                            )
                        offspring_sequences.extend([offspring1, offspring2])

            logger.info(f"Selected samples: {len(selected_sequences)}")
            logger.info(f"Number Crossed-Over samples: {len(offspring_sequences)}")

            current_population = [
                seq["sequence"] for seq in selected_sequences
            ] + offspring_sequences

            if len(current_population) < num_sequences:
                while len(current_population) < num_sequences:
                    current_population.extend(
                        self.mutator.mutation_strategy.mutate(
                            self.sequence, num_mutations, self.intervals
                        )
                    )
                    if len(current_population) >= num_sequences:
                        break

            if len(current_population) >= num_sequences:
                random.shuffle(current_population)
                current_population = current_population[:num_sequences]

            higher_scoring_sequences = 0
            for temp_seq in iteration_scored_sequences:
                if temp_seq["score"] > current_best_score:
                    current_best_score = temp_seq["score"]
                    higher_scoring_sequences += 1

            end_time = time.time()
            elapsed_time = end_time - start_time
            iteration_info[iteration + 1] = {
                "Iteration": iteration + 1,
                "best_score": current_best_score,
                "higher_scoring_sequences": higher_scoring_sequences,
                "elapsed_time": elapsed_time,
            }
            logger.info(
                f" Iteration {iteration + 1}: Best Score: {current_best_score},"
                f" Higher Scoring Sequences: {higher_scoring_sequences}, "
                f" Time: {elapsed_time} seconds,"
                f" Population length : {len(current_population)}"
            )
            if time_budget is not None and elapsed_time > time_budget:
                logger.warning(f"Used all the given time budget of {time_budget}s")
                break

        all_scored_sequences = sorted(
            all_scored_sequences, key=lambda x: x["score"], reverse=True
        )

        df = pd.DataFrame(all_scored_sequences)
        df = df.drop_duplicates()

        all_scored_sequences = df.to_dict(orient="records")

        return all_scored_sequences, iteration_info

    def score_sequence(self, sequence: str) -> Dict[str, Any]:
        """Scores a single protein sequence.

        Args:
            sequence (str): The protein sequence to score.

        Returns:
            Dict[str, Any]: The score of the sequence.
        """
        sequence_embedding = self.protein_model.embed([sequence])[0]
        embeddings = [
            sequence_embedding,
            self.substrate_embedding,
            self.product_embedding,
        ]
        ordered_embeddings = [
            embeddings[self.concat_order.index(item)] for item in self.concat_order
        ]
        combined_embedding = np.concatenate(ordered_embeddings)
        combined_embedding = combined_embedding.reshape(1, -1)

        score = self.scorer.predict_proba(combined_embedding)[0][1]
        return {"sequence": sequence, "score": score}

    def score_sequences(self, sequences: List[str]) -> List[Dict[str, float]]:
        """Scores a list of protein sequences.

        Args:
            sequences (List[str]): The list of protein sequences to score.

        Returns:
            List[Dict[str, float]]: A list of dictionaries
            containing sequences and their scores.
        """
        sequence_embeddings = self.protein_model.embed(sequences)

        output = []
        for position in range(len(sequence_embeddings)):
            sequence_embedding = sequence_embeddings[position]
            embeddings = [
                sequence_embedding,
                self.substrate_embedding,
                self.product_embedding,
            ]
            ordered_embeddings = [
                embeddings[self.concat_order.index(item)] for item in self.concat_order
            ]
            combined_embedding = np.concatenate(ordered_embeddings)
            combined_embedding = combined_embedding.reshape(1, -1)

            score = self.scorer.predict_proba(combined_embedding)[0][1]
            output.append({"sequence": sequences[position], "score": score})

        return output
