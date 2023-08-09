#
# MIT License
#
# Copyright (c) 2023 GT4SD team
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
import json
import random
import time
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
import numpy as np
from joblib import load
from loguru import logger

from .processing import (
    AutoModelFromHFEmbedding,
    CrossoverGenerator,
    SelectionGenerator,
    StringEmbedding,
    Unmasker,
    reconstruct_sequence_with_mutation_range,
    sanitize_intervals,
    sanitize_intervals_with_padding,
)

#: Transition matrix representation
TransitionMatrix = MutableMapping[str, MutableMapping[str, float]]
#: Transition matrix configuration
TransitionConfiguration = MutableMapping[
    str, Union[MutableMapping[str, float], Sequence[str]]
]

#: Supported features
SUPPORTED_FEATURE_SET: Set[str] = {"substrate", "product", "sequence"}

#: IUPAC code mapping
IUPAC_CODES = OrderedDict(
    [
        ("Ala", "A"),
        ("Asx", "B"),  # Aspartate or Asparagine
        ("Cys", "C"),
        ("Asp", "D"),
        ("Glu", "E"),
        ("Phe", "F"),
        ("Gly", "G"),
        ("His", "H"),
        ("Ile", "I"),
        ("Lys", "K"),
        ("Leu", "L"),
        ("Met", "M"),
        ("Asn", "N"),
        ("Pyl", "O"),  # Pyrrolysin
        ("Pro", "P"),
        ("Gln", "Q"),
        ("Arg", "R"),
        ("Ser", "S"),
        ("Thr", "T"),
        ("Sec", "U"),  # Selenocysteine
        ("Val", "V"),
        ("Trp", "W"),
        ("Xaa", "X"),  # Any AA
        ("Tyr", "Y"),
        ("Glx", "Z"),  # Glutamate or Glutamine
    ]
)
#: IUPAC character set
IUPAC_CHARACTER_SET: Set[str] = set(IUPAC_CODES.values())
#: IUPAC uniform mutation mapping
IUPAC_MUTATION_MAPPING: TransitionConfiguration = {
    iupac_character: sorted(list(IUPAC_CHARACTER_SET - {iupac_character, "X"}))
    for iupac_character in IUPAC_CHARACTER_SET
}


class Mutations:
    """Mutations definition class."""

    def __init__(self, transition_configuration: Dict[str, Any]) -> None:
        """Generate the mutation given the configuration for the transitions.

        Args:
            transition_configuration: transition configuration.
        """
        self.transition_matrix = Mutations.transition_configuration_to_matrix(
            transition_configuration
        )

    @staticmethod
    def transition_configuration_to_matrix(
        transition_configuration: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """Transform a configuration into a valid transition matrix.

        Args:
            transition_configuration: transition configuration.

        Returns:
            A transition matrix.
        """
        transition_matrix: Dict[str, Dict[str, float]] = dict()
        for transition_source, transition_targets in transition_configuration.items():
            if isinstance(transition_targets, dict):
                total = float(sum(transition_targets.values()))
                transition_matrix[transition_source] = {
                    transition_target: transtion_element / total
                    for transition_target, transtion_element in transition_targets.items()
                }
            else:
                transition_matrix[transition_source] = {
                    transition_target: 1 / len(transition_targets)
                    for transition_target in transition_targets
                }
        return transition_matrix

    @staticmethod
    def from_json(filepath: str) -> "Mutations":
        """Parse the mutation from a JSON containing the transition configuration.

        Returns:
            The mutations object.
        """
        with open(filepath) as fp:
            return Mutations(json.load(fp))

    def mutate(self, source: str) -> str:
        """Mutate a source string.

        Args:
            source: Source string.

        Returns:
            The mutated target.
        """
        targets, probabilities = zip(*self.transition_matrix[source].items())
        return np.random.choice(targets, size=1, p=probabilities).item()


class MutationGenerator:
    def __init__(self, sequence: str) -> None:
        self.sequence = sequence

    def get_mutations(self, number_of_mutated_sequences: int = 1) -> List[str]:
        raise NotImplementedError("Implement the method in a sub-class")


class MutationLanguageModel:
    def __init__(self, mutation_model_parameters: Dict[str, str]) -> None:
        """Load language model for mutation suggestion

        Args:
            mutation_model_parameters: Example: { "model_path" : "facebook/esm2_t33_650M_UR50D",
            "tokenizer_path" : "facebook/esm2_t33_650M_UR50D" }.
        """
        self.load_mutation_model = Unmasker(mutation_model_parameters)


class MutationGeneratorLanguageModeling(MutationGenerator):
    def __init__(
        self,
        sequence: str,
        mutation_object: MutationLanguageModel,
        top_k: int = 2,
        maximum_number_of_mutations: int = 4,
    ) -> None:
        """Language model mutations generator

        Args:
            sequence (str): An amino acid sequence.
            mutation_object (MutationLanguageModel): A mutation object.
            top_k (int): Number of alternatives for each replacement. Defaults to 2.
            maximum_number_of_mutations (int): Maximum number of mutations. Defaults to 4.
        """
        super().__init__(sequence)
        self.sequence_length = len(sequence)
        self.mutation_object = mutation_object
        self.top_k = top_k
        self.maximum_number_of_mutations = maximum_number_of_mutations

    def get_mutations(self, number_of_mutated_sequences: int = 1) -> List[str]:
        """Get mutations.

        Args:
            number_of_mutated_sequences (int): Number of mutated sequences to return. Defaults to 1.

        Returns:
            List[str]: Mutated sequence(s).
        """
        output: List[str] = []
        while len(output) < number_of_mutated_sequences:

            tmp_sequence = list(self.sequence)

            number_of_mutations = random.randint(1, self.maximum_number_of_mutations)
            positions = sorted(
                random.sample(range(self.sequence_length), number_of_mutations)
            )

            for pos in positions:
                tmp_sequence[
                    pos
                ] = self.mutation_object.load_mutation_model.tokenizer.mask_token

            tmp_masked_sequence = " ".join(tmp_sequence)

            replacement_lst = self.mutation_object.load_mutation_model.unmask(
                tmp_masked_sequence, self.top_k
            )

            if len(positions) > 1:
                replacement_lst = list(map(list, zip(*replacement_lst)))

                for replacement in replacement_lst:
                    tmp_tokenized_sequence = list(self.sequence)
                    for indx in range(len(replacement)):
                        tmp_tokenized_sequence[positions[indx]] = replacement[indx]
                    output.append("".join(tmp_tokenized_sequence))

            else:
                for i in range(len(replacement_lst[0])):
                    tmp_tokenized_sequence = list(self.sequence)
                    tmp_tokenized_sequence[positions[0]] = replacement_lst[0][i]
                    output.append("".join(tmp_tokenized_sequence))

        return output


class MutationGeneratorTransitionMatrix(MutationGenerator):
    def __init__(
        self,
        sequence: str,
        mutation_object: Mutations,
        maximum_number_of_mutations: int = 4,
    ) -> None:
        """Transition matrix mutations generator.

        Args:
            sequence (str): An amino acid sequence.
            mutation_object (Mutations): A mutation object.
            maximum_number_of_mutations (int): Maximum number of mutations. Defaults to 4.
        """
        super().__init__(sequence)
        self.sequence_length = len(sequence)
        self.mutation_object = mutation_object
        self.maximum_number_of_mutations = maximum_number_of_mutations

    def get_single_sequence_with_mutations(self) -> str:
        """Mutate the sequence.

        Returns:
            str: Mutated sequence.
        """
        number_of_mutations = random.randint(1, self.maximum_number_of_mutations)
        positions = sorted(
            random.sample(range(self.sequence_length), number_of_mutations)
        )
        mutated_sequence = ""
        start_position = -1
        for position in positions:
            mutated_sequence += self.sequence[(start_position + 1) : position]
            mutated_sequence += self.mutation_object.mutate(self.sequence[position])
            start_position = position
        mutated_sequence += self.sequence[(start_position + 1) :]
        return mutated_sequence

    def get_mutations(self, number_of_mutated_sequences: int = 1) -> List[str]:
        """Generate mutations.

        Args:
            number_of_mutated_sequences (int): Number of sequences to return. Defaults to 1.

        Returns:
            List[str]: Mutated sequence(s).
        """
        return [
            self.get_single_sequence_with_mutations()
            for _ in range(number_of_mutated_sequences)
        ]


class MutationGeneratorFactory:
    @staticmethod
    def get_mutation_generator(
        mutation_object: Union[Mutations, MutationLanguageModel],
        sequence: str,
        **mutation_object_parameters: int,
    ) -> MutationGenerator:
        """Get an instance of a mutation generator based on the mutation object.

        Args:
            mutation_object (Union[Mutations, MutationLanguageModel]): Mutation object.
            sequence (str): Amino acid sequence.
            **mutation_object_parameters: Parameters for mutation object.

        Returns:
            MutationGenerator: Instance of mutation generator.
        """
        mutation_generator: MutationGenerator
        if isinstance(mutation_object, MutationLanguageModel):
            mutation_generator = MutationGeneratorLanguageModeling(
                sequence=sequence,
                mutation_object=mutation_object,
                top_k=mutation_object_parameters["top_k"],
                maximum_number_of_mutations=mutation_object_parameters[
                    "maximum_number_of_mutations"
                ],
            )
        elif isinstance(mutation_object, Mutations):
            mutation_generator = MutationGeneratorTransitionMatrix(
                sequence=sequence,
                mutation_object=mutation_object,
                maximum_number_of_mutations=mutation_object_parameters[
                    "maximum_number_of_mutations"
                ],
            )
        else:
            raise ValueError(
                f"Mutations with type: {type(mutation_object)} not supported!"
            )
        return mutation_generator


SUPPORTED_FEATURE_SET = {"substrate", "sequence", "product"}
MUTATION_GENERATORS: Dict[str, Type[MutationGenerator]] = {
    "transition-matrix": MutationGeneratorTransitionMatrix,
    "language-modeling": MutationGeneratorLanguageModeling,
}

CROSSOVER_GENERATOR: Dict[str, Callable[[str, str, float], Tuple[str, str]]] = {
    "single_point": lambda a_sequence, another_sequence, _: CrossoverGenerator().single_point_crossover(
        a_sequence, another_sequence
    ),
    "uniform": lambda a_sequence, another_sequence, probability: CrossoverGenerator(
        probability
    ).uniform_crossover(a_sequence, another_sequence),
}

SELECTION_GENERATOR: Dict[str, Callable[[Any, Any], List[Any]]] = {
    "generic": lambda scores, k: SelectionGenerator().selection(scores, k)
}


class Scorer:
    def __init__(self, scorer_filepath: str):
        self.scorer_filepath = scorer_filepath
        self.scorer = load(scorer_filepath)

    def predict_proba(self, feature_vector):
        return self.scorer.predict_proba(feature_vector)


class EnzymeOptimizer:
    """Optimize an enzyme to catalyze a reaction from substrate to product."""

    def __init__(
        self,
        scorer_filepath: str,
        substrate: str,
        product: str,
        sequence: str,
        protein_embedding: StringEmbedding = AutoModelFromHFEmbedding(
            model_kwargs={
                "model_path": "facebook/esm2_t33_650M_UR50D",
                "tokenizer_path": "facebook/esm2_t33_650M_UR50D",
                "cache_dir": "/dccstor/yna/.cache/",
            }
        ),
        molecule_embedding: StringEmbedding = AutoModelFromHFEmbedding(
            model_kwargs={
                "model_path": "seyonec/ChemBERTa-zinc-base-v1",
                "tokenizer_path": "seyonec/ChemBERTa-zinc-base-v1",
                "cache_dir": "/dccstor/yna/.cache/",
            }
        ),
        ordering: List[str] = ["substrate", "sequence", "product"],
    ) -> None:
        """Initialize the enzyme designer.

        Args:
            scorer_filepath (str): Pickled scorer filepath.
            substrate (str): Substrate SMILES.
            product (str): Product SMILES.
            sequence (str): AA sequence representing the enzyme to optimize.
            protein_embedding (StringEmbedding, optional): Protein embedding class.
            Defaults to esm2_t33_650M_UR50D.
            molecule_embedding (StringEmbedding, optional): Molecule embedding class.
            Defaults to ChemBERTa version 1.
            ordering (List[str], optional): Ordering of the features for the scorer.
            Defaults to ["substrate", "product", "sequence"].

        Raises:
            ValueError: Ordering provided is not feasible.
        """
        if len(set(ordering).intersection(SUPPORTED_FEATURE_SET)) < 3:
            raise ValueError(
                f"Ordering={ordering} should contain only the three admissible values: {sorted(list(SUPPORTED_FEATURE_SET))}"
            )
        else:
            self._ordering = ordering
        self.scorer_filepath = scorer_filepath
        self.scorer = Scorer(scorer_filepath)
        self.substrate = substrate
        self.product = product

        self.protein_embedding = protein_embedding
        self.molecule_embedding = molecule_embedding
        self.embedded_vectors = {
            "substrate": self.molecule_embedding.embed([self.substrate]),
            "product": self.molecule_embedding.embed([self.product]),
        }
        self.sequence = sequence
        self.sequence_length = len(sequence)

    def extract_fragment_embedding(
        self, sequence: str, intervals: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Extract the embeddings for each fragment in a sequence.

        Args:
            sequence (str): A sequence from which to extract the fragments.
            intervals (List[Tuple[int, int]]): List of ranges in the sequence, zero-based.
            The same interval is applied to all sequences.

        Returns:
            np.ndarray: The mean embedding of the input sequence based on the intervals.
        """
        fragments: List[str] = []
        for start, end in intervals:
            size_fragment = end - start
            fragments.append("".join(sequence[:size_fragment]))
            sequence = sequence[size_fragment:]
        sequence_embedding = np.array(
            [self.protein_embedding.embed([fragment]) for fragment in fragments]
        )
        sequence_embedding = (
            sequence_embedding / np.linalg.norm(sequence_embedding)
        ).mean(axis=0)

        return sequence_embedding

    def score_sequence(
        self,
        sequence: str,
        intervals: Optional[List[Tuple[int, int]]] = None,
        fragment_embeddings: Optional[bool] = False,
    ) -> float:
        """Score a given sequence.

        Args:
            sequence (str): A sequence to score.
            intervals (Optional[List[Tuple[int, int]]], optional): List of ranges in the sequence, zero-based.
            fragment_embeddings (Optional[bool], optional): Set to True for fragment embeddings. Defaults to False.

        Returns:
            float: The score of the input sequence.
        """
        if fragment_embeddings and intervals is not None:
            embedded_vectors = {
                "sequence": self.extract_fragment_embedding(sequence, intervals)
            }
        else:
            embedded_vectors = {"sequence": self.protein_embedding.embed([sequence])}
        embedded_vectors.update(self.embedded_vectors)
        feature_vector = np.concatenate(
            [embedded_vectors[feature] for feature in self._ordering], axis=1
        )
        return self.scorer.predict_proba(feature_vector)[0][1]

    def score_sequences(
        self,
        sequences: List[str],
        intervals: Optional[List[Tuple[int, int]]] = None,
        fragment_embeddings: Optional[bool] = False,
    ) -> List[Dict[str, Any]]:
        """Score a given list of sequences.

        Args:
            sequences (List[str]): A list of sequences to score.
            intervals (Optional[List[Tuple[int, int]]], optional): List of ranges in the sequence, zero-based.
            fragment_embeddings (Optional[bool], optional): Set to True for fragment embeddings. Defaults to False.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing sequence-score pairs.
        """
        number_of_sequences = len(sequences)
        embedded_matrices = {
            "substrate": np.repeat(
                self.embedded_vectors["substrate"], number_of_sequences, axis=0
            ),
            "product": np.repeat(
                self.embedded_vectors["product"], number_of_sequences, axis=0
            ),
        }

        if fragment_embeddings and intervals is not None:
            embeddings = []
            for sequence in sequences:
                embeddings.append(self.extract_fragment_embedding(sequence, intervals))
            embedded_matrices["sequence"] = np.array(embeddings)
        else:
            embedded_matrices["sequence"] = self.protein_embedding.embed(sequences)
        feature_vector = np.concatenate(
            [embedded_matrices[feature] for feature in self._ordering], axis=1
        )
        return [
            {"sequence": sequence, "score": score}
            for sequence, score in zip(
                sequences, self.scorer.predict_proba(feature_vector)[:, 1]
            )
        ]

    def sequence_generation(
        self,
        sequence_from_intervals: str,
        mutation_object: MutationGenerator,
        initial_population: List[str] = [],
        number_of_samples_to_generate: int = 1,
    ) -> List[str]:
        """Generate sequences.

        Args:
            sequence_from_intervals (str): Original sequence extracted from intervals.
            mutation_object (MutationGenerator): A mutation object.
            initial_population (List[str], optional): List of initial samples. Defaults to [].
            number_of_samples_to_generate (int, optional): Number of samples to generate. Defaults to 1.

        Returns:
            List[str]: A list of generated sequences.
        """
        lst_mutated_sequences: List[str] = []

        if len(initial_population) >= 1:
            for initial_sequence in initial_population:
                mutation_object.sequence = initial_sequence
                lst_mutated_sequences += mutation_object.get_mutations()

            if number_of_samples_to_generate > len(lst_mutated_sequences):
                mutation_object.sequence = sequence_from_intervals
                lst_mutated_sequences += mutation_object.get_mutations(
                    number_of_mutated_sequences=number_of_samples_to_generate
                    - len(lst_mutated_sequences)
                )

        else:
            lst_mutated_sequences += mutation_object.get_mutations(
                number_of_mutated_sequences=number_of_samples_to_generate
            )

        return list(set(lst_mutated_sequences))

    def sequence_evaluation(
        self,
        original_sequence_score: float,
        current_best_score: float,
        mutated_sequences_range: List[str],
        visited_sequences: set,
        intervals: List[Tuple[int, int]],
        batch_size: Optional[int] = None,
    ) -> Tuple[Set[Any], List[Dict[str, Any]], List[Dict[str, Any]], float]:
        """Evaluate sequences.

        Args:
            original_sequence_score (float): Score of the original sequence.
            current_best_score (float): Score of the current best sequence.
            mutated_sequences_range (List[str]): List of mutated sequences (concatenated fragments of sequences).
            visited_sequences (set): Set of sequences already evaluated in past optimization steps.
            intervals (List[Tuple[int, int]]): List of ranges in the sequence, zero-based.
            batch_size (Optional[int], optional): Number of sequences to evaluate in one round.

        Returns:
            Tuple[Set[Any], List[Dict[str, Any]], List[Dict[str, Any]], float]: A tuple containing visited sequences, temporary results, temporary results for selection, and the current best score.
        """
        temporary_results: List[Dict[str, Any]] = []
        temporary_results_for_selection: List[Dict[str, Any]] = []

        if not batch_size:
            batch_size = 8

        lst_mutated_sequences: List[str] = []

        for mutated_fragments in mutated_sequences_range:
            mutated_sequence = reconstruct_sequence_with_mutation_range(
                sequence=self.sequence,
                mutated_sequence_range=mutated_fragments.strip(),
                intervals=intervals,
            )

            if mutated_sequence not in visited_sequences:
                visited_sequences.add(mutated_sequence)
                lst_mutated_sequences.append(mutated_sequence)

        for i in range(0, len(lst_mutated_sequences), batch_size):
            for scored_sequence in self.score_sequences(
                lst_mutated_sequences[i : i + batch_size]
            ):
                if scored_sequence["score"] > original_sequence_score:
                    temporary_results_for_selection.append(scored_sequence)
                    temporary_results.append(scored_sequence)
                    if scored_sequence["score"] > current_best_score:
                        current_best_score = scored_sequence["score"]
                else:
                    temporary_results.append(scored_sequence)

        return (
            visited_sequences,
            temporary_results,
            temporary_results_for_selection,
            current_best_score,
        )

    def selection_crossover(
        self,
        sequence_from_intervals: str,
        tmp_results: List[Dict[str, Any]],
        intervals: List[Tuple[int, int]],
        selection_method: str,
        crossover_method: str,
        crossover_probability: float = 0.5,
        top_k_selection: Optional[int] = -1,
    ) -> List[str]:
        """Perform Selection and Crossover.

        Args:
            sequence_from_intervals (str): Original sequence extracted from intervals.
            tmp_results (List[Dict[str, Any]]): The temporary results.
            intervals (List[Tuple[int, int]]): List of ranges in the sequence, zero-based.
            selection_method (str): Methodology used for selection.
            crossover_method (str): Methodology used for crossover.
            crossover_probability (float, optional): Crossover probability. Used in case Uniform crossover is selected. Defaults to 0.5.
            top_k_selection (Optional[int], optional): Number of samples to select. Defaults to -1.

        Returns:
            List[str]: New samples for the next round of optimization.
        """
        crossover: Optional[
            Callable[[str, str, float], Tuple[str, str]]
        ] = CROSSOVER_GENERATOR.get(crossover_method)
        selection: Optional[Callable[[Any, Any], List[Any]]] = SELECTION_GENERATOR.get(
            selection_method
        )

        if crossover is None:
            raise ValueError(f"Invalid crossover method: {crossover_method}")
        if selection is None:
            raise ValueError(f"Invalid selection method: {selection_method}")

        selected_children = selection(tmp_results, top_k_selection)

        children: List[str] = []
        for pos in range(len(selected_children)):
            selected_child_mutated_fragments = "".join(
                [self.sequence[start:end] for start, end in intervals]
            )

            if crossover_method == "uniform":
                new_child_1, new_child_2 = crossover(
                    selected_child_mutated_fragments,
                    sequence_from_intervals,
                    crossover_probability,
                )
            else:
                new_child_1, new_child_2 = crossover(
                    selected_child_mutated_fragments, sequence_from_intervals, 1
                )

            children.append(new_child_1)
            children.append(new_child_2)

        return list(set(children))

    def optimize(
        self,
        number_of_mutations: int,
        intervals: List[Tuple[int, int]] = None,
        number_of_steps: int = 10,
        batch_size: int = 8,
        full_sequence_embedding: bool = True,
        number_of_sequences: Optional[int] = None,
        seed: int = 42,
        minimum_interval_length: int = 8,
        time_budget: Optional[int] = None,
        mutation_generator: Optional[str] = "transition-matrix",
        mutation_generator_parameters: Dict[str, Any] = {
            "mutation_object": Mutations,
            "maximum_number_of_mutations": 4,
        },
        top_k: Optional[int] = 2,
        pad_intervals: Optional[bool] = False,
        population_per_iteration: Optional[int] = None,
        with_genetic_algorithm: Optional[bool] = False,
        selection_method: str = "generic",
        top_k_selection: Optional[int] = None,
        crossover_method: str = "single_point",
        crossover_probability: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Optimize the enzyme given a number of mutations and a range.
        If the range limits are not provided, the full sequence is optimized, which might be inefficient.
        The sampling is performed by exploring mutations with a slightly smart random sampling.

        Args:
            number_of_mutations (int): Number of allowed mutations.
            intervals (List[Tuple[int, int]], optional): List of ranges in the sequence, zero-based. Defaults to None (optimize the full sequence).
            number_of_steps (int, optional): Number of optimization steps. Defaults to 10.
            batch_size (int, optional): Number of sequences to embed together. Defaults to 8.
            full_sequence_embedding (bool, optional): Perform embeddings with respect to the full sequence. Defaults to True. False = respect to intervals fragments.
            number_of_sequences (Optional[int], optional): Number of optimal sequences to return. Defaults to None (returns all).
            seed (int, optional): Seed for random number generation. Defaults to 42.
            minimum_interval_length (int, optional): Minimum length per interval in case full_sequence_embedding=True. Defaults to 8.
            time_budget (Optional[int], optional): Maximum allowed runtime in seconds. Defaults to None (no time limit, running for number_of_steps steps).
            mutation_generator (str, optional): Type of mutation generation. Defaults to "transition-matrix".
            mutation_generator_parameters (Dict[str, Any], optional): Mutation generation parameters. Defaults to uniform sampling of IUPAC AAs.
            top_k (Optional[int], optional): How many suggested AAs to accept. Defaults to top 2.
            pad_intervals (Optional[bool], optional): If True, in case a fragment of sequence has length < 8: it's padded to a length of at least 8.
            population_per_iteration (Optional[int], optional): Number of sample sequences per optimization step.
            with_genetic_algorithm (Optional[bool], optional): Optimize using a genetic algorithm.
            selection_method (str, optional): Methodology used for selection in case of genetic algorithm optimization. Defaults to "generic".
            top_k_selection (Optional[int], optional): Number of suggested mutants per amino acid to consider.
            crossover_method (str, optional): Crossover method selection. Options are "single_point" or "uniform". Defaults to "single_point".
            crossover_probability (float, optional): Crossover probability in case of uniform crossover method. Defaults to 0.5.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing a candidate optimal sequence and the related score. Sorted from best to worst.
            Note that when no limit on the returned number of sequences is set, the worst sequence is the original unmutated sequence.
            If the optimization fails, only the original sequence is returned.
        """

        random.seed(seed)
        # check if interval is None. In case it is, take the interval as the whole sequence
        if intervals is None:
            intervals = [(0, self.sequence_length)]
        else:
            intervals = sanitize_intervals(
                intervals
            )  # here we merge and sort the intervals

        # pad the sequences to a minimum length of 8
        if pad_intervals:
            intervals = sanitize_intervals_with_padding(
                intervals=intervals,
                max_value=self.sequence_length,
                pad_value=minimum_interval_length,
            )

        # check that the intervals are within the range of the sequence length
        if intervals[-1][1] > self.sequence_length:
            raise ValueError(
                "Check provided intervals; at least one interval is larger than the sequence length."
            )

        # mutate the sequence based on intervals
        self.maximum_number_of_mutations = number_of_mutations

        if self.maximum_number_of_mutations > self.sequence_length:
            logger.warning(
                f"Resetting maximum number of mutations ({self.maximum_number_of_mutations}), as it is higher than sequence length: {self.sequence_length}"
            )
            self.maximum_number_of_mutations = self.sequence_length
        if self.maximum_number_of_mutations < 1:
            logger.warning(
                f"Maximum number of mutations cannot be lower than 1 ({self.maximum_number_of_mutations}), resetting to 1."
            )
            self.maximum_number_of_mutations = 1

        logger.info(
            f"Maximum number of mutations for the intervals: {self.maximum_number_of_mutations}"
        )

        # Check if population size is set
        if not population_per_iteration:
            population_per_iteration = batch_size

        # Create a sequence based on the intervals
        sequence_from_intervals = "".join(
            [self.sequence[start:end] for start, end in intervals]
        )

        if isinstance(mutation_generator, str):
            mutation_generator_type = MUTATION_GENERATORS[mutation_generator]
        else:
            raise ValueError(
                f"Mutation generator with type: {type(mutation_generator_type)} not supported!"
            )

        if mutation_generator == "language-modeling":
            mutation_generator_parameters["top_k"] = top_k
            del mutation_generator_parameters["maximum_number_of_mutations"]

        Mutation_Generator = mutation_generator_type(
            sequence=sequence_from_intervals, **mutation_generator_parameters
        )

        if full_sequence_embedding:
            scored_original_sequence: Dict[Any, Any] = {
                "score": float(self.score_sequence(self.sequence)),
                "sequence": self.sequence,
            }
        else:
            scored_original_sequence = {
                "score": float(
                    self.score_sequence(
                        self.sequence, intervals=intervals, fragment_embeddings=True
                    )
                ),
                "sequence": self.sequence,
            }

        original_sequence_score_ = scored_original_sequence["score"]

        logger.info(f"Original sequence score: {original_sequence_score_}")
        results: List[Dict[str, Any]] = [scored_original_sequence]

        visited_sequences: Set[str] = set()
        start_time = time.time()

        population: List[str] = []
        current_best_score = original_sequence_score_
        for step in range(number_of_steps):
            logger.info(f"Optimization step={step + 1}")

            (
                updated_visited_sequences,
                temporary_results,
                temporary_results_for_selection,
                new_current_best_score,
            ) = self.sequence_evaluation(
                original_sequence_score=original_sequence_score_,
                current_best_score=current_best_score,
                mutated_sequences_range=self.sequence_generation(
                    sequence_from_intervals=sequence_from_intervals,
                    mutation_object=Mutation_Generator,
                    initial_population=population,
                    number_of_samples_to_generate=population_per_iteration,
                ),
                visited_sequences=visited_sequences,
                intervals=intervals,
                batch_size=batch_size,
            )

            current_best_score = new_current_best_score
            visited_sequences = updated_visited_sequences
            results += temporary_results

            if with_genetic_algorithm:
                population = self.selection_crossover(
                    sequence_from_intervals,
                    temporary_results_for_selection,
                    intervals=intervals,
                    selection_method=selection_method,
                    top_k_selection=top_k_selection,
                    crossover_method=crossover_method,
                    crossover_probability=crossover_probability,
                )

            logger.info(
                f"Best score at step={step + 1}: {current_best_score}. Elapsed time: {int(time.time() - start_time)}s"
            )
            elapsed_time = int(time.time() - start_time)
            if time_budget is not None:
                if elapsed_time > time_budget:
                    logger.warning(
                        f"Used all the given time budget of {time_budget}s, exiting optimization loop"
                    )
                    break

        logger.info(
            f"Optimization completed, visiting {len(visited_sequences)} mutated sequences"
        )
        sorted_results = sorted(
            results, key=lambda result: result["score"], reverse=True
        )[:number_of_sequences]

        return sorted_results
