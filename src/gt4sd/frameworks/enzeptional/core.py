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
"""Enzyme optimization."""

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
from transformers import T5EncoderModel, pipeline

from .processing import (
    CrossoverGenerator,
    AutoModelFromHFEmbedding,
    SelectionGenerator,
    StringEmbedding,
    TAPEEmbedding,
    reconstruct_sequence_with_mutation_range,
    sanitize_intervals,
    sanitize_intervals_with_padding,
)

#: transition matrix representation
TransitionMatrix = MutableMapping[str, MutableMapping[str, float]]
#: transition matrix configuration
TransitionConfiguration = MutableMapping[
    str, Union[MutableMapping[str, float], Sequence[str]]
]



#: supported features
SUPPORTED_FEATURE_SET = set(["substrate", "product", "sequence"])

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
IUPAC_CHARACTER_SET = set(IUPAC_CODES.values())
#: IUPAC uniform mutation mapping, we exclude 'X' from the mapping values because it denotes a generic AA
IUPAC_MUTATION_MAPPING: TransitionConfiguration = {
    iupac_character: sorted(list(IUPAC_CHARACTER_SET - {iupac_character, "X"}))
    for iupac_character in IUPAC_CHARACTER_SET
}


class Mutations:
    """Mutations definition class."""

    def __init__(self, transition_configuration: TransitionConfiguration) -> None:
        """Generate the mutation given the configuration for the transitions.
        Args:
            transition_configuration: transition configuration.
        """
        self.transition_matrix = Mutations.transition_configuration_to_matrix(
            transition_configuration
        )

    @staticmethod
    def transition_configuration_to_matrix(
        transition_configuration: TransitionConfiguration,
    ) -> TransitionMatrix:
        """Transform a configuration into a valid transition matrix.
        Args:
            transition_configuration: transition configuration.
        Returns:
            a transition matrix.
        """
        transition_matrix: TransitionMatrix = dict()
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
            the mutations object.
        """
        with open(filepath) as fp:
            return Mutations(json.load(fp))

    def mutate(self, source: str) -> str:
        """Mutate a source string.
        Args:
            source: source string.
        Returns:
            the mutated target.
        """
        targets, probabilities = zip(*self.transition_matrix[source].items())
        return np.random.choice(targets, size=1, p=probabilities).item()

class MutationGenerator:
    def __init__(self, sequence: str) -> None:
        self.sequence = sequence

    def get_mutations(self, number_of_mutated_sequences: int = 1) -> List[str]:
        raise NotImplementedError("Implement the method in a sub-class")
     
## Select Model

MODEL_EMBEDDING_MAPPING: Dict[str, Type[StringEmbedding]] = {
    "transformers": AutoModelFromHFEmbedding,
    "tape": TAPEEmbedding
}
    
class MutationLanguageModel:
    def __init__(
        self,
        mutation_model_type: str,
        mutation_model_parameters: Dict[str, str]
    ) -> None:
        """Load language model for mutation suggestion

        Args:
            mutation_model_type (str): Type of model to use. Defaults to "transformers".
            mutation_model_parameters: Example: { "mutation_model_type" : "facebook/esm2_t33_650M_UR50D", "mutation_model_path" : "facebook/esm2_t33_650M_UR50D" }.
        """
        
        if mutation_model_type == "transformers":
            self.load_mutation_model = MODEL_EMBEDDING_MAPPING.get(mutation_model_type)(
                mutation_model_parameters.get("mutation_model_path"), 
                mutation_model_parameters.get("mutation_tokenizer_filepath"))
        elif mutation_model_type == "tape":
            self.load_mutation_model = MODEL_EMBEDDING_MAPPING.get(mutation_model_type)(
                mutation_model_parameters.get("model_type"), 
                mutation_model_parameters.get("model_dir"),
                mutation_model_parameters.get("aa_vocabulary"))
        else:
            raise ValueError(
                f"Check files/directories for muatation model!"
            )
        
        # self.load_mutation_model = AutoModelFromHFEmbedding(
        #     mutation_model_path, mutation_tokenizer_filepath
        # )


class MutationGeneratorLanguageModeling(MutationGenerator):
    def __init__(
        self,
        sequence: str,
        mutation_object: MutationLanguageModel,
        top_k: int = 2,
        maximum_number_of_mutations: int = 4,
    ) -> None:
        """Langauge model mutations generator

        Args:
            sequence (str): an amino acid sequence
            mutation_object (MutationLanguageModel): a mutation object
            top_k (int): Number of alternatives to consider when selecting the replacement for each amino acid. Defaults to 2.
            maximum_number_of_mutations (int): maximum number of mutations. Defaults to 4.
        """
        super().__init__(sequence)
        self.sequence_length = len(sequence)
        self.mutation_object = mutation_object
        self.top_k = top_k
        self.maximum_number_of_mutations = maximum_number_of_mutations
        self.tokenized_sequence = list(self.sequence)
        self.pipeline = pipeline(
            "fill-mask",
            model=self.mutation_object.load_mutation_model.model,
            tokenizer=self.mutation_object.load_mutation_model.tokenizer,
            top_k=self.top_k,
        )

    def get_mutations(self, number_of_mutated_sequences: int = 1) -> List[str]:
        """Get mutations

        Args:
            number_of_mutated_sequences (int): Number of mutated sequences to return. Defaults to 1.

        Returns:
            List[str]: return sequence/s
        """
        output: List[str] = []
        while len(output) <= number_of_mutated_sequences:

            tmp_sequence = self.tokenized_sequence

            number_of_mutations = random.randint(1, self.maximum_number_of_mutations)
            positions = sorted(
                random.sample(range(self.sequence_length), number_of_mutations)
            )

            for pos in positions:
                if (
                    self.mutation_object.load_mutation_model.model.config_class.model_type
                    == "esm"
                ):
                    tmp_sequence[pos] = "<mask>"
                else:
                    tmp_sequence[pos] = "[MASK]"

            tmp_masked_sequence = " ".join(tmp_sequence)

            replacement = []
            for out in self.pipeline(tmp_masked_sequence):
                tmp_replacement = []
                if number_of_mutations > 1:
                    for internal_out in out:
                        tmp_replacement.append(internal_out["token_str"])
                    replacement.append(tmp_replacement)
                else:
                    replacement.append(out["token_str"])

            replacement = np.array(replacement).T
            for i in range(len(replacement)):
                tmp_tokenized_sequence = self.tokenized_sequence
                for j in range(number_of_mutations):
                    tmp_tokenized_sequence[positions[j]] = replacement[i][j]
                output.append("".join(tmp_tokenized_sequence))

        return output


class MutationGeneratorTransitionMatrix(MutationGenerator):
    def __init__(
        self,
        sequence: str,
        mutation_object: Mutations = Mutations(IUPAC_MUTATION_MAPPING),
        maximum_number_of_mutations: int = 4,
    ) -> None:
        """Transition matrix mutations generator

        Args:
            sequence (str): an amino acid sequence
            mutation_object (MutationLanguageModel): a mutation object
            maximum_number_of_mutations (int): maximum number of mutations. Defaults to 4.
        """
        super().__init__(sequence)
        self.sequence_length = len(sequence)
        self.mutation_object = mutation_object
        self.maximum_number_of_mutations = maximum_number_of_mutations

    def get_single_sequence_with_mutations(self) -> str:
        """mutate the sequence from init

        Returns:
            str: mutated sequence
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
        """Generate # number of mutations

        Args:
            number_of_mutated_sequences (int): number of sequences to return. Defaults to 1.

        Returns:
            List[str]: sequence/s
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
                f"mutations with type: {type(mutation_object)} not supported!"
            )
        return mutation_generator


MUTATION_GENERATORS: Dict[str, Type[MutationGenerator]] = {
    "transition-matrix": MutationGeneratorTransitionMatrix,
    "language-modeling": MutationGeneratorLanguageModeling,
}

CROSSOVER_GENERATOR: Dict[
    str,
    Callable[[str, str, float], Tuple[str, str]]
] = {
    "single_point": lambda a_sequence, another_sequence, _: CrossoverGenerator().single_point_crossover(
        a_sequence, another_sequence
    ),
    "uniform": lambda a_sequence, another_sequence, probability: CrossoverGenerator(probability).uniform_crossover(
        a_sequence, another_sequence
    ),
}

SELECTION_GENERATOR: Dict[str, Callable[[Any, Any], List[Any]]] = {
    "generic": lambda scores, k: SelectionGenerator().selection(scores, k)
}


class EnzymeOptimizer:
    """Optimize an enzyme to catalyze a reaction from substrate to product."""

    def __init__(
        self,
        scorer_filepath: str,
        substrate: str,
        product: str,
        sequence: str,
        protein_embedding: StringEmbedding = TAPEEmbedding(),
        molecule_embedding: StringEmbedding = AutoModelFromHFEmbedding(
            "seyonec/ChemBERTa-zinc-base-v1", "seyonec/ChemBERTa-zinc-base-v1"
        ),
        ordering: List[str] = ["substrate", "product", "sequence"],
    ) -> None:
        """Initialize the enzyme designer.
        Args:
            scorer_filepath: pickled scorer filepath.
            substrate: substrate SMILES.
            product: product SMILES.
            sequence: AA sequence representing the enzyme to optimize.
            protein_embedding: protein embedding class. Defaults to TAPE bert-base.
            molecule_embedding: molecule embedding class. Defaults to ChemBERTa version 1.
            ordering: ordering of the features for the scorer. Defaults to ["substrate", "product", "sequence"].
        Raises:
            ValueError: ordering provided is not feasible.
        """
        if len(set(ordering).intersection(SUPPORTED_FEATURE_SET)) < 3:
            raise ValueError(
                f"ordering={ordering} should contain only the three admissible values: {sorted(list(SUPPORTED_FEATURE_SET))}"
            )
        else:
            self._ordering = ordering
        self.scorer_filepath = scorer_filepath
        self.scorer = load(scorer_filepath)
        self.substrate = substrate
        self.product = product

        self.protein_embedding = protein_embedding
        self.molecule_embedding = molecule_embedding
        self.embedded_vectors = {
            "substrate": self.molecule_embedding.embed_one(self.substrate),
            "product": self.molecule_embedding.embed_one(self.product),
        }
        self.sequence = sequence
        self.sequence_length = len(sequence)

    def extract_fragment_embedding(
        self, sequence: str, intervals: List[Tuple[int, int]]
    ):
        """extrcat the embeddings for each fragment in a sequence.
        Args:
            sequence: a sequence from which extract the fragments.
            intervals: list of ranges in the sequence, zero-based. The same interval is applied to all sequences
        Returns:
            The mean embedding of the input sequence based on the intervals.
        """
        fragments: List[str] = []
        for start, end in intervals:
            size_fragment = end - start
            fragments.append("".join(sequence[:size_fragment]))
            sequence = sequence[size_fragment:]
        sequence_embedding = np.array(
            [self.protein_embedding.embed_one(fragment) for fragment in fragments]
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
            sequence: a sequence to score.
            intervals: list of ranges in the sequence, zero-based
            fragment_embeddings (bool). In case of fragment embeddings, set fragment_embeddings = True. Defaults is False.
        Returns:
            Input sequence and score.
        """
        if type(self.protein_embedding.model) == T5EncoderModel:
            sequence = " ".join(list(sequence))
        if fragment_embeddings and intervals is not None:
            embedded_vectors = {
                "sequence": self.extract_fragment_embedding(sequence, intervals)
            }
        else:
            embedded_vectors = {"sequence": self.protein_embedding.embed_one(sequence)}
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
            sequences: a list of sequencea to score.
            intervals: list of ranges in the sequence, zero-based. The same interval is applied to all sequences
            fragment_embeddings (bool). In case of fragment embeddings, set fragment_embeddings = True. Defaults is False.
        Returns:
            Input sequences and scores.
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
        if type(self.protein_embedding.model) == T5EncoderModel:
            sequences = [" ".join(list(sequence)) for sequence in sequences]

        if fragment_embeddings and intervals is not None:
            embeddings = []
            for sequence in sequences:
                embeddings.append(self.extract_fragment_embedding(sequence, intervals))
            embedded_matrices["sequence"] = np.array(embeddings)

        else:
            embedded_matrices["sequence"] = self.protein_embedding.embed_multiple(
                sequences
            )
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
        initial_population: List[str] = None,
        number_of_samples_to_generate: int = 1,
    ) -> List[str]:
        """generate sequences.
        Args:
            sequence_from_intervals: orignial sequence extrcated from interval.
            mutation_object: mutation object
            initial_population: list of samples,
            number_of_samples_to_generate: int. number of samples to generate
        Returns:
            a list of sequences.
        """
        lst_mutated_sequences: List[str] = []
        if initial_population:
            for indx in range(len(initial_population)):
                initial_sequence = initial_population[indx]
                mutation_object.sequence = initial_sequence
                lst_mutated_sequences += mutation_object.get_mutations()
            if number_of_samples_to_generate < len(lst_mutated_sequences):
                mutation_object.sequence = sequence_from_intervals
                lst_mutated_sequences += mutation_object.get_mutations(
                    number_of_mutated_sequences= number_of_samples_to_generate - len(lst_mutated_sequences)
                )
        else:
            lst_mutated_sequences += mutation_object.get_mutations(
                number_of_mutated_sequences=number_of_samples_to_generate
            )

        return list(set(lst_mutated_sequences))

    def sequence_evaluation(
        self,
        original_sequence_score: float,
        mutated_sequences_range: List[str],
        visited_sequences: set,
        intervals: List[Tuple[int, int]],
        batch_size: Optional[int] = None,
    ) -> Tuple[Set[Any], List[Dict[str, Any]]]:
        """evaluate sequences.
        Args:
            original_sequence_score: float. Score of the original sequence
            mutated_sequences_range: list of mutated sequences (just concatenated fragents of sequences)
            visited_sequences: set of sequences already evaluated in the past optimization steps
            intervals: list of ranges in the sequence, zero-based.
            batch_size: number of sequences to evaluated in one round
        Returns:
            a list of sequences.
        """
        temporary_results: List[Dict[str, Any]] = []

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
            temporary_results += [
                scored_sequence
                for scored_sequence in self.score_sequences(
                    lst_mutated_sequences[i : i + batch_size]
                )
                if scored_sequence["score"] > original_sequence_score
            ]

        return visited_sequences, temporary_results

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
        """Perform Selection and Crossover

        Args:
            sequence_from_intervals (str): orignial sequence extrcated from interval.
            tmp_results (List[Dict[str, Any]]): the temporary results
            intervals (List[Tuple[int, int]]): list of ranges in the sequence, zero-based. The same interval is applied to all sequences
            selection_method (str): methodology used to selection
            crossover_method (str): methodology used for crossover
            crossover_probability (float, optional): Crossover probability. Used in case Uniform crossover is selected. Defaults to 0.5.
            top_k_selection (Optional[int], optional): Number of samples to select. Defaults to -1.

        Returns:
            List[str]: New samples for the next round of optimization
        """

        crossover: Callable[[str, str, float], Tuple[str, str]] = CROSSOVER_GENERATOR.get(crossover_method)
        selection: Callable[[Any, Any], List[Any]] = SELECTION_GENERATOR.get(selection_method)

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
                    selected_child_mutated_fragments, sequence_from_intervals, 1)

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
        mutation_generator_type: Union[
            str, Type[MutationGenerator]
        ] = MutationGeneratorTransitionMatrix,
        mutation_generator_parameters: Dict[str, Any] = {
            "mutation_object": Mutations(IUPAC_MUTATION_MAPPING),
            "maximum_number_of_mutations": 4,
        },
        top_k: Optional[int] = 2,
        pad_intervals: Optional[bool] = False,
        population_per_itaration: Optional[int] = None,
        with_genetic_algorithm: Optional[bool] = False,
        selection_method: str = "generic",
        top_k_selection: Optional[int] = None,
        crossover_method: str = "single_point",
        crossover_probability: float = 0.5,
    ) -> List[Dict[str, Any]]:

        """Optimize the enzyme given a number of mutations and a range.
        If the range limits are not provided the full sequence is optimized, this might be inefficient.
        The sampling is performing by exploring mutations with a slightly smart random sampling.
        Args:
            number_of_mutations: number of allowed mutations.
            intervals: list of ranges in the sequence, zero-based. Defaults to None, a.k.a. use optimize the full sequence.
            number_of_steps: number of optimization steps. Defaults to 10.
            batch_size: number of sequences to embedded together. Defaults to 8
            full_sequence_embedding: perform embeddings respect to the full sequence. Defaults to True. False = respect to intervals fragments
            number_of_sequences: number of optimal seuqence returned. Defaults to None, a.k.a, returns all.
            seed: seed for random number generation. Defaults to 42.
            minimum_interval_length: minim length per interval in case the full_sequence_embedding=True.
            time_budget: maximum allowed runtime in seconds. Defaults to None, a.k.a., no time limit, running for number_of_steps steps.
            mutation_generator_type: mutations generation type. 
            mutation_generator_parameters: mutation generation parameters. Defaults to uniform sampling of IUPAC AAs.
            top_k: How many suggested AA to accept. Defaults to top 2.
            pad_intervals: if True, in case a fragment of sequence has length < 8: it's going to be padded to a length = at least 8
            population_per_itaration: number of samples sequences per optimization step
            with_genetic_algorithm: Optimize using a genetic algorith
            selection_method: in case of genetic algorithm optimization. selection method with number of samples to select for crossover Defaults=5
            top_k_selection: number of suggested mutants per amino acid to consider.
            crossover_method: crossover method selection. Options are single point (single_point) or uniform.
            crossover_probability: crossover probabilty in case of crossover method.
        Raises:
            ValueError: in case an invalid range is provided.
        Returns:
            a list of dictionaries containing a candidate optimal sequence and the related score. Sorted from best to worst.
            Note that, when no limit on the returned number of sequences is set, the worst sequence is the original unmutated sequence.
            If the optimization fails, only the original sequence is returned.
        """

        random.seed(seed)
        # check if interval is None. In case it is, take as interval the whole sequence
        if intervals is None:
            intervals = [(0, self.sequence_length)]
        else:
            intervals = sanitize_intervals(
                intervals
            )  # here we merged and sorted the intervals

        # pad the sequences, to a minimal length of 8
        if pad_intervals:
            intervals = sanitize_intervals_with_padding(
                intervals=intervals,
                sequence_length=self.sequence_length,
                minimum_interval_length=minimum_interval_length,
            )

        # check that the intervals are in the range of the sequence length
        if intervals[-1][1] > self.sequence_length:
            raise ValueError(
                "check provided intervals, at least an interval is larger than the sequence length"
            )

        # mutate the sequence from intervals
        self.maximum_number_of_mutations = number_of_mutations

        if self.maximum_number_of_mutations > self.sequence_length:
            logger.warning(
                f"resetting maximum number of mutations ({self.maximum_number_of_mutations}), since it is higher than sequence length: {self.sequence_length}"
            )
            self.maximum_number_of_mutations = self.sequence_length
        if self.maximum_number_of_mutations < 1:
            logger.warning(
                f"maximum number of mutations can't be lower than 1 ({self.maximum_number_of_mutations}), resetting to 1"
            )
            self.maximum_number_of_mutations = 1

        logger.info(
            f"maximum number of mutations for the intervals: {self.maximum_number_of_mutations}"
        )

        # Check if population size is set
        if not population_per_itaration:
            population_per_itaration = batch_size

        # create a sequence from based on the intervals
        sequence_from_intervals = "".join(
            [self.sequence[start:end] for start, end in intervals]
        )

        # mutation_object = mutation_generator_parameters.pop(
        #     "mutation_object", Mutations(IUPAC_MUTATION_MAPPING)
        # )
        if isinstance(mutation_generator_type, str):
            mutation_generator_type = MUTATION_GENERATORS[mutation_generator_type]
        elif Type[MutationGenerator] == Type[mutation_generator_type]:
            mutation_generator_type = mutation_generator_type
        else:
            raise ValueError(
                f"mutation generator with type: {type(mutation_generator_type)} not supported!"
            )

        if Type[mutation_generator_type] == Type[MutationGeneratorLanguageModeling]:
            mutation_generator_parameters["top_k"] = top_k
            del mutation_generator_parameters["maximum_number_of_mutations"]


        
        mutation_generator = mutation_generator_type(
            # mutation_object=mutation_object,
            sequence=sequence_from_intervals,
            **mutation_generator_parameters,
        )

        if full_sequence_embedding:
            scored_original_sequence : Dict[str, Any] = {
                "score": self.score_sequence(self.sequence),
                "sequence": self.sequence,
            }
        else:
            scored_original_sequence = {
                "score": self.score_sequence(
                    self.sequence, intervals=intervals, fragment_embeddings=True
                ),
                "sequence": self.sequence,
            }

        original_sequence_score_  = scored_original_sequence["score"]
        
        logger.info(f"original sequence score: {original_sequence_score_}")
        results: List[Dict[str, Any]] = [scored_original_sequence]

        visited_sequences: Set[str] = set()
        start_time = time.time()

        population: List[str] = []
        for step in range(number_of_steps):
            logger.info(f"optimization step={step + 1}")

            updated_visited_sequences, temporary_results = self.sequence_evaluation(
                original_sequence_score=original_sequence_score_,
                mutated_sequences_range=self.sequence_generation(
                    sequence_from_intervals=sequence_from_intervals,
                    mutation_object=mutation_generator,
                    initial_population=population,
                    number_of_samples_to_generate=population_per_itaration,
                ),
                visited_sequences=visited_sequences,
                intervals=intervals,
                batch_size=batch_size,
            )

            visited_sequences = updated_visited_sequences
            results += temporary_results

            if with_genetic_algorithm:
                population = self.selection_crossover(
                    sequence_from_intervals,
                    temporary_results,
                    intervals=intervals,
                    selection_method=selection_method,
                    top_k_selection=top_k_selection,
                    crossover_method=crossover_method,
                    crossover_probability=crossover_probability,
                )

            logger.info(
                f"best score at step={step + 1}: {max([scored_sequence['score'] for scored_sequence in results])}"
            )
            elapsed_time = int(time.time() - start_time)
            if time_budget is not None:
                if elapsed_time > time_budget:
                    logger.warning(
                        f"used all the given time budget of {time_budget}s, exting optimization loop"
                    )
                    break

        logger.info(
            f"optimization completed visiting {len(visited_sequences)} mutated sequences"
        )
        sorted_results = sorted(
            results, key=lambda result: result["score"], reverse=True
        )[:number_of_sequences]
        if len(sorted_results) < 2:
            logger.error(
                "optimization failed, could not find a mutated sequence more optimal than the original"
            )
        else:
            logger.info(
                f"found {len(sorted_results) -  1} optimal mutated sequences, best score: {sorted_results[0]['score']}"
            )
        return sorted_results

