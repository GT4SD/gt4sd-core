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
"""Enzyme optimization."""

import json
import logging
import random
import time
from collections import OrderedDict
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
from joblib import load

from .processing import (
    HuggingFaceTransformerEmbedding,
    StringEmbedding,
    TAPEEmbedding,
    reconstruct_sequence_with_mutation_range,
    sanitize_intervals,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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


class AASequence:
    def __init__(
        self, sequence: str, mutations: Mutations = Mutations(IUPAC_MUTATION_MAPPING)
    ) -> None:
        """Initialize an AA sequence representation.

        Args:
            sequence: AA sequence.
            mutations: mutations definition. Defaults to uniform sampling of IUPAC AAs.
        """
        self.sequence = sequence
        self.sequence_length = len(sequence)
        self.mutations = mutations

    def mutate(self, maximum_number_of_mutations: int) -> str:
        """Mutate the sequence in multiple positions.

        Args:
            maximum_number_of_mutations: maximum number of mutations.

        Returns:
            the mutated sequence.
        """
        if maximum_number_of_mutations > self.sequence_length:
            logger.warning(
                f"resetting maximum number of mutations ({maximum_number_of_mutations}), since it is higher than sequence length: {self.sequence_length}"
            )
            maximum_number_of_mutations = self.sequence_length
        if maximum_number_of_mutations < 1:
            logger.warning(
                f"maximum number of mutations can't be lower than 1 ({maximum_number_of_mutations}), resetting to 1"
            )
            maximum_number_of_mutations = 1
        number_of_mutations = random.randint(1, maximum_number_of_mutations)
        positions = sorted(
            random.sample(range(self.sequence_length), number_of_mutations)
        )
        mutated_sequence = ""
        start_position = -1
        for position in positions:
            mutated_sequence += self.sequence[(start_position + 1) : position]
            mutated_sequence += self.mutations.mutate(self.sequence[position])
            start_position = position
        mutated_sequence += self.sequence[(start_position + 1) :]
        return mutated_sequence


class EnzymeOptimizer:
    """Optimize an enzyme to catalyze a reaction from substrate to product."""

    def __init__(
        self,
        scorer_filepath: str,
        substrate: str,
        product: str,
        sequence: str,
        protein_embedding: StringEmbedding = TAPEEmbedding(),
        molecule_embedding: StringEmbedding = HuggingFaceTransformerEmbedding(),
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

        Example:
            An example optimizing a specific reaction::

                filepath = f"/path/to/model/scoring_model.pkl"
                substrate = "NC1=CC=C(N)C=C1"
                product = "CNC1=CC=C(NC(=O)C2=CC=C(C=C2)C(C)=O)C=C1"
                sequence = (
                    "MSIQIKQSTMVRPAEETPNKSLWLSNIDMILRTPYSHTGAVLIYKQPDNNEDNIHPSSSMYFDANILIEALSKA"
                    "LVPFYPMAGRLKINGDRYEIDCNAEGALFVEAESSHVLEDFGDFRPNDELHRVMVPTCDYSKGISSFPLLMVQLT"
                    "RFRCGGVSIGFAQHHHVCDGMAHFEFNNSWARIAKGLLPALEPVHDRYLHLRPRNPPQIKYSHSQFEPFVPSLPN"
                    "ELLDGKTNKSQTLFILSREQINTLKQKLDLSNNTTRLSTYEVVAAHVWRSVSKARGLSDHEEIKLIMPVDGRSRIN"
                    "NPSLPKGYCGNVVFLAVCTATVGDLSCNPLTDTAGKVQEALKGLDDDYLRSAIDHTESKPGLPVPYMGSPEKTLYPN"
                    "VLVNSWGRIPYQAMDFGWGSPTFFGISNIFYDGQCFLIPSRDGDGSMTLAINLFSSHLSRFKKYFYDF"
                )
                # instantiate the designer
                designer = EnzymeOptimizer(
                    scorer_filepath=filepath, substrate=substrate, product=product, sequence=sequence
                )


                # with this sequence length every steps takes ~5s
                # optimize between positions 150 and 405 allowing for a maximum of 5 mutations.
                results = designer.optimize(
                    number_of_mutations=5, number_of_steps=10, number_of_samples_per_step=8,
                    intervals=[(150, 405)]
                )
                best_score = results[0]["score"]
                best_sequence = results[0]["sequence"]
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

    def score_sequence(self, sequence: str) -> float:
        """Score a given sequence.

        Args:
            sequence: a sequence to score.

        Returns:
            score for the sequence.
        """
        embedded_vectors = {"sequence": self.protein_embedding.embed_one(sequence)}
        embedded_vectors.update(self.embedded_vectors)
        feature_vector = np.concatenate(
            [embedded_vectors[feature] for feature in self._ordering], axis=1
        )
        return self.scorer.predict_proba(feature_vector)[0][1]

    def score_sequences(self, sequences: List[str]) -> List[Dict[str, Any]]:
        """Score a given sequence list.

        Args:
            sequences: a list of sequences to score.

        Returns:
            a list of dictionaries of sequences and related scores.
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
        embedded_matrices["sequence"] = self.protein_embedding(sequences)
        feature_vector = np.concatenate(
            [embedded_matrices[feature] for feature in self._ordering], axis=1
        )
        return [
            {"sequence": sequence, "score": score}
            for sequence, score in zip(
                sequences, self.scorer.predict_proba(feature_vector)[:, 1]
            )
        ]

    def optimize(
        self,
        number_of_mutations: int,
        intervals: Optional[List[Tuple[int, int]]] = None,
        number_of_steps: int = 10,
        number_of_samples_per_step: int = 32,
        number_of_sequences: Optional[int] = None,
        seed: int = 42,
        time_budget: Optional[int] = None,
        mutations: Mutations = Mutations(IUPAC_MUTATION_MAPPING),
    ) -> List[Dict[str, Any]]:
        """Optimize the enzyme given a number of mutations and a range.

        If the range limits are not provided the full sequence is optimized, this might be inefficient.
        The sampling is performing by exploring mutations with a slightly smart random sampling.

        Args:
            number_of_mutations: number of allowed mutations.
            intervals: list of ranges in the sequence, zero-based. Defaults to None, a.k.a. use optimize the full sequence.
            number_of_steps: number of optimization steps. Defaults to 100.
            number_of_samples_per_step: number of samples sequences per optimization step. Defaults to 32.
            number_of_sequences: number of optimal seuqence returned. Defaults to None, a.k.a, returns all.
            seed: seed for random number generation. Defaults to 42.
            time_budget: maximum allowed runtime in seconds. Defaults to None, a.k.a., no time limit, running for number_of_steps steps.
            mutations: mutations definition. Defaults to uniform sampling of IUPAC AAs.

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

        # check that the intervals are in the range of the sequence length
        if intervals[-1][1] > self.sequence_length:
            raise ValueError(
                "check provided intervals, at least an interval is larger than the sequence length"
            )

        # create a sequence from based on the intervals
        sequence_from_intervals = "".join(
            [self.sequence[start:end] for start, end in intervals]
        )

        # mutate the sequence from intervals
        aa_sequence_range = AASequence(sequence_from_intervals, mutations=mutations)
        maximum_number_of_mutations = number_of_mutations

        logger.info(
            f"maximum number of mutations for the intervals: {maximum_number_of_mutations}"
        )
        scored_original_sequence = {
            "score": self.score_sequence(self.sequence),
            "sequence": self.sequence,
        }
        original_sequence_score = scored_original_sequence["score"]
        logger.info(f"original sequence score: {original_sequence_score}")
        results: List[Dict[str, Any]] = [scored_original_sequence]
        # slightly smart random sampling
        visited_sequences = set()
        start_time = time.time()
        for step in range(number_of_steps):
            logger.info(f"optimization step={step + 1}")
            mutated_sequences = []

            for _ in range(number_of_samples_per_step):
                mutated_sequence_range = aa_sequence_range.mutate(
                    maximum_number_of_mutations=maximum_number_of_mutations
                )

                mutated_sequence = reconstruct_sequence_with_mutation_range(
                    sequence=self.sequence,
                    mutated_sequence_range=mutated_sequence_range,
                    intervals=intervals,
                )

                # make sure we do not revisit
                if mutated_sequence not in visited_sequences:
                    visited_sequences.add(mutated_sequence)
                    mutated_sequences.append(mutated_sequence)

            # add only mutated sequences that are more optimal than the original
            results += [
                scored_sequence
                for scored_sequence in self.score_sequences(mutated_sequences)
                if scored_sequence["score"] > original_sequence_score
            ]
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
