""" Enzyme Optimizer with Genetic Algorithm"""

import random
import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .core import IUPAC_MUTATION_MAPPING, AASequence, EnzymeOptimizer, Mutations
from .processing import (
    HuggingFaceTransformerEmbedding,
    StringEmbedding,
    TAPEEmbedding,
    crossover,
    reconstruct_sequence_with_mutation_range,
    sanitize_intervals,
    selection,
)


class EnzymeOptimizerGeneticAlgorithm(EnzymeOptimizer):
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

        EnzymeOptimizer.__init__(
            self,
            scorer_filepath,
            substrate,
            product,
            sequence,
            protein_embedding,
            molecule_embedding,
            ordering,
        )

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
        if self.protein_embedding == "ProtTransXL" in str(type(self.protein_embedding)):
            scored_original_sequence = {
                "score": self.protTrans_score_sequence(self.sequence),
                "sequence": self.sequence,
            }

        else:
            scored_original_sequence = {
                "score": self.score_sequence(self.sequence),
                "sequence": self.sequence,
            }

        # Keep track of best solution
        original_sequence_score = scored_original_sequence["score"]
        logger.info(f"original sequence score: {original_sequence_score}")
        results: List[Dict[str, Any]] = [scored_original_sequence]

        visited_sequences = set()
        start_time = time.time()
        for step in range(number_of_steps):
            logger.info(f"optimization step={step + 1}")
            population: List[str] = []

            # create population of mutated sequences from original sequence
            while len(population) < (number_of_samples_per_step * number_of_steps):  #
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
                    population.append(mutated_sequence)

            population_scores = []
            for i in range(
                0, len(population), number_of_samples_per_step
            ):  # score the population in chunks

                # evaluate all candidates in the population
                if "ProtTransXL" in str(type(self.protein_embedding)):
                    chunk_scores = self.protTrans_score_sequences(population)

                else:
                    chunk_scores = self.score_sequences(population)

            population_scores.extend(chunk_scores)

            # check for new best solution
            results += [
                scored_sequence
                for scored_sequence in population_scores
                if scored_sequence["score"] > original_sequence_score
            ]

            logger.info(
                f"best score at step={step + 1}: {max([scored_sequence['score'] for scored_sequence in results])}"
            )

            # select parents
            selected = selection(population_scores)
            # create the next generation
            children = list()
            for i in range(0, len(selected) - 1):
                p1, p2 = selected[i]["sequence"], selected[i + 1]["sequence"]

                # Extract active site given the sequence and the intervals
                p1_active_site = "".join([p1[start:end] for start, end in intervals])

                p2_active_site = "".join([p2[start:end] for start, end in intervals])
                # crossover over selected parents
                cross_over_1, cross_over_2 = crossover(p1_active_site, p2_active_site)

                children.append(cross_over_1)
                children.append(cross_over_2)

                new_p1 = reconstruct_sequence_with_mutation_range(
                    sequence=p1,
                    mutated_sequence_range=p1_active_site,
                    intervals=intervals,
                )
                new_p2 = reconstruct_sequence_with_mutation_range(
                    sequence=p2,
                    mutated_sequence_range=p2_active_site,
                    intervals=intervals,
                )
                children.append(new_p1)
                children.append(new_p2)

                children.append(p1)
                children.append(p1)

            # replace population, children are the new population for the next generation
            population = children

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
