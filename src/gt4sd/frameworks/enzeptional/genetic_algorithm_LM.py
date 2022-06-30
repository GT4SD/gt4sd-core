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
""" Enzyme Optimizer with Genetic Algorithm"""
import itertools
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from loguru import logger
from transformers import PreTrainedTokenizer, pipeline

from .core_LM import EnzymeOptimizer, sequence_masking
from .processing import (
    HuggingFaceTransformerEmbedding,
    StringEmbedding,
    reconstruct_sequence_with_mutation_range,
    sanitize_intervals,
    sanitize_intervals_with_padding,
    selection,
    uniform_crossover,
)


class EnzymeOptimizerGeneticAlgorithm(EnzymeOptimizer):
    """Optimize an enzyme to catalyze a reaction from substrate to product."""

    def __init__(
        self,
        scorer_filepath: str,
        substrate: str,
        product: str,
        sequence: str,
        protein_embedding_type: str,
        protein_embedding_path: Optional[str],
        mutation_model_type: str,
        mutation_model_path: Optional[str],
        molecule_embedding: StringEmbedding = HuggingFaceTransformerEmbedding(),
        ordering: List[str] = ["substrate", "product", "sequence"],
    ) -> None:

        super().__init__(
            scorer_filepath,
            substrate,
            product,
            sequence,
            protein_embedding_type,
            protein_embedding_path,
            mutation_model_type,
            mutation_model_path,
            molecule_embedding,
            ordering,
        )

    def optimize(
        self,
        number_of_mutations: int,
        intervals: Optional[List[Tuple[int, int]]] = None,
        number_of_steps: int = 10,
        number_of_sequences: Optional[int] = None,
        seed: int = 42,
        time_budget: Optional[int] = None,
        pad_intervals: Optional[bool] = False,
        number_of_best_predictions=2,
        population_per_itaration=100,
        number_selection_per_iteration=20,
    ) -> List[Dict[str, Any]]:
        """Optimize the enzyme given a number of mutations and a range.
        If the range limits are not provided the full sequence is optimized, this might be inefficient.
        The sampling is performing by exploring mutations with a slightly smart random sampling.
        Args:
            number_of_mutations: number of allowed mutations.
            intervals: list of ranges in the sequence, zero-based. Defaults to None, a.k.a. use optimize the full sequence.
            number_of_steps: number of optimization steps. Defaults to 100.
            number_of_iterations_per_step: number of samples sequences per optimization step. Defaults to 32.
            number_of_sequences: number of optimal seuqence returned. Defaults to None, a.k.a, returns all.
            seed: seed for random number generation. Defaults to 42.
            time_budget: maximum allowed runtime in seconds. Defaults to None, a.k.a., no time limit, running for number_of_steps steps.
            pad_intervals: if True, in case a fragment of sequence has length < 8: it's going to be padded to a length = at least 8
            number_of_best_predictions: top_k for mask prediciont of masked sequences.
            population_per_itaration: number of samples in the population at each iteration.
            number_selection_per_iteration: number of top performing sequences to keep at each iteration.

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

        if pad_intervals:
            intervals = sanitize_intervals_with_padding(intervals, self.sequence_length)

        # check that the intervals are in the range of the sequence length
        if intervals[-1][1] > self.sequence_length:
            raise ValueError(
                "check provided intervals, at least an interval is larger than the sequence length"
            )

        # mutate the sequence from intervals
        maximum_number_of_mutations = number_of_mutations

        logger.info(
            f"maximum number of mutations for the intervals: {maximum_number_of_mutations}"
        )

        extracted_interval = [
            self.sequence[start - 1 : end - 1] for start, end in intervals
        ]

        logger.info(f"{extracted_interval}")

        # Iterations
        scored_original_active_site_interval = self.score_sequences(extracted_interval)
        scored_original_active_site_interval["intervals"] = ""

        original_sequence_score = scored_original_active_site_interval["score"]

        logger.info(f"original sequence score: {original_sequence_score}")
        # results: List[Dict[str, Any]] = [scored_original_active_site_interval]
        results: List[
            Dict[str, Any]
        ] = []  # without including the original sequence in the result file
        # slightly smart random sampling
        visited_sequences = []
        start_time = time.time()

        # pipeline for mutations suggestion
        fill_mask = pipeline(
            "fill-mask",
            model=self.mutation_model,
            tokenizer=cast(
                Union[str, PreTrainedTokenizer, None], self.mutation_model_tokenizer
            ),
            top_k=number_of_best_predictions,
        )

        tokenized_seq = " ".join(self.mutation_model_tokenizer.tokenize(self.sequence))

        population: List[Dict[str, Any]] = []

        for step in range(number_of_steps):
            logger.info(f"optimization step={step + 1}")

            tmp_population = []
            # create population of mutated sequences from original sequence
            while len(population) < population_per_itaration:  #

                masked_sequence_intervals = sequence_masking(
                    tokenized_seq, intervals, number_of_mutations
                )
                population.append(
                    {"masked_sequence_intervals": masked_sequence_intervals}
                )

            # score population
            logger.info(f"scoring mutated sequences of step {step+1}")
            for tmp_intervals in population:
                masked_sequence_intervals = tmp_intervals["masked_sequence_intervals"]
                # combine the MASK predictions
                # extract the top_k prediction for each mask
                best_fits = []
                for element in masked_sequence_intervals:
                    if element.count("[MASK]") < 1:
                        best_fits.append(
                            ["".join(element)] * number_of_best_predictions
                        )

                    elif element.count("[MASK]") == 1:
                        tmp_best = []
                        for out in fill_mask(" ".join(element)):
                            tmp_best.append("".join(out["sequence"].split()))
                        best_fits.append(tmp_best)
                    else:
                        tmp_best = []
                        tmp_token_str = []
                        for output in fill_mask(" ".join(element)):
                            tmp_str = []
                            for i in output:
                                tmp_str.append(i["token_str"])
                            tmp_token_str.append(tmp_str)
                        tmp_token_str = np.array(tmp_token_str).T.tolist()

                        for k in range(len(tmp_token_str)):
                            tmp_seq = element
                            token_str_to_consider = tmp_token_str[k]
                            for val in range(len(element)):
                                if element[val] == "[MASK]":
                                    tmp_seq[val] = token_str_to_consider[0]
                                    token_str_to_consider = token_str_to_consider[1:]

                            tmp_best.append("".join(tmp_seq))
                        best_fits.append(tmp_best)

                combination_lst: List[Any] = list(set(itertools.product(*best_fits)))

                for mutated_active_site in combination_lst:
                    mutated_active_site = list(mutated_active_site)
                    if mutated_active_site not in visited_sequences:
                        visited_sequences.append(mutated_active_site)

                        scored_sequence = self.score_sequences(mutated_active_site)
                        scored_sequence["intervals"] = intervals

                        if scored_sequence["score"] > original_sequence_score:
                            results += [scored_sequence]
                            tmp_population += [scored_sequence]

            logger.info(
                f"best score at step={step + 1}: {max([scored_sequence['score'] for scored_sequence in results])}"
            )

            if tmp_population:
                # select best filt, do crossover and save for next generation
                selected_children = selection(
                    tmp_population, k=number_selection_per_iteration
                )

                # create the next generation
                children: List[Dict[str, Any]] = []
                for pos in range(0, len(selected_children) - 1):
                    selected_child = selected_children[pos]
                    # select fragment where to perform crossover
                    fragment_pos = random.randrange(len(selected_child["sequence"]))

                    child_fragment_to_cross = selected_child["sequence"][fragment_pos]
                    parent_fragment_to_cross = extracted_interval[fragment_pos]

                    for crossover_child in uniform_crossover(
                        parent_fragment_to_cross.split(),
                        child_fragment_to_cross.split(),
                        0.3,
                    ):

                        # add crossover to next generation
                        tmp_child = selected_child
                        tmp_child["sequence"][fragment_pos] = crossover_child

                        tmp_child = reconstruct_sequence_with_mutation_range(
                            self.sequence, "".join(tmp_child["sequence"]), intervals
                        )

                        (masked_sequence_intervals_child) = sequence_masking(
                            " ".join(self.mutation_model_tokenizer.tokenize(tmp_child)),
                            intervals,
                            number_of_mutations,
                        )
                        children.append(
                            {
                                "masked_sequence_intervals": masked_sequence_intervals_child,
                            }
                        )

                # replace population, children are the new population for the next generation
                population = children
            else:
                population = []

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
