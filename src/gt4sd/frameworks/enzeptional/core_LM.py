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

import abc
import itertools
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from joblib import load
from loguru import logger
from transformers import (
    AlbertForMaskedLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    XLNetLMHeadModel,
    pipeline,
)

from .processing import (
    HuggingFaceTransformerEmbedding,
    LMEmbedding,
    ProtAlbertBert,
    ProtTransXL,
    StringEmbedding,
    TAPEEmbedding,
    sanitize_intervals,
    sanitize_intervals_with_padding,
)

#: supported features
SUPPORTED_FEATURE_SET = set(["substrate", "product", "sequence"])

PROTEIN_EMBEDDING_CLASS = {
    "prottrans": ProtTransXL,
    "tape": TAPEEmbedding,
    "from_scratch_albert": LMEmbedding,
    "from_scratch_xlnet": LMEmbedding,
    "prot_albert": ProtAlbertBert,
    "prot_bert": ProtAlbertBert,
}


LM_CLASS = {"albert": AlbertForMaskedLM, "xlnet": XLNetLMHeadModel}


def sequence_masking(
    tokenized_sequence: str, intervals: List[Tuple[int, int]], number_of_masks: int
) -> List[List[str]]:

    """apply n masks on sequence within the interval.

    Args:
        tokenized_sequence: tokenized input sequence.
        intervals: range of positions within which we apply the mask
        number_of_masks: number of masks to apply.

    Returns:
        list of sequences with masks in position.
    """
    # randomly choose intervals to select single positions from
    # this include the case in qhich we want multiple masks in the same interval
    intervals_to_select_from = sorted(
        random.choices(intervals, k=number_of_masks),
        key=lambda interval: interval[0],
    )

    # select the positions to mask
    selected_pos_to_mask = list(
        set(
            [
                random.sample(range(interval[0], interval[1]), 1)[0]
                for interval in intervals_to_select_from
            ]
        )
    )
    selected_pos_to_mask = sorted(selected_pos_to_mask)

    # mask the sequence
    masked_sequence = " ".join(
        [
            "[MASK]" if i in selected_pos_to_mask else tokenized_sequence[i]
            for i in range(len(tokenized_sequence))
        ]
    )

    # extract the intervals
    masked_sequence_intervals = [
        masked_sequence.split()[start:end] for start, end in intervals
    ]

    return masked_sequence_intervals


class EnzymeOptimizer:
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
        """
        Args:
            scorer_filepath: pickled scorer filepath.
            substrate: substrate SMILES.
            product: product SMILES.
            sequence: AA sequence representing the enzyme to optimize.
            protein_embedding_type: type of protein embedding to use (Prot Albert, Brot Bert, TAPE, Prottrans, Albert from scratch, etc.)
            protein_embedding_path: path to the protein embedding model (just the directory)
            mutation_model_type: type of LM to use to suggest mutations. from_scratch_xnet or from_scratch_albert
            mutation_model_path: path to the LM use to suggest mutations
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
                    scorer_filepath=filepath, substrate=substrate, product=product, sequence=sequence,
                    protein_embedding_path="tape", protein_embedding_type="tape",
                    mutation_model_type="albert", mutation_model_path="path to LM folder"
                )
                # with this sequence length every steps takes ~5s
                # optimize between positions 150 and 405 allowing for a maximum of 5 mutations.
                results = designer.optimize(
                    number_of_mutations=5, number_of_steps=10, number_of_samples_per_step=8,
                    intervals=[(150, 405)], pad_intervals=True
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
        self.protein_embedding_type = protein_embedding_type.lower()
        try:
            if self.protein_embedding_type in ["prot_albert", "prot_bert"]:
                self.protein_embedding = PROTEIN_EMBEDDING_CLASS.get(
                    self.protein_embedding_type, ProtAlbertBert
                )(protein_embedding_path)
            elif self.protein_embedding_type == "from_scratch_xnet":
                self.protein_embedding = PROTEIN_EMBEDDING_CLASS.get(
                    self.protein_embedding_type, LMEmbedding
                )("xlnet", protein_embedding_path)
            elif self.protein_embedding_type == "from_scratch_albert":
                self.protein_embedding = PROTEIN_EMBEDDING_CLASS.get(
                    self.protein_embedding_type, LMEmbedding
                )("albert", protein_embedding_path)
            else:
                self.protein_embedding = PROTEIN_EMBEDDING_CLASS.get(
                    self.protein_embedding_type, TAPEEmbedding
                )()
        except TypeError:
            print(
                f"check the spelling of the protein embedder, otherwise chooce another one. Here are the protein embedders available {list(PROTEIN_EMBEDDING_CLASS.values())}"
            )
            print(
                f"Here are the protein embedders available {list(PROTEIN_EMBEDDING_CLASS.values())}"
            )

        self.molecule_embedding = molecule_embedding
        self.embedded_vectors = {
            "substrate": self.molecule_embedding.embed_one(self.substrate),
            "product": self.molecule_embedding.embed_one(self.product),
        }
        self.sequence = sequence
        self.sequence_length = len(sequence)

        self.mutation_model = LM_CLASS.get(
            mutation_model_type, AlbertForMaskedLM
        ).from_pretrained(mutation_model_path)
        self.mutation_model_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"{mutation_model_path}/tokenizer.json",
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
        )

    def score_sequence(self, sequence: str) -> float:
        """Score a given sequence.
        Args:
            sequence: a sequence to score.
        Returns:
            score for the sequence.
        """
        if self.protein_embedding_type == "prottrans":
            sequence = " ".join(list(sequence))
        embedded_vectors = {"sequence": self.protein_embedding.embed_one(sequence)}
        embedded_vectors.update(self.embedded_vectors)
        feature_vector = np.concatenate(
            [embedded_vectors[feature] for feature in self._ordering], axis=1
        )
        return self.scorer.predict_proba(feature_vector)[0][1]

    def score_sequences(self, sequences: List[str]) -> Dict[str, Any]:
        """Score a given sequence list.
        Args:
            sequences: a list of sequences to score.
        Returns:
            a list of dictionaries of sequences and related scores.
        """
        embedded_matrices = {
            "substrate": self.embedded_vectors["substrate"],
            "product": self.embedded_vectors["product"],
        }
        if self.protein_embedding_type == "prottrans":
            sequences = [" ".join(list(sequence)) for sequence in sequences]
        sequence_embedding = self.protein_embedding(sequences)

        if self.protein_embedding_type in ["prottrans", "tape"]:
            sequence_embedding = sequence_embedding / np.linalg.norm(sequence_embedding)
            sequence_embedding = sequence_embedding.mean(axis=0)

        embedded_matrices["sequence"] = sequence_embedding.reshape(
            1, sequence_embedding.shape[0]
        )
        feature_vector = np.concatenate(
            [embedded_matrices[feature] for feature in self._ordering], axis=1
        )

        return {
            "sequence": sequences,
            "score": self.scorer.predict_proba(feature_vector)[:, 1][0],
        }

    @abc.abstractclassmethod
    def optimize(
        self,
        number_of_mutations: int,
        intervals: Optional[List[Tuple[int, int]]] = None,
        number_of_steps: int = 10,
        number_of_samples_per_step: int = 32,
        number_of_sequences: Optional[int] = None,
        seed: int = 42,
        time_budget: Optional[int] = None,
        pad_intervals: bool = False,
        number_of_best_predictions: int = 2,
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

        for step in range(number_of_steps):
            logger.info(f"optimization step={step + 1}")

            for _ in range(number_of_samples_per_step):

                # extract the intervals
                masked_sequence_intervals = sequence_masking(
                    tokenized_seq, intervals, number_of_mutations
                )

                logger.info("mutating the fragments...")

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

                # combine the MASK predictions
                combination_lst: List[Any] = list(set(itertools.product(*best_fits)))

                logger.info("scoring the fragments...")
                for mutated_active_site in combination_lst:
                    mutated_active_site = list(mutated_active_site)
                    if mutated_active_site not in visited_sequences:
                        visited_sequences.append(mutated_active_site)

                        scored_sequence = self.score_sequences(mutated_active_site)
                        scored_sequence["intervals"] = intervals

                        if scored_sequence["score"] > original_sequence_score:
                            results += [scored_sequence]

            logger.info(
                f"best score at step={step + 1}: {max([scored_sequence['score'] for scored_sequence in results])}"
            )
            logger.info("\n")
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
