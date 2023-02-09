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
"""enzeptional - data processing utilities."""

import random
import warnings
from abc import ABC
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from tape.datasets import pad_sequences
from tape.registry import registry
from tape.tokenizers import TAPETokenizer
from transformers import (
    AlbertTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertTokenizer,
)


# from ..torch import device_claim
def device_claim(_):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


warnings.simplefilter(action="ignore", category=FutureWarning)

T = TypeVar("T")  # used for sample embedding


class AbstractEmbedding(ABC, Generic[T]):
    """Abstract embedding class."""
    model: Any

    def embed_one(self, sample: T) -> np.ndarray:
        """Embed one sample.

        Args:
            sample: sample representation.

        Returns:
            embedding vector for the sample.
        """
        raise NotImplementedError

    def embed_multiple(self, samples: List[T]) -> np.ndarray:
        """Embed multiple samples sample.

        Args:
            samples: a list of sample representations.

        Returns:
            embedding vectors for the samples.
        """
        raise NotImplementedError


StringEmbedding = AbstractEmbedding[str]


class AutoModelFromHFEmbedding(StringEmbedding):
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str],
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Initialize the HF transformers embedding class.
        Args:
            model_path: relative path of the model in hugginface or in local.
            tokenizer_path: relative path of the tokenizer in hugginface or in local.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """
        # get device
        self.device = device_claim(device)
        try:
            self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        except ValueError or RuntimeError:
            self.model = AutoModel.from_pretrained(model_path)

        self.model = self.model.to(self.device)
        self.model.eval()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception:
            try:
                self.tokenizer = AlbertTokenizer.from_pretrained(
                    tokenizer_path, do_lower_case=False
                )
            except OSError:
                self.tokenizer = BertTokenizer.from_pretrained(
                    tokenizer_path, do_lower_case=False
                )

    def __type__(self):
        return type(self.model)

    def embed_one(self, sample: str) -> np.ndarray:
        """Embed one protein sequence.
        Args:
            sample: a strings representing a molecule.
        Returns:
            a numpy array containing the embedding vector.
        """
        token_ids = self.tokenizer.encode_plus(
            sample, add_special_tokens=True, padding=True, return_tensors="pt"
        )
        return self.__call__(token_ids)

    def embed_multiple(self, samples: List[str]) -> np.ndarray:
        """Embed multiple protein sequences.
        Args:
            samples: a list of strings representing molecules.
        Returns:
            a numpy array containing the embedding vectors.
        """

        token_ids = self.tokenizer.batch_encode_plus(
            samples, add_special_tokens=True, padding=True
        )

        return self.__call__(token_ids)

    def __call__(self, token_ids) -> Any:

        input_data = {
            "input_ids": torch.as_tensor(np.array(token_ids["input_ids"])).to(
                self.device
            ),
            "attention_mask": torch.as_tensor(
                np.array(token_ids["attention_mask"])
            ).to(self.device),
        }

        with torch.no_grad():
            sequence_embeddings = self.model(**input_data)[0].cpu().detach().numpy()

        sequence_lenghts = input_data["attention_mask"].sum(1)
        
        return np.array(
            [
                sequence_embedding[:sequence_length].mean(0)
                for sequence_embedding, sequence_length in zip(  # type:ignore
                    sequence_embeddings, sequence_lenghts
                )
            ]
        )


class TAPEEmbedding(StringEmbedding):
    def __init__(
        self,
        model_type: str = "transformer",
        model_dir: str = "bert-base",
        aa_vocabulary: str = "iupac",
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Initialize the TAPE embedding class.
        Args:
            model_type: TAPE model type. Defaults to "transformer".
            model_dir: model directory. Defaults to "bert-base".
            aa_vocabulary: type of vocabulary. Defaults to "iupac".
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """
        # get device
        self.device = device_claim(device)
        # task and model definition
        self.task_specification = registry.get_task_spec("embed")
        self.model = registry.get_task_model(
            model_type, self.task_specification.name, load_dir=model_dir
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = TAPETokenizer(vocab=aa_vocabulary)
        
    def embed_one(self, sample: str) -> np.ndarray:
        """Embed one protein sequence.
        Args:
            sample: a strings representing a molecule.
        Returns:
            a numpy array containing the embedding vector.
        """
        
        return self.embed_multiple([sample])


    def embed_multiple(self, samples: List[str]) -> np.ndarray:
        """Embed multiple protein sequences.
        Args:
            samples: a list of strings representing molecules.
        Returns:
            a numpy array containing the embedding vectors.
        """

        token_ids = {}
        token_ids["ids"], token_ids["mask"] = zip(
            *[
                [
                    self.tokenizer.encode(sequence),
                    np.ones_like(self.tokenizer.encode(sequence)),
                ]
                for sequence in samples
            ]
        )
       

        return self.__call__(token_ids)

    def __call__(self, token_ids) -> Any:

        input_data = {
            "input_ids": torch.from_numpy(pad_sequences(token_ids["ids"])).to(
                self.device
            ),
            "input_mask": torch.from_numpy(pad_sequences(token_ids["mask"])).to(
                self.device
            ),
        }

        with torch.no_grad():
            sequence_embeddings = self.model(**input_data)[0].cpu().detach().numpy()

        sequence_lenghts = input_data["input_mask"].sum(1)

        return np.array(
            [
                sequence_embedding[:sequence_length].mean(0)
                for sequence_embedding, sequence_length in zip(  # type:ignore
                    sequence_embeddings, sequence_lenghts
                )
            ]
        )



def mutate_sequence_with_variant(sequence: str, variant: str) -> str:
    """Given an AA sequence and a variant returns the mutated AA sequence.

    Args:
        sequence: an AA sequence.
        variant: a variant annotation.

    Returns:
        the mutated sequence.
    """
    edits = [
        (int(variant_string[1:-1]), variant[0], variant_string[-1])
        for variant_string in map(str.strip, variant.split("/"))
    ]
    mutated_sequence = list(sequence)
    for index, _, aa_to in edits:
        mutated_sequence[index] = aa_to
    return "".join(mutated_sequence)


def sanitize_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sanitize intervals merging overlapping ones and sorting them.
    Args:
        intervals: intervals to sanitize.
    Returns:
        sorted and non overlapping intervals.
    """
    if len(intervals) > 1:
        sorted_intervals = sorted(intervals, key=lambda interval: interval[0])
        merged_intervals = []
        previous = sorted_intervals[0]
        for pos in range(1, len(sorted_intervals)):
            current = sorted_intervals[pos]
            previous_end = previous[1]
            if current[0] <= previous_end:
                previous_end = max(previous_end, current[1])
                previous = (previous[0], previous_end)
                if pos == len(sorted_intervals) - 1:
                    merged_intervals.append(previous)
            else:
                merged_intervals.append(previous)
                previous = current
                if pos == len(sorted_intervals) - 1:
                    merged_intervals.append(current)
        return merged_intervals
    else:
        return intervals


def round_up(number):
    return int(number) + (number % 1 > 0)


def sanitize_intervals_with_padding(
    intervals: List[Tuple[int, int]],
    sequence_length: int = 64,
    minimum_interval_length: int = 8,
) -> List[Tuple[int, int]]:
    """Sanitize intervals padding the intervals that contain less then 8 AAs to 8.

    Args:
        intervals: intervals to sanitize.

    Returns:
        sorted, non overlapping, and padded intervals.
    """

    clean_intervals = sanitize_intervals(intervals)
    padded_intervals = []
    for start, end in clean_intervals:
        diff = end - start
        if diff < minimum_interval_length:
            to_pad = round_up((minimum_interval_length - diff) / 2)
            if start - to_pad < 0:
                new_start = 0
                if end + to_pad > sequence_length:
                    new_end = sequence_length
                else:
                    if end + (to_pad * 2) <= sequence_length:
                        new_end = end + (to_pad * 2)
                    else:
                        new_end = end + to_pad

                padded_intervals.append((new_start, new_end))
                if new_end - new_start < minimum_interval_length:
                    warnings.warn("Warning. an dinterval has less than 8 AA")
            else:
                if end + to_pad > sequence_length:
                    new_end = sequence_length
                    if start - (to_pad * 2) >= 0:
                        new_start = start - (to_pad * 2)
                    else:
                        new_start = start - to_pad
                    padded_intervals.append((new_start, new_end))
                    if new_end - new_start < minimum_interval_length:
                        warnings.warn("Warning. an dinterval has less than 8 AA")
                else:
                    padded_intervals.append((start - to_pad, end + to_pad))
        else:
            padded_intervals.append((start, end))
    return sanitize_intervals(padded_intervals)


def reconstruct_sequence_with_mutation_range(
    sequence: str, mutated_sequence_range: str, intervals: List[Tuple[int, int]]
) -> str:
    """Reconstruct a sequence replacing in given positions sub-sequences from a mutated range.

    Args:
        sequence: original sequence.
        mutated_sequence_range: mutated sequence range.
        intervals: sorted and non overlapping intervals.

    Returns:
        reconstructed sequence.
    """
    # create the mutated sequence, considering sorted intervals
    mutated_sequence = list(sequence)
    # split fragments by intervals
    for start, end in intervals:
        size_fragment = end - start
        mutated_sequence[start:end] = list(mutated_sequence_range[:size_fragment])
        mutated_sequence_range = mutated_sequence_range[size_fragment:]
    return "".join(mutated_sequence)


class SelectionGenerator:
    def __init__(self) -> None:
        pass

    def selection(
        self, scores: List[Dict[str, Any]], k: Optional[int] = -1
    ) -> List[Any]:
        """Select the top k mutated sequences based on the score
        Args:
            scores: dictionary containing sequences and scores.
            k: number of top sequences to return.
        return:
            The top k sequences
        """
        return list(sorted(scores, key=lambda d: d["score"], reverse=True))[:k]


class CrossoverGenerator:
    def __init__(
        self,
        threshold_probability: float = 0.5) -> None:
        
        self.threshold_probability = threshold_probability
        
        '''
        threshold_probability: threshold probability of exchange of an AA between the two input sequences
        '''
        
        pass

    def single_point_crossover(
        self, a_sequence: str, another_sequence: str
    ) -> Tuple[str, str]:
        """Given two sequences perform a single point crossover

        Args:
            a_sequence: a sequence.
            another_sequence: a sequence

        Returns:
            return two sequences that are the crossover of the input sequences
        """

        # select crossover point that is not at the end of the string
        random_point = random.randint(1, len(a_sequence) - 2)
        new_sequence_1 = "".join(
            np.append(a_sequence[:random_point], another_sequence[random_point:])
        )
        new_sequence_2 = "".join(
            np.append(another_sequence[:random_point], a_sequence[random_point:])
        )
        return new_sequence_1, new_sequence_2

    def uniform_crossover(
        self, a_sequence: str, another_sequence: str, 
    ) -> Tuple[str, str]:
        """Given two sequences perform an uniform crossover

        Args:
            a_sequence: a sequence.
            another_sequence: a sequence.

        Returns:
            return two sequences that are the crossover of the input sequences
        """
        list_a_sequence = list(a_sequence)
        list_another_sequence = list(another_sequence)

        for pos in range(len(list_a_sequence)):
            tmp_prob = random.uniform(0, 1)
            if tmp_prob > self.threshold_probability:
                tmp = list_a_sequence[pos]
                list_a_sequence[pos] = list_another_sequence[pos]
                list_another_sequence[pos] = tmp

        return "".join(list_a_sequence), "".join(list_another_sequence)

