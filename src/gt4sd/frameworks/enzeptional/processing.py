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
"""enzeptional - data processing utilities."""


import random
import warnings
from abc import ABC
from collections import defaultdict
from itertools import chain
from operator import methodcaller
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from tape.datasets import pad_sequences
from tape.registry import registry
from tape.tokenizers import TAPETokenizer
from transformers import (
    AlbertForMaskedLM,
    AlbertModel,
    AlbertTokenizer,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    PreTrainedTokenizerFast,
    T5EncoderModel,
    T5Tokenizer,
    XLNetLMHeadModel,
)


# from ..torch import device_claim
def device_claim(_):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def merge_dictionaries(
    list_of_dictionaries: List[Dict[str, Any]]
) -> Dict[str, List[Any]]:
    # initialise defaultdict of lists
    output = defaultdict(list)

    # iterate dictionary items
    dictionary_items = map(methodcaller("items"), list_of_dictionaries)
    for k, v in chain.from_iterable(dictionary_items):
        output[k].append(v)

    return output


T = TypeVar("T")  # used for sample embedding


class Embedding(ABC, Generic[T]):
    """Abstract embedding class."""

    def embed_one(self, sample: T) -> np.ndarray:
        """Embed one sample.

        Args:
            sample: sample representation.

        Returns:
            embedding vector for the sample.
        """
        return self.__call__([sample])

    def __call__(self, samples: List[T]) -> np.ndarray:
        """Embed multiple samples sample.

        Args:
            samples: a list of sample representations.

        Returns:
            embedding vectors for the samples.
        """
        raise NotImplementedError


StringEmbedding = Embedding[str]


class TAPEEmbedding(StringEmbedding):
    """Embed AA sequence using TAPE."""

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

    def _encode_and_mask(self, sequence: str) -> Tuple[Any, Any]:
        """Encode and mask a sequence.
        Args:
            sequence: AA sequence.
        Returns:
            a tuple containing the token ids and the mask.
        """
        token_ids = self.tokenizer.encode(sequence)
        return token_ids, np.ones_like(token_ids)

    def __call__(self, samples: List[str]) -> np.ndarray:
        """Embed multiple protein sequences using TAPE.
        Args:
            samples: a list of protein sequences.
        Returns:
            a numpy array containing the embedding vectors.
        """
        # prepare input
        token_ids, masks = zip(
            *[self._encode_and_mask(sequence) for sequence in samples]
        )
        input_data = {
            "input_ids": torch.from_numpy(pad_sequences(token_ids)).to(self.device),
            "input_mask": torch.from_numpy(pad_sequences(masks)).to(self.device),
        }
        sequence_lenghts = input_data["input_mask"].sum(1)
        sequence_embeddings = self.model(**input_data)[0].cpu().detach().numpy()
        # get average embedding
        return np.array(
            [
                sequence_embedding[:sequence_length].mean(0)
                for sequence_embedding, sequence_length in zip(  # type:ignore
                    sequence_embeddings, sequence_lenghts
                )
            ]
        )


class LMEmbedding(StringEmbedding):
    """Embed a string representation of a molecule using a LM model"""

    def __init__(
        self,
        model: str,
        path_LM_model: str,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Initialize the ProtTrans embedding class.

        Args:
            model_name: Langauge model name.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """

        # get device
        self.device = device_claim(device)
        # tokenizer and model definition

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"{path_LM_model}/tokenizer.json",
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
        )
        if "albert" in model:
            self.model = AlbertForMaskedLM.from_pretrained(path_LM_model)
        else:
            self.model = XLNetLMHeadModel.from_pretrained(path_LM_model)

        self.model = self.model.to(self.device)
        self.model.eval()

    def __call__(self, samples: List[str]) -> np.ndarray:
        """Embed multiple protein sequences using ProtTrans Xl model.

        Args:
            samples: a list of strings representing molecules.
                        Words/characters in the string mush be separated by a white space

        Returns:
            a numpy array containing the embedding vectors.
        """

        # prepare input
        token_ids = self.tokenizer.batch_encode_plus(
            samples, add_special_tokens=True, padding=True
        )

        input_ids = torch.tensor(token_ids["input_ids"]).to(self.device)
        attention_mask = torch.tensor(token_ids["attention_mask"]).to(self.device)
        with torch.no_grad():
            embedding = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding[0].cpu().detach().numpy()
        sequence_lenghts = [len(seq) for seq in samples]

        embeddings = np.array(
            [
                sequence_embedding[:sequence_length].mean(0)
                for sequence_embedding, sequence_length in zip(  # type:ignore
                    embedding, sequence_lenghts
                )
            ]
        )

        embeddings = embeddings / np.linalg.norm(embeddings)
        embeddings = embeddings.mean(axis=0)
        return embeddings


class ProtTransXL(StringEmbedding):
    """Embed a string representation of a molecule using a ProtTrans_XLnet model"""

    def __init__(
        self,
        model_name: str = "Rostlab/prot_t5_xl_uniref50",
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Initialize the ProtTrans embedding class.

        Args:
            model_name: ProtTrans model name.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """

        self.model_name = model_name
        # get device
        self.device = device_claim(device)
        # tokenizer and model definition
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_name, do_lower_case=False
        )
        self.model = T5EncoderModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    def __call__(self, samples: List[str]) -> np.ndarray:
        """Embed multiple protein sequences using ProtTrans Xl model.

        Args:
            samples: a list of strings representing molecules.
                        Words/characters in the string mush be separated by a white space

        Returns:
            a numpy array containing the embedding vectors.
        """

        # prepare input
        if len(samples) < 2:
            token_ids = self.tokenizer.encode_plus(
                samples, add_special_tokens=True, padding=True, return_tensors="pt"
            )
        else:
            token_ids = self.tokenizer.batch_encode_plus(
                samples, add_special_tokens=True, padding=True
            )

        input_ids = torch.tensor(token_ids["input_ids"]).to(self.device)
        attention_mask = torch.tensor(token_ids["attention_mask"]).to(self.device)

        with torch.no_grad():
            embedding = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # For feature extraction we recommend to use the encoder embedding
        sequence_embeddings = embedding.last_hidden_state.cpu().numpy()
        # decoder_embedding = embedding[0].cpu().numpy()

        sequence_lenghts = attention_mask.sum(1)
        # get average embedding
        return np.array(
            [
                sequence_embedding[:sequence_length].mean(0)
                for sequence_embedding, sequence_length in zip(  # type:ignore
                    sequence_embeddings, sequence_lenghts
                )
            ]
        )


class ProtAlbertBert(StringEmbedding):
    """Embed a string representation of a molecule using an HF transformers model."""

    def __init__(
        self,
        model_path: str,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Initialize the HF transformers embedding class.
        Args:
            model_path: relative path of the model in hugginface.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """
        # get device
        self.device = device_claim(device)
        # tokenizer and model definition
        if "albert" in model_path:
            self.tokenizer = AlbertTokenizer.from_pretrained(
                model_path, do_lower_case=False
            )
            self.model = AlbertModel.from_pretrained(model_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(
                model_path, do_lower_case=False
            )
            self.model = BertModel.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def __call__(self, samples: List[str]) -> np.ndarray:
        """Embed multiple protein sequences using TAPE.
        Args:
            samples: a list of strings representing molecules.
        Returns:
            a numpy array containing the embedding vectors.
        """
        if len(samples) < 2:
            token_ids = self.tokenizer.encode_plus(
                samples,
                add_special_tokens=True,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            token_ids = self.tokenizer.batch_encode_plus(
                samples, add_special_tokens=True, padding="max_length"
            )

        input_ids = torch.tensor(token_ids["input_ids"]).to(self.device)
        attention_mask = torch.tensor(token_ids["attention_mask"]).to(self.device)

        with torch.no_grad():
            embedding = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # For feature extraction we recommend to use the encoder embedding
        sequence_embeddings = embedding[0].cpu().numpy()

        sequence_lenghts = attention_mask.sum(1)
        # get average embedding
        return np.array(
            [
                sequence_embedding[:sequence_length].mean(0)
                for sequence_embedding, sequence_length in zip(  # type:ignore
                    sequence_embeddings, sequence_lenghts
                )
            ]
        )


class HuggingFaceTransformerEmbedding(StringEmbedding):
    """Embed a string representation of a molecule using an HF transformers model."""

    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        tokenizer_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Initialize the HF transformers embedding class.
        Args:
            model_name: model name. Defaults to "seyonec/ChemBERTa-zinc-base-v1".
            tokenizer_name: tokenizer name. Defaults to "seyonec/ChemBERTa-zinc-base-v1".
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """
        # get device
        self.device = device_claim(device)
        # tokenizer and model definition
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(tokenizer_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    def __call__(self, samples: List[str]) -> np.ndarray:
        """Embed multiple protein sequences using TAPE.
        Args:
            samples: a list of strings representing molecules.
        Returns:
            a numpy array containing the embedding vectors.
        """
        # get the CLS token representation from each SMILES.
        return (
            self.model(
                **{
                    key: tensor.to(self.device)
                    for key, tensor in self.tokenizer(
                        samples, return_tensors="pt", padding=True
                    ).items()
                }
            )[0][:, 0, :]
            .detach()
            .cpu()
            .numpy()
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
            merged_intervals.append(current)
    return merged_intervals


def round_up(number):
    return int(number) + (number % 1 > 0)


def sanitize_intervals_with_padding(
    intervals: List[Tuple[int, int]], sequence_length: int = 64
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
        if diff < 8:
            to_pad = round_up((8 - diff) / 2)
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
                if new_end - new_start < 8:
                    warnings.warn("Warning. an dinterval has less than 8 AA")
            else:
                if end + to_pad > sequence_length:
                    new_end = sequence_length
                    if start - (to_pad * 2) >= 0:
                        new_start = start - (to_pad * 2)
                    else:
                        new_start = start - to_pad
                    padded_intervals.append((new_start, new_end))
                    if new_end - new_start < 8:
                        warnings.warn("Warning. an dinterval has less than 8 AA")
                else:
                    padded_intervals.append((start - to_pad, end + to_pad))
        else:
            padded_intervals.append((start, end))
    return sanitize_intervals(padded_intervals)


def reconstruct_sequence_with_mutation_range(
    sequence: str, mutated_sequence_range: str, intervals: List[Tuple[int, int]]
):
    """Reconstruct a sequence replacing in given positions sub-sequences from a mutated range.

    Args:
        sequence: original sequence.
        mutated_sequence_range: mutated sequence range.
        intervals: sorted and non overlapping intervals.

    Returns:
        reconstructed sequence.
    """
    # create the mutated sequence, considering sorted intervals
    mutated_range_offset = 0  # offset with respect to the mutated_sequence_range
    mutated_sequence_offset = 0  # offset with respect to the full mutated sequence.
    mutated_sequence = ""
    for start, end in intervals:
        mutated_sequence += sequence[mutated_sequence_offset:start]
        chunk_length = end - start + 1
        mutated_sequence += mutated_sequence_range[
            mutated_range_offset : mutated_range_offset + chunk_length
        ]
        mutated_range_offset += chunk_length
        mutated_sequence_offset = end + 1
    mutated_sequence += sequence[end + 1 :]
    return mutated_sequence


def selection(scores: List[Dict[str, Any]], k: int = 20) -> List[Any]:
    """Select the top k mutated sequences based on the score

    Args:
        scores: dictionary containing sequences and scores.
        K: number of top sequences to return

    return:
        retuen the top K sequences

    """

    res = list(sorted(scores, key=lambda d: d["score"], reverse=True))[:k]

    return res


def single_point_crossover(sequence_1: str, sequence_2: str) -> Tuple[str, str]:
    """Given two sequences perform a single point crossover

    Args:
        sequence_1: a sequence.
        sequence_2: a sequence

    Returns:
        return two sequences that are the crossover of the input sequences
    """

    # select crossover point that is not at the end of the string
    random_point = random.randint(1, len(sequence_1) - 2)
    new_sequence_1 = "".join(
        np.append(sequence_1[:random_point], sequence_2[random_point:])
    )
    new_sequence_2 = "".join(
        np.append(sequence_1[random_point:], sequence_2[:random_point])
    )
    return new_sequence_1, new_sequence_2


def uniform_crossover(
    sequence_1: List[str], sequence_2: List[str], p: float
) -> Tuple[str, str]:
    """Given two sequences perform an uniform crossover

    Args:
        sequence_1: a sequence.
        sequence_2: a sequence.
        p: threshold probability of exchange of an AA between the two input sequences

    Returns:
        return two sequences that are the crossover of the input sequences
    """

    sequence_1 = sequence_1
    sequence_2 = sequence_2

    for pos in range(len(sequence_1)):
        tmp_prob = random.uniform(0, 1)
        if tmp_prob > p:
            tmp = sequence_1[pos]
            sequence_1[pos] = sequence_2[pos]
            sequence_2[pos] = tmp

    return "".join(sequence_1), "".join(sequence_2)
