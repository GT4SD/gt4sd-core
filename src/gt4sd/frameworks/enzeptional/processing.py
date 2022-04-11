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

from abc import ABC
from typing import Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from tape.datasets import pad_sequences
from tape.registry import registry
from tape.tokenizers import TAPETokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer

from ..torch import device_claim

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

    def _encode_and_mask(self, sequence: str) -> Tuple[np.ndarray, np.ndarray]:
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
        self.model = AutoModelWithLMHead.from_pretrained(tokenizer_name)
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
    merged_intervals = [sorted_intervals[0]]
    for current in sorted_intervals:
        previous_end = merged_intervals[-1][1]
        if current[0] <= previous_end:
            previous_end = max(previous_end, current[1])
        else:
            merged_intervals.append(current)
    return merged_intervals


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
