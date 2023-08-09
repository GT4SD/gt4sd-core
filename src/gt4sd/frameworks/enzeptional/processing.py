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
import os
import torch
import random
import warnings
from abc import ABC
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union
import numpy as np
from tape.datasets import pad_sequences
from tape.registry import registry
from tape.tokenizers import TAPETokenizer
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    T5Model,
    T5Tokenizer,
)

os.environ["TRANSFORMERS_CACHE"] = "/dccstor/yna/.cache/"
torch.hub.set_dir("/dccstor/yna/.cache/torch/hub")


def device_claim(device: Optional[Union[torch.device, str]]) -> torch.device:
    """
    Claims the appropriate device for inference.

    Args:
        device: Device where the inference is running, either as a dedicated class or a string.

    Returns:
        The claimed device.
    """
    return torch.device(
        "cuda:0" if torch.cuda.is_available() and device != "cpu" else "cpu"
    )


warnings.simplefilter(action="ignore", category=FutureWarning)

T = TypeVar("T")  # used for sample embedding


class AbstractEmbedding(ABC, Generic[T]):
    """Abstract embedding class."""

    model: Any

    def embed(self, samples: List[T]) -> np.ndarray:
        """
        Embed multiple samples.

        Args:
            samples: A list of sample representations.

        Returns:
            Embedding vectors for the samples.
        """
        raise NotImplementedError


class StringEmbedding(AbstractEmbedding[str]):
    pass


class AutoModelFromHFEmbedding(StringEmbedding):
    def __init__(
        self,
        model_kwargs: Dict[str, Any] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """
        Initialize the HF transformers embedding class.

        Args:
            model_kwargs: Contains arguments to load the model and tokenizer from Hugging Face.
                The following arguments are required:
                    - pretrained_model_name_or_path: Path to the pretrained model.
                    - tokenizer_path: Path to the tokenizer.
                Optional arguments:
                    - model_name: Name of the model to load. If not provided, it is set to None.
                    - cache_dir: Path to the cache directory. If not provided, it is set to None.
            device: Device where the inference is running, either as a dedicated class or a string.
        """
        self.device = device_claim(device)
        self.model_path = model_kwargs.get("model_path", "facebook/esm2_t33_650M_UR50D")
        self.tokenizer_path = model_kwargs.get(
            "tokenizer_path", "facebook/esm2_t33_650M_UR50D"
        )
        self.cache_dir = model_kwargs.get("cache_dir", None)

        if self.cache_dir is not None:
            os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
            model_kwargs = {
                "pretrained_model_name_or_path": self.model_path,
                "cache_dir": self.cache_dir,
            }
        else:
            model_kwargs = {"pretrained_model_name_or_path": self.model_path}

        self.model = AutoModel.from_pretrained(**model_kwargs).to(self.device).eval()
        if self.device.type == "cuda:0":
            self.model.cuda()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        except Exception:
            self.tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def __type__(self):
        return type(self.model)

    def embed(self, samples: List[str]) -> np.ndarray:
        """
        Embed one or multiple samples.

        Args:
            samples: List of sequences.

        Returns:
            Embedding vector for each sample.
        """
        inputs = self.tokenizer(
            samples, add_special_tokens=True, padding=True, return_tensors="pt"
        )

        with torch.no_grad():
            sequence_embeddings = (
                self.model.encoder(**inputs.to(self.device))[0].cpu().detach().numpy()
                if self.__type__() == T5Model
                else self.model(**inputs.to(self.device))[0].cpu().detach().numpy()
            )

        sequence_lengths = inputs["attention_mask"].sum(1)

        return np.array(
            [
                sequence_embedding[:sequence_length].mean(0)
                for sequence_embedding, sequence_length in zip(
                    sequence_embeddings, sequence_lengths
                )
            ]
        )


class TAPEEmbedding(StringEmbedding):
    def __init__(
        self,
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """
        Initialize the TAPE embedding class.

        Args:
            model_kwargs: Contains arguments to load the model and tokenizer from Hugging Face.
                The following arguments are required:
                    - model_type: TAPE model type. Defaults to "transformer".
                    - model_dir: Model directory. Defaults to "bert-base".
                    - aa_vocabulary: Type of vocabulary. Defaults to "iupac".
            device: Device where the inference is running, either as a dedicated class or a string.
                If not provided, it is inferred.
        """
        self.device = device_claim(device)
        self.task_specification = registry.get_task_spec("embed")
        self.model = registry.get_task_model(
            model_kwargs.get("model_path") if model_kwargs is not None else "transformer",
            self.task_specification.name,
            load_dir=model_kwargs.get("model_dir") if model_kwargs is not None else "bert-base"
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = TAPETokenizer(vocab=model_kwargs.get("aa_vocabulary") if model_kwargs is not None else "iupac")

    def __type__(self):
        return type(self.model)

    def embed(self, samples: List[str]) -> np.ndarray:
        """
        Embed one or multiple samples.

        Args:
            samples: List of sequences.

        Returns:
            Embedding vector for each sample.
        """
        token_ids: Dict[str, List[int]] = {"ids": [], "mask": []}
        for sequence in samples:
            encoded_sequence = self.tokenizer.encode(sequence)
            token_ids["ids"].append(encoded_sequence)
            token_ids["mask"].append(np.ones_like(encoded_sequence))

        input_ids = torch.from_numpy(pad_sequences(token_ids["ids"])).to(self.device)
        input_mask = torch.from_numpy(pad_sequences(token_ids["mask"])).to(self.device)

        inputs = {
            "input_ids": input_ids,
            "input_mask": input_mask,
        }

        with torch.no_grad():
            sequence_embeddings = self.model(**inputs)[0].cpu().detach().numpy()

        sequence_lengths = input_mask.sum(1)

        return np.array(
            [
                sequence_embedding[:sequence_length].mean(0)
                for sequence_embedding, sequence_length in zip(
                    sequence_embeddings, sequence_lengths
                )
            ]
        )


class Unmasker(ABC):
    def __init__(
        self,
        model_kwargs: Dict[str, Any] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """
        Initialize the Unmasker class.

        Args:
            model_kwargs: Contains arguments to load the model and tokenizer from Hugging Face.
                The following arguments are required:
                    - model_path: Path to the pretrained model.
                    - tokenizer_path: Path to the tokenizer.
                Optional arguments:
                    - cache_dir: Path to the cache directory.
            device: Device where the inference is running, either as a dedicated class or a string.
                If not provided, it is inferred.
        """
        self.device = device_claim(device)
        self.model_path = model_kwargs.get("model_path", None)
        self.tokenizer_path = model_kwargs.get("tokenizer_path", None)
        self.cache_dir = model_kwargs.get("cache_dir", None)

        if self.cache_dir is not None:
            model_kwargs = {
                "pretrained_model_name_or_path": self.model_path,
                "cache_dir": self.cache_dir,
            }
        else:
            model_kwargs = {"pretrained_model_name_or_path": self.model_path}

        self.model = (
            AutoModelForMaskedLM.from_pretrained(**model_kwargs).to(self.device).eval()
        )
        if self.device.type == "cuda:0":
            self.model.cuda()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        except Exception:
            self.tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_path)

    def unmask(self, sequence: str, top_k: int = 2) -> List[List[str]]:
        """
        Unmask a sequence.

        Args:
            sequence: Input sequence with masked tokens.
            top_k: Number of top predicted tokens to return for each masked position. Defaults to 2.

        Returns:
            List of lists containing top predicted tokens for each masked position.
        """
        inputs = self.tokenizer(
            sequence, return_tensors="pt", add_special_tokens=True, padding=True
        )
        mask_token_index = torch.where(
            inputs["input_ids"] == self.tokenizer.mask_token_id
        )[1]
        logits = self.model(inputs["input_ids"], return_dict=True)["logits"]
        mask_token_logits = logits[0, mask_token_index, :]

        top_tokens = []
        for i in range(len(mask_token_index)):
            top_n_tokens = (
                torch.topk(mask_token_logits, top_k, dim=1).indices[i].tolist()
            )
            tmp_top_tokens = []
            for token in top_n_tokens:
                tmp_top_tokens.append(self.tokenizer.decode([token]))
            top_tokens.append(tmp_top_tokens)

        return top_tokens


def mutate_sequence_with_variant(sequence: str, variant: str) -> str:
    """
    Mutate an AA sequence with a variant.

    Args:
        sequence: AA sequence.
        variant: Variant annotation.

    Returns:
        Mutated AA sequence.
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
    """
    Merge overlapping intervals.

    Args:
        intervals: List of tuples (int, int) representing intervals.

    Returns:
        Merged intervals.
    """
    intervals.sort(key=lambda x: x[0])
    merged: List[Tuple[int, int]] = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
    return sorted(merged, key=lambda x: x[0])


def round_up(number: float) -> int:
    """
    Round up a floating-point number to the nearest integer.

    Args:
        number: Floating-point number.

    Returns:
        Nearest integer rounded up.
    """
    return int(number) + (number % 1 > 0)


def sanitize_intervals_with_padding(
    intervals: List[Tuple[int, int]], pad_value: int, max_value: int
) -> List[Tuple[int, int]]:
    """
    Sanitize intervals with padding.

    Args:
        intervals: List of tuples (int, int) representing intervals.
        pad_value: Value to pad the intervals.
        max_value: Maximum length of the sequence.

    Returns:
        Padded intervals.
    """
    padded_intervals = []
    for interval in intervals:
        diff = interval[1] - interval[0]
        value_to_pad = int((pad_value - diff) / 2)
        if value_to_pad >= 0:
            left = interval[0] - value_to_pad
            right = interval[1] + value_to_pad
            if 0 <= left < right <= max_value:
                padded_intervals.append((left, right))
            elif left < 0 and right <= max_value:
                padded_intervals.append((0, right))
            elif 0 <= left < max_value < right:
                padded_intervals.append((left, max_value))
            else:
                padded_intervals.append((0, max_value))
        else:
            padded_intervals.append(interval)
    return sanitize_intervals(padded_intervals)


def reconstruct_sequence_with_mutation_range(
    sequence: str, mutated_sequence_range: str, intervals: List[Tuple[int, int]]
) -> str:
    """
    Reconstruct a sequence by replacing sub-sequences from a mutated range at given positions.

    Args:
        sequence: Original sequence.
        mutated_sequence_range: Mutated sequence range.
        intervals: Sorted and non-overlapping intervals.

    Returns:
        Reconstructed sequence.
    """
    mutated_sequence = list(sequence)
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
        """
        Select the top k mutated sequences based on the score.

        Args:
            scores: Dictionary containing sequences and scores.
            k: Number of top sequences to return.

        Returns:
            The top k sequences.
        """
        return list(sorted(scores, key=lambda d: d["score"], reverse=True))[:k]


class CrossoverGenerator:
    def __init__(self, threshold_probability: float = 0.5) -> None:
        """
        Initialize the CrossoverGenerator class.

        Args:
            threshold_probability: Threshold probability of exchange of an AA between the two input sequences.
        """
        self.threshold_probability = threshold_probability

    def single_point_crossover(
        self, a_sequence: str, another_sequence: str
    ) -> Tuple[str, str]:
        """
        Perform a single point crossover between two sequences.

        Args:
            a_sequence: First sequence.
            another_sequence: Second sequence.

        Returns:
            Two sequences that are the crossover of the input sequences.
        """
        random_point = random.randint(1, len(a_sequence) - 2)
        new_sequence_1 = "".join(
            np.append(a_sequence[:random_point], another_sequence[random_point:])
        )
        new_sequence_2 = "".join(
            np.append(another_sequence[:random_point], a_sequence[random_point:])
        )
        return new_sequence_1, new_sequence_2

    def uniform_crossover(
        self, a_sequence: str, another_sequence: str
    ) -> Tuple[str, str]:
        """
        Perform a uniform crossover between two sequences.

        Args:
            a_sequence: First sequence.
            another_sequence: Second sequence.

        Returns:
            Two sequences that are the crossover of the input sequences.
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
