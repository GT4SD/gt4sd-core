#
# MIT License
#
# Copyright (c) 2024 GT4SD team
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
from abc import ABC
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from tape.datasets import pad_sequences
from tape.registry import registry
from tape.tokenizers import TAPETokenizer
from transformers import (
    AutoModel,
    EsmForMaskedLM,
    AutoTokenizer,
    T5Tokenizer,
)
import math
import random
import logging
from itertools import product as iter_product
from gt4sd.frameworks.torch import get_device


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# os.environ["TRANSFORMERS_CACHE"] = "~/.cache/huggingface/"
# torch.hub.set_dir("/dccstor/yna/.cache/torch/hub")


class ModelCache:
    """
    A simple cache mechanism for storing and retrieving models.
    """

    def __init__(self):
        """
        Initializes the cache as an empty dictionary.
        """
        self.cache = {}

    def get(self, key):
        """
        Retrieves a model from the cache using the given key.

        Args:
            key: The key used to store the model.

        Returns:
            The model associated with the key, or None if not found.
        """
        return self.cache.get(key)

    def add(self, key, model):
        """
        Adds a model to the cache with the specified key.

        Args:
            key: The key to associate with the model.
            model: The model to be cached.
        """
        self.cache[key] = model


ENZEPTIONAL_MODEL_CACHE = ModelCache()


class StringEmbedding(ABC):
    """
    Abstract base class for embedding string data.

    Attributes:
        model (Any): The embedding model.
    """

    model: Any

    def embed(self, samples: List[str]) -> np.ndarray:
        """Abstract method for embedding a list of string samples.

        Args:
            samples (List[str]): The list of strings to be embedded.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError


class HFandTAPEModelUtility(StringEmbedding):
    """
    Utility class for handling both Hugging Face and TAPE models for embedding
    and unmasking tasks.
    """

    def __init__(
        self,
        embedding_model_path: str,
        tokenizer_path: str,
        unmasking_model_path: Optional[str] = None,
        is_tape_model: bool = False,
        device: Optional[Union[torch.device, str]] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initializes the utility with specified model and tokenizer paths.

        Args:
            embedding_model_path (str): Path to the embedding model.
            tokenizer_path (str): Path to the tokenizer.
            unmasking_model_path (Optional[str], optional): Path to the unmasking model, if applicable. Defaults to None.
            is_tape_model (bool, optional): Flag to indicate if a TAPE model is being used. Defaults to False.
            device (Optional[Union[torch.device, str]], optional): The compute device to use ('cpu' or 'cuda:0'). Defaults to None.
            cache_dir (Optional[str], optional): Path to cache directory. Defaults to None.
        """
        self.device = get_device()
        self.is_tape_model = is_tape_model

        embedding_cache_key = f"embedding_{embedding_model_path}"
        self.embedding_model = ENZEPTIONAL_MODEL_CACHE.get(embedding_cache_key)
        if not self.embedding_model:
            if is_tape_model:
                self.embedding_model = registry.get_task_model(
                    embedding_model_path,
                    "embed",
                    load_dir=embedding_model_path,
                ).to(self.device)
            else:
                if cache_dir:
                    self.embedding_model = (
                        AutoModel.from_pretrained(
                            embedding_model_path,
                            cache_dir=cache_dir,
                        )
                        .to(self.device)
                        .eval()
                    )
                else:
                    self.embedding_model = (
                        AutoModel.from_pretrained(
                            embedding_model_path,
                        )
                        .to(self.device)
                        .eval()
                    )

                ENZEPTIONAL_MODEL_CACHE.add(embedding_cache_key, self.embedding_model)

        if unmasking_model_path is not None:
            unmasking_cache_key = f"unmasking_{unmasking_model_path}"
            self.unmasking_model = ENZEPTIONAL_MODEL_CACHE.get(unmasking_cache_key)
            if not self.unmasking_model:
                if cache_dir:
                    self.unmasking_model = (
                        EsmForMaskedLM.from_pretrained(
                            unmasking_model_path,
                            cache_dir=cache_dir,
                        )
                        .to(self.device)
                        .eval()
                    )
                else:
                    self.unmasking_model = (
                        EsmForMaskedLM.from_pretrained(
                            unmasking_model_path,
                        )
                        .to(self.device)
                        .eval()
                    )
                ENZEPTIONAL_MODEL_CACHE.add(unmasking_cache_key, self.unmasking_model)
        else:
            logger.error("No Unmasking model loaded. Check you model inputs")

        if is_tape_model:
            self.tokenizer = TAPETokenizer(vocab="iupac")
        else:
            self.tokenizer = self._load_tokenizer(tokenizer_path)

    def _load_tokenizer(self, tokenizer_path: str):
        """Loads a tokenizer based on the given path, caching it for future use.

        Args:
            tokenizer_path (str): Path to the tokenizer.

        Returns:
            The loaded tokenizer
        """
        tokenizer_cache_key = f"tokenizer_{tokenizer_path}"
        tokenizer = ENZEPTIONAL_MODEL_CACHE.get(tokenizer_cache_key)
        if not tokenizer:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            except Exception:
                tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
            ENZEPTIONAL_MODEL_CACHE.add(tokenizer_cache_key, tokenizer)
        return tokenizer

    def embed(self, samples: List[str]) -> np.ndarray:
        """Embeds a list of samples using either TAPE or Hugging Face models.

        Args:
            samples (List[str]): List of strings to be embedded.

        Returns:
            np.ndarray: The resulting embeddings.
        """
        if self.is_tape_model:
            return self._embed_tape(samples)
        else:
            return self._embed_huggingface(samples)

    def _embed_tape(self, samples: List[str]) -> np.ndarray:
        """mbeds samples using a TAPE model.

        Args:
            samples (List[str]): List of strings to be embedded.

        Returns:
            np.ndarray: The resulting embeddings.
        """
        token_ids: Dict[str, Any] = {"ids": [], "mask": []}
        for sequence in samples:
            encoded_sequence = self.tokenizer.encode(sequence)
            token_ids["ids"].append(encoded_sequence)
            token_ids["mask"].append(np.ones_like(encoded_sequence))

        input_ids = torch.from_numpy(pad_sequences(token_ids["ids"])).to(self.device)
        input_mask = torch.from_numpy(pad_sequences(token_ids["mask"])).to(self.device)

        inputs = {"input_ids": input_ids, "input_mask": input_mask}

        with torch.no_grad():
            sequence_embeddings = (
                self.embedding_model(**inputs)[0].cpu().detach().numpy()
            )

        sequence_lengths = input_mask.sum(1)

        return np.array(
            [
                sequence_embedding[:sequence_length].mean(0)
                for sequence_embedding, sequence_length in zip(
                    sequence_embeddings, sequence_lengths
                )
            ]
        )

    def _embed_huggingface(self, samples: List[str]) -> np.ndarray:
        """Embeds samples using a Hugging Face model.

        Args:
            samples (List[str]): List of strings to be embedded.

        Returns:
            np.ndarray: The resulting embeddings.
        """
        inputs = self.tokenizer(
            samples,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            sequence_embeddings = outputs[0].cpu().detach().numpy()

        sequence_lengths = inputs["attention_mask"].sum(1)

        return np.array(
            [
                sequence_embedding[:sequence_length].mean(0)
                for sequence_embedding, sequence_length in zip(
                    sequence_embeddings, sequence_lengths
                )
            ]
        )

    def unmask(self, sequence: str, top_k: int = 2) -> List[str]:
        """Unmasks a given sequence using the model, retrieving top-k predictions.

        Args:
            sequence (str): The sequence with masked tokens.
            top_k (int, optional): Number of top predictions to retrieve. Defaults to 2.

        Raises:
            NotImplementedError: If TAPE model is used.
            KeyError: If the model used is not supported.

        Returns:
            List[str]: List of top-k predicted sequences.
        """
        if self.is_tape_model:
            logger.error("Unmasking is not supported for TAPE models.")
            raise NotImplementedError("Unmasking is not supported for TAPE models.")

        try:
            return self._unmask_with_model(sequence, top_k)
        except (KeyError, NotImplementedError) as e:
            logger.warning(f"{e} Standard unmasking failed ")
            raise KeyError("Check the unmasking model you want to use")

    def _unmask_with_model(self, sequence: str, top_k: int) -> List[str]:
        """Unmasks a sequence using the model, providing top-k predictions.

        Args:
            sequence (str): The sequence with masked tokens.
            top_k (int): Number of top predictions to retrieve.

        Raises:
            KeyError: If model used do not support unmasking.

        Returns:
            List[str]: List of top-k predicted sequences.
        """
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
        ).to(self.device)
        mask_token_index = torch.where(
            inputs["input_ids"] == self.tokenizer.mask_token_id
        )[1]

        with torch.no_grad():
            outputs = self.unmasking_model(inputs["input_ids"].to(self.device))

        if "logits" in outputs:
            logits = outputs.logits
        else:
            raise KeyError("Logits not available in the model's output.")

        mask_token_logits = logits[0, mask_token_index, :]

        top_tokens: List[Any] = []
        for i in range(len(mask_token_index)):
            top_n_tokens = (
                torch.topk(mask_token_logits, top_k, dim=1).indices[i].tolist()
            )
            top_tokens.append(
                [self.tokenizer.decode([token]) for token in top_n_tokens]
            )

        mask_token_index = mask_token_index.cpu().numpy()
        mutated_sequences = []
        tmp_top_tokens = [tuple(tokens) for tokens in top_tokens]
        if len(set(tmp_top_tokens)) == 1:
            for i in range(top_k):
                temp_sequence = sequence.split(" ")
                for mask_index in mask_token_index:
                    temp_sequence[mask_index - 1] = tmp_top_tokens[0][i]
                mutated_sequences.append("".join(temp_sequence))
        else:
            for combination in list(iter_product(*tmp_top_tokens)):
                temp_sequence = sequence.split(" ")
                for i, mask_index in enumerate(mask_token_index):
                    temp_sequence[mask_index - 1] = combination[i]
                mutated_sequences.append("".join(temp_sequence))

        return mutated_sequences


def mutate_sequence_with_variant(sequence: str, variant: str) -> str:
    """Applies a specified variant mutation to an amino acid sequence.

    Args:
        sequence (str): The original amino acid sequence.
        variant (str): The variant to apply, formatted as a string.

    Returns:
        str: The mutated amino acid sequence.
    """
    mutated_sequence = list(sequence)
    for variant_string in variant.split("/"):
        index = int(variant_string[1:-1]) - 1
        mutated_sequence[index] = variant_string[-1]
    return "".join(mutated_sequence)


def sanitize_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merges overlapping intervals into a single interval.

    Args:
        intervals (List[Tuple[int, int]]): A list of
        start and end points of intervals.

    Returns:
        List[Tuple[int, int]]: A list of merged intervals.
    """
    intervals.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in intervals:
        if not merged or merged[-1][1] < start:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def round_up(number: float) -> int:
    """Rounds up a floating-point number to the nearest integer.

    Args:
        number (float): The number to round up.

    Returns:
        int: The rounded-up integer.
    """
    return math.ceil(number)


def sanitize_intervals_with_padding(
    intervals: List[Tuple[int, int]], pad_value: int, max_value: int
) -> List[Tuple[int, int]]:
    """Pads and sanitizes intervals within a given range.

    Args:
        intervals (List[Tuple[int, int]]): A list of intervals.
        pad_value (int): The value to pad intervals with.
        max_value (int): The maximum value for the range of intervals.

    Returns:
        List[Tuple[int, int]]: A list of padded and sanitized intervals.
    """

    def pad_interval(
        interval: Tuple[int, int], pad: int, max_val: int
    ) -> Tuple[int, int]:
        """Pads an individual interval within the constraints of a maximum value.

        Args:
            interval (Tuple[int, int]): The interval to pad.
            pad (int): The padding value.
            max_val (int): The maximum value for the interval.

        Returns:
            Tuple[int, int]: The padded interval.
        """
        start, end = interval
        interval_length = end - start
        padding_needed = max(0, pad - interval_length) // 2

        padded_start = max(0, start - padding_needed)
        padded_end = min(max_val, end + padding_needed)

        if padded_end > max_val:
            padded_start = max(0, padded_start - (padded_end - max_val))
        return padded_start, padded_end

    padded_intervals = [
        pad_interval(interval, pad_value, max_value) for interval in intervals
    ]
    return sanitize_intervals(padded_intervals)


def reconstruct_sequence_with_mutation_range(
    sequence: str,
    mutated_sequence_range: str,
    intervals: List[Tuple[int, int]],
) -> str:
    """Reconstructs a sequence by inserting a mutated sequence
    range at specific intervals.

    Args:
        sequence (str): The original sequence.
        mutated_sequence_range (str): The range of the sequence to be mutated.
        intervals (List[Tuple[int, int]]): The intervals where
        mutations are applied.

    Returns:
        str: The reconstructed sequence with mutations.
    """
    mutated_sequence = list(sequence)
    range_index = 0
    for start, end in intervals:
        size_fragment = end - start
        mutated_sequence[start:end] = list(
            mutated_sequence_range[range_index : range_index + size_fragment]
        )
        range_index += size_fragment
    return "".join(mutated_sequence)


class SelectionGenerator:
    """
    A generator for selecting top sequences based on their scores.
    """

    def selection(
        self,
        pool_of_sequences: List[Dict[str, Any]],
        k: float = 0.8,
    ) -> List[Any]:
        """Selects a subset of sequences from a pool based on their scores.

        Args:
            pool_of_sequences (List[Dict[str, Any]]): A list of
            dictionaries, each containing a sequence and its score.
            k (float): A fraction representing the proportion
            of top sequences to select. Defaults to 0.8.

        Returns:
            List[Any]: A list of the top k sequences based on scores.
        """
        n_samples_to_select = int(len(pool_of_sequences) * k)
        return list(sorted(pool_of_sequences, key=lambda d: d["score"], reverse=True))[
            :n_samples_to_select
        ]


class CrossoverGenerator:
    """
    A generator for performing crossover operations between sequences.
    """

    def __init__(self, threshold_probability: float = 0.5) -> None:
        """Initializes the CrossoverGenerator with a specified
        threshold probability.

        Args:
            threshold_probability (float, optional): The probability
            threshold used in uniform crossover. Defaults to 0.5.
        """
        self.threshold_probability = threshold_probability

    def sp_crossover(self, a_sequence: str, another_sequence: str) -> Tuple[str, str]:
        """Performs a single point crossover between two sequences.

        Args:
            a_sequence (str): The first sequence for crossover.
            another_sequence (str): The second sequence for crossover.

        Returns:
            Tuple[str, str]: A tuple of two new sequences resulting
            from the crossover.
        """
        random_point = random.randint(1, len(a_sequence) - 2)
        return (
            a_sequence[:random_point] + another_sequence[random_point:],
            another_sequence[:random_point] + a_sequence[random_point:],
        )

    def uniform_crossover(
        self, a_sequence: str, another_sequence: str
    ) -> Tuple[str, str]:
        """Performs a uniform crossover between two sequences.

        Args:
            a_sequence (str): The first sequence for crossover.
            another_sequence (str): The second sequence for crossover.

        Returns:
            Tuple[str, str]: A tuple of two new sequences resulting
            from the crossover.
        """
        return (
            "".join(
                a if random.random() > self.threshold_probability else b
                for a, b in zip(a_sequence, another_sequence)
            ),
            "".join(
                b if random.random() > self.threshold_probability else a
                for a, b in zip(a_sequence, another_sequence)
            ),
        )
