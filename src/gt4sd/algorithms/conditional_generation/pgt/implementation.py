"""
Implementation details for HuggingFace generation algorithms.
Parts of the implementation inspired by: https://github.com/huggingface/transformers/blob/v4.2.1/examples/text-generation/run_generation.py.
"""

import logging
import os
import re
from typing import List, Optional, Tuple, Union

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from ....frameworks.torch import device_claim

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MAXIMUM_LENGTH = int(10000)


def adjust_length_to_model(length: int, maximum_sequence_length: int):
    """Adjust sequence length.
    Args:
        length: target length.
        maximum_sequence_length: maximum sequence length.
    Returns:
        the adjusted length.
    """
    if length < 0 and maximum_sequence_length > 0:
        logger.warning(
            f"negative length, adjusting to model supported length {maximum_sequence_length}"
        )
        length = maximum_sequence_length
    elif 0 < maximum_sequence_length < length:
        logger.warning(
            f"longer then model supported length, adjusting to {maximum_sequence_length}"
        )
        length = maximum_sequence_length
    elif length < 0:
        logger.warning(f"negative length, adjusting to maximal length {MAXIMUM_LENGTH}")
        length = MAXIMUM_LENGTH
    return length


STOPPING_PUNCTUATION_REGEX = re.compile(r"(.+(?=\.|!|;)(.|!|;)|(.*))")


class Generator:
    """Implementation of a generator."""

    def __init__(
        self,
        resources_path: str,
        model_type: str,
        model_name: str,
        prompt: str,
        max_length: int,
        top_k: int,
        top_p: float,
        num_return_sequences: int,
        no_repeat_ngram_size: int = 2,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """PGT generation algorithm.
        Args:
            resources_path: path to the cache.
            model_type: type of the model.
            model_name: name of the model weights/version.
            prompt: prompt for text generation.
            max_length: max length of the generated text.
            top_k: number of top-k probability token to keep.
            top_p: only tokens with cumulative probabilities summing up to this value are kept.
            num_return_sequences: number of generated sequences.
            no_repeat_ngram_size: size of n-gram to not appear twice.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """
        self.device = device_claim(device)
        self.resources_path = resources_path
        self.model_type = model_type
        self.model_name = model_name
        self.prompt = prompt
        self.length = max_length
        self.k = top_k
        self.p = top_p
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.number_of_sequences = num_return_sequences
        self.load_model()

    def load_model(self) -> None:
        """Load a pretrained PGT model."""

        if (
            os.path.exists(self.resources_path)
            and len(os.listdir(self.resources_path)) > 0
        ):
            model_name_or_path = self.resources_path
        else:
            logger.error(f"{self.resources_path} not found")

        self.tokenizer = GPT2Tokenizer.from_pretrained(  # type:ignore
            model_name_or_path,
            sep_token="<|sep|>",
            mask_token="[MASK]",
            pad_token="<|pad|>",
            additional_special_tokens=["<|mask_sep|>"],
        )
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        # adjusting length
        self.length = adjust_length_to_model(
            self.length, self.model.config.max_position_embeddings
        )

    def generate_case(
        self, input_text: Union[str, Tuple[str]]
    ) -> Union[List[str], List[Tuple[str, ...]]]:
        """Sample text snippets.

        Returns:
            generated text snippets.
        """

        if isinstance(input_text, tuple):
            input_prompt = self.prompt.format(*input_text)
        else:
            input_prompt = self.prompt.format(input_text)

        encoded_prompt = self.tokenizer.encode(input_prompt, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = self.model.generate(
            input_ids=input_ids,
            max_length=self.length,
            top_k=self.k,
            top_p=self.p,
            do_sample=True,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            num_return_sequences=self.number_of_sequences,
        )

        # NOTE: remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences: List[str] = []

        for generated_sequence in output_sequences:
            generated_sequence = generated_sequence.tolist()
            text = self.tokenizer.decode(generated_sequence)
            text = text.replace(input_prompt, "")
            text = text.split("<|endoftext|>")[0]
            text = text.replace("<|pad|>", "")
            text = text.strip()
            text = STOPPING_PUNCTUATION_REGEX.search(  # type:ignore
                text
            ).group()

            generated_sequences.append(text)

        return self.format_output(input_text, generated_sequences)

    def format_output(
        self, input_text: Union[str, Tuple[str]], generated_sequences: List[str]
    ) -> Union[List[str], List[Tuple[str, ...]]]:
        """Format output. In the general case just return the generated sequences.

        Args:
            input_text: generation input.
            generated_sequences: generated sequences.

        Returns:
            formatted generated sequences.
        """
        return generated_sequences


class EditGenerator(Generator):
    """Implementation of edit generator."""

    def format_output(
        self, input_text: Union[str, Tuple[str]], generated_sequences: List[str]
    ) -> Union[List[str], List[Tuple[str, ...]]]:
        """Format output for the patent editing task.

        Args:
           input_text: generation input.
           generated_sequences: generated sequences.

        Returns:
           formatted generated sequences.
        """

        filtered_generated_sequences = []

        number_of_masks = input_text.count("[MASK]")
        print(generated_sequences)
        for seq in generated_sequences:
            generated_masked_tokens = seq.split("<|mask_sep|>")

            filtered_generated_sequence = []
            for i in range(number_of_masks):
                if i < len(generated_masked_tokens):
                    gen_token = generated_masked_tokens[i]
                else:
                    gen_token = "[MASK NOT PREDICTED]"
                filtered_generated_sequence.append(gen_token.strip())

            filtered_generated_sequences.append(tuple(filtered_generated_sequence))

        return filtered_generated_sequences


class CoherenceCheckGenerator(Generator):
    """Implementation of coherence check generator."""

    def format_output(
        self, input_text: Union[str, Tuple[str]], generated_sequences: List[str]
    ) -> Union[List[str], List[Tuple[str, ...]]]:
        """Format output for the patent coherence task.

        Args:
           input_text: generation input.
           generated_sequences: generated sequences.

        Returns:
           formatted generated sequences.
        """

        if (
            "yes" in generated_sequences[0].lower()
            and "no" not in generated_sequences[0].lower()
        ):
            return ["yes"]
        elif (
            "no" in generated_sequences[0].lower()
            and "yes" not in generated_sequences[0].lower()
        ):
            return ["no"]
        else:
            return ["NA"]
