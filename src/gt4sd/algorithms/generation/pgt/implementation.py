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
"""
Implementation details for PGT algorithms.
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
GENERATION_PROMPTS = {
    "title-to-abstract": "{} <|sep|> Given the above title, suggest an abstract <|sep|>",
    "abstract-to-claim": "{} <|sep|> Given the above abstract, suggest a claim <|sep|>",
    "claim-to-abstract": "{} <|sep|> Given the above claim, suggest an abstract <|sep|>",
    "abstract-to-title": "{} <|sep|> Given the above abstract, suggest a title <|sep|>",
}
EDITING_TYPES = ["abstract", "claim"]
COHERENCE_TYPES = ["title-abstract", "abstract-claim", "title-claim"]

STOPPING_PUNCTUATION_REGEX = re.compile(r"(.+(?=\.|!|;)(.|!|;)|(.*))")


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


class Generator:
    """Implementation of a generator."""

    def __init__(
        self,
        resources_path: str,
        model_type: str,
        model_name: str,
        max_length: int,
        top_k: int,
        top_p: float,
        num_return_sequences: int,
        prompt: str = "This is an interesting prompt",
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

    def generate_case(self) -> Union[List[str], List[Tuple[str, ...]]]:
        """Sample text snippets.

        Returns:
            generated text snippets.
        """

        self.prompt = re.sub(" +", " ", self.prompt)
        self.prompt = re.sub(r'\s([?.!,:;·"](?:\s|$))', r"\1", self.prompt)

        encoded_prompt = self.tokenizer.encode(self.prompt, return_tensors="pt")
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
            text = text.replace(self.prompt, "")
            text = text.split("<|endoftext|>")[0]
            text = text.replace("<|pad|>", "")
            text = text.strip()
            text = STOPPING_PUNCTUATION_REGEX.search(  # type:ignore
                text
            ).group()

            generated_sequences.append(text)

        return self.format_output(self.prompt, generated_sequences)

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


class PartGenerator(Generator):
    """Implementation of edit generator."""

    def __init__(
        self,
        resources_path: str,
        input_text: str,
        model_type: str,
        model_name: str,
        task: str,
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
            input_text: input text for generation.
            task: generation task.
            model_type: type of the model.
            model_name: name of the model weights/version.
            max_length: max length of the generated text.
            top_k: number of top-k probability token to keep.
            top_p: only tokens with cumulative probabilities summing up to this value are kept.
            num_return_sequences: number of generated sequences.
            no_repeat_ngram_size: size of n-gram to not appear twice.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """

        if task not in GENERATION_PROMPTS:
            raise ValueError(f"{task} is not a valid option for task.")

        prompt = GENERATION_PROMPTS[task]
        prompt = prompt.format(input_text)

        super().__init__(
            resources_path=resources_path,
            model_type=model_type,
            model_name=model_name,
            prompt=prompt,
            max_length=max_length,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
            device=device,
        )


class EditGenerator(Generator):
    """Implementation of edit generator."""

    def __init__(
        self,
        resources_path: str,
        input_text: str,
        model_type: str,
        model_name: str,
        max_length: int,
        top_k: int,
        top_p: float,
        num_return_sequences: int,
        no_repeat_ngram_size: int = 2,
        device: Optional[Union[torch.device, str]] = None,
        input_type: str = "abstract",
    ):
        """PGT generation algorithm.
        Args:
            resources_path: path to the cache.
            input_text: input text for generation.
            model_type: type of the model.
            model_name: name of the model weights/version.
            max_length: max length of the generated text.
            top_k: number of top-k probability token to keep.
            top_p: only tokens with cumulative probabilities summing up to this value are kept.
            num_return_sequences: number of generated sequences.
            no_repeat_ngram_size: size of n-gram to not appear twice.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
            input_type: part of a patent the input text belongs.
        """

        if input_type not in EDITING_TYPES:
            raise ValueError(
                f"{input_type} is not a valid option for editing input type."
            )

        prompt = f"{input_text} <|sep|> Replace the [MASK] tokens in the above {input_type} <|sep|>"

        super().__init__(
            resources_path=resources_path,
            model_type=model_type,
            model_name=model_name,
            prompt=prompt,
            max_length=max_length,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
            device=device,
        )

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

        number_of_masks = input_text.count("[MASK]") - 1

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

    def __init__(
        self,
        resources_path: str,
        input_a: str,
        input_b: str,
        model_type: str,
        model_name: str,
        max_length: int,
        top_k: int,
        top_p: float,
        num_return_sequences: int,
        no_repeat_ngram_size: int = 2,
        device: Optional[Union[torch.device, str]] = None,
        coherence_type: str = "title-abstract",
    ):
        """PGT generation algorithm.
        Args:
            resources_path: path to the cache.
            input_a: first input for coherence check.
            input_b: second input for coherence check.
            model_type: type of the model.
            model_name: name of the model weights/version.
            max_length: max length of the generated text.
            top_k: number of top-k probability token to keep.
            top_p: only tokens with cumulative probabilities summing up to this value are kept.
            num_return_sequences: number of generated sequences.
            no_repeat_ngram_size: size of n-gram to not appear twice.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
            coherence_type: input types for the check.
        """

        type_a, type_b = self.extract_coherence_types(coherence_type)

        prompt = f"{input_a} <|sep|> {input_b} <|sep|> Do the above {type_a} and {type_b} belong to the same patent? <|sep|>"

        super().__init__(
            resources_path=resources_path,
            model_type=model_type,
            model_name=model_name,
            prompt=prompt,
            max_length=max_length,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
            device=device,
        )

    def extract_coherence_types(self, coherence_type: str) -> Tuple[str, str]:
        """Check the validity and extract coherence types of input text.

        Args:
            coherence_type: Input types of the coherence check.
        Returns:
            tuple containing the type of the input.
        """

        if coherence_type in COHERENCE_TYPES:
            type_a, type_b = coherence_type.split("-")

            return type_a, type_b

        else:
            raise ValueError(f"{coherence_type} is not a valid coherence type.")

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
