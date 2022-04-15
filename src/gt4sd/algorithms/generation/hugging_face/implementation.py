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
Implementation details for HuggingFace generation algorithms.

Parts of the implementation inspired by: https://github.com/huggingface/transformers/blob/v4.2.1/examples/text-generation/run_generation.py.
"""

import logging
import os
from typing import List, Optional, Union

import numpy as np
import torch
from transformers import (
    BasicTokenizer,
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

from ....frameworks.torch import device_claim

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MAXIMUM_LENGTH = int(10000)
# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(seed: int = 42) -> None:
    """Set seed for all random number generators.

    Args:
        seed: seed to set. Defaults to 42.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)  # type:ignore


def prepare_ctrl_input(tokenizer: BasicTokenizer, prompt: str, **kwargs):
    if kwargs.get("temperature", 1.0) > 0.7:
        logger.warning(
            "CTRL typically works better with lower temperatures (and lower k)."
        )

    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)  # type:ignore
    if not any(
        encoded_prompt[0] == x for x in tokenizer.control_codes.values()  # type:ignore
    ):
        logger.warning(
            "not starting generation from a control code so you will not get good results"
        )
    return prompt


def prepare_prefix_input(tokenizer: BasicTokenizer, prompt: str, **kwargs):
    prefix = kwargs["prefix"] if kwargs.get("prefix", "") else PREFIX
    prompt = prefix + prompt
    return prompt


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


MODEL_TYPES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer, None),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer, prepare_ctrl_input),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, None),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer, prepare_prefix_input),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer, prepare_prefix_input),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer, None),
}


class Generator:
    """Implementation of a generator."""

    def __init__(
        self,
        resources_path: str,
        model_type: str,
        model_name: str,
        prompt: str,
        length: int,
        stop_token: str,
        temperature: float,
        repetition_penalty: float,
        k: int,
        p: float,
        prefix: str,
        number_of_sequences: int,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """An HuggingFace generation algorithm.

        Args:
            resources_path: path to the cache.
            model_type: type of the model.
            model_name: name of the model weights/version.
            prompt: prompt for text generation.
            length: length of the generated text.
            stop_token: stop token for text generation.
            temperature: temperature for sampling, the lower the greedier the sampling.
            repetition_penalty: primarily useful for CTRL model, where 1.2 should be used.
            k: number of top-k probability token to keep.
            p: only tokens with cumulative probabilities summing up to this value are kept.
            prefix: text defining context provided prior to the prompt.
            number_of_sequences: number of generated sequences.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.
        """
        self.device = device_claim(device)
        self.resources_path = resources_path
        self.model_type = model_type
        self.model_name = model_name
        self.prompt = prompt
        self.length = length
        self.stop_token = None if stop_token == "" else stop_token
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.k = k
        self.p = p
        self.prefix = prefix
        self.number_of_sequences = number_of_sequences
        self.load_model()

    def load_model(self) -> None:
        """Load a pretrained HuggingFace generation model."""
        try:
            model_class, tokenizer_class, preprocessing_function = MODEL_TYPES[
                self.model_type
            ]
        except KeyError:
            raise KeyError(f"model type: {self.model_type} not supported")
        if (
            os.path.exists(self.resources_path)
            and len(os.listdir(self.resources_path)) > 0
        ):
            model_name_or_path = self.resources_path
        else:
            model_name_or_path = self.model_name
        self.preprocessing_function = preprocessing_function
        self.tokenizer = tokenizer_class.from_pretrained(  # type:ignore
            model_name_or_path
        )
        self.model = model_class.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        # adjusting length
        self.length = adjust_length_to_model(
            self.length, self.model.config.max_position_embeddings
        )

    def sample(self) -> List[str]:
        """Sample text snippets.

        Returns:
            generated text snippets.
        """
        if self.preprocessing_function is not None:
            preprocessed_prompt_text = self.preprocessing_function(
                self.tokenizer,
                self.prompt,
                prefix=self.prefix,
                temperature=self.temperature,
            )

            if self.model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}

            encoded_prompt = self.tokenizer.encode(
                preprocessed_prompt_text,
                add_special_tokens=False,
                return_tensors="pt",
                **tokenizer_kwargs,
            )
        else:
            encoded_prompt = self.tokenizer.encode(
                self.prefix + self.prompt, add_special_tokens=False, return_tensors="pt"
            )

        encoded_prompt = encoded_prompt.to(self.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = self.model.generate(
            input_ids=input_ids,
            max_length=self.length + len(encoded_prompt[0]),
            temperature=self.temperature,
            top_k=self.k,
            top_p=self.p,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
            num_return_sequences=self.number_of_sequences,
        )

        # NOTE: remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences: List[str] = []

        for generated_sequence in output_sequences:
            generated_sequence = generated_sequence.tolist()
            text = self.tokenizer.decode(
                generated_sequence, clean_up_tokenization_spaces=True
            )
            text = text[: text.find(self.stop_token) if self.stop_token else None]
            total_sequence = (
                self.prompt
                + text[
                    len(
                        self.tokenizer.decode(
                            encoded_prompt[0], clean_up_tokenization_spaces=True
                        )
                    ) :
                ]
            )
            generated_sequences.append(total_sequence)

        return generated_sequences
