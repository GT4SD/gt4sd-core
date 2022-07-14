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
"""Implementation of Regression Transformer conditional generators."""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from terminator.collators import MaskedTextCollator, PropertyCollator
from terminator.inference import InferenceRT
from terminator.search import SEARCH_FACTORY, Search
from terminator.selfies import decoder, encoder
from terminator.tokenization import InferenceBertTokenizer
from transformers import AutoConfig, AutoModelWithLMHead, XLNetLMHeadModel

from ....domains.materials import Sequence, validate_molecules
from ....frameworks.torch import device_claim, map_tensor_dict

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ConditionalGenerator:
    """Main interface for a regression transformer."""

    # device where the inference is running.
    device: torch.device

    # The task the RT is currently performing. Either 'regression' or 'generation'.
    task: str

    # method to convert logits to predictions. Either GreedySearch or SamplingSearch.
    search: Search

    # ***** Additional attributes for text generation *****
    # data collator for property prediction of self-generated items.
    property_collator: PropertyCollator

    # percentage of tolerated deviation between desired and obtained property.
    tolerance: float = 20

    # number of samples obtained per call.
    batch_size: int = 8

    # a dictionary to specify sampling from a specific seed molecule.
    sampling_wrapper: Dict[str, Any] = {}

    def __init__(
        self, resources_path: str, device: Optional[Union[torch.device, str]] = None
    ) -> None:
        """
        Initialize the generator.

        Args:
            resources_path: directory where to find models and parameters.
            temperature: temperature for the sampling. Defaults to 1.4.
            generated_length: maximum length of the generated molecules.
                Defaults to 100.
            samples_per_protein: number of points sampled per protein.
                It has to be greater than 1. Defaults to 10.
            device: device where the inference is running either as a dedicated class
                or a string. If not provided is inferred.
        """
        # device
        self.device = device_claim(device)

        # Set up the data preparation pipeline
        if not os.path.exists(os.path.join(resources_path, "inference.json")):
            raise OSError(
                f"algorithm_version {resources_path.split('/')[-1]} does not exist."
            )
        self.tokenizer = InferenceBertTokenizer.from_pretrained(
            resources_path, pad_even=False
        )
        self.collator = MaskedTextCollator(self.tokenizer)

        # Set up model: First load the pretrained XLNet model
        xlnet_model, config = self.load_model(resources_path)
        # Initialize the custom RT model
        self.model = InferenceRT(xlnet_model, self.tokenizer, config)

        # Set up inference parameters
        self.load_inference(resources_path)

    def load_model(self, resources_path: str) -> Tuple[XLNetLMHeadModel, Any]:
        """
        Loading a XLNetLMHeadModel which constitutes the base of a RT model.

        Args:
            resources_path: path to the model.

        Returns:
            XLNetLMHeadModel: base of a Regression Transformer model.
            XLNetConfig: configuration of the model.
        """
        config_name = os.path.join(resources_path, "config.json")
        config = AutoConfig.from_pretrained(config_name, mem_len=1024)
        xlnet_model = AutoModelWithLMHead.from_pretrained(
            resources_path, from_tf=False, config=config
        )
        xlnet_model.resize_token_embeddings(len(self.tokenizer))
        xlnet_model.to(self.device)
        xlnet_model.eval()
        logger.info(f"Model restored from {resources_path}")
        return xlnet_model, config

    def load_inference(self, resources_path: str) -> None:
        """
        Load and set up all parameters necessary for inference.

        Args:
            resources_path: path to the model folder.
        """
        try:
            with open(os.path.join(resources_path, "inference.json"), "r") as f:
                data = json.load(f)
            self.properties = data["property_token"]
            if isinstance(self.properties, str):
                self.properties = [self.properties]

            # Optional normalize parameter (for property) defaults to True
            self.do_normalize = data.get("normalize", [True] * len(self.properties))

            self.property_mask_lengths = [
                data["property_mask_length"][p] for p in self.properties
            ]
            self._mins = [
                data.get("property_ranges", {}).get(p, [0, 1])[0]
                for p in self.properties
            ]
            self._maxs = [
                data.get("property_ranges", {}).get(p, [0, 1])[1]
                for p in self.properties
            ]
            self.metadata = data

            # Tolerance defined on the original scale
            self.tolerances = [
                (self._maxs[i] - self._mins[i]) * self.tolerance / 100.0
                for i in range(len(self.properties))
            ]
        except Exception:
            raise ValueError(
                f"Could not restore inference parameters from {resources_path}"
            )

    def denormalize(self, x: float, idx: int, precision: int = 4) -> float:
        """
        Denormalize from the model scale to the original scale.

        Args:
            x: normalized value (often in [0,1]).
            idx: index of the property.
            precision: optional rounding precision. Defaults to 4.

        Returns:
            float: Value in regular scale.
        """

        # If the property was not normalized, return the value
        if not self.do_normalize[idx]:
            return x

        return round(
            x * (self._maxs[idx] - self._mins[idx]) + self._mins[idx], precision
        )

    def normalize(self, x: str, idx: int, precision: int = 3) -> float:
        """
        Normalize from original scale to desired scale.

        Args:
            x: unnormalized input.
            idx: index of the property.
            precision: optional rounding precision.

        Returns:
            float: Normalized value.
        """
        # Error handling
        if not isinstance(x, float) and not self.isfloat(x):
            raise TypeError(f"{x} is not a float and cant safely be casted.")

        x_float = float(x)
        # If this property does not require normalization, return it
        if not self.do_normalize[idx]:
            return x_float
        normed = round(
            (x_float - self._mins[idx]) / (self._maxs[idx] - self._mins[idx]), precision
        )
        return normed

    def validate_input(self, x: str) -> None:
        """
        Sanity checking for formatting of the input string.

        Args:
            x: The string to be validated.

        Raises:
            ValueError: If string was formatted incorrectly.
        """

        if self.tokenizer.expression_separator not in x:
            raise ValueError(
                f"Expression separator {self.tokenizer.expression_separator} not "
                f"found in input {x}."
            )
        if self.tokenizer.mask_token not in x:
            raise ValueError(
                f"Nothing to do, no mask to fill ({self.tokenizer.mask_token}) found "
                f"in input {x}."
            )
        if not all(p in x for p in self.properties):
            raise ValueError(
                f"Not all property tokens ({self.properties}) found in input"
            )

        text_sequence = x.split(self.tokenizer.expression_separator)[-1]
        number_sequence = x[: -len(text_sequence) - 1]

        if (
            self.tokenizer.mask_token in text_sequence
            and self.tokenizer.mask_token in number_sequence
        ):
            raise ValueError(
                f"Do not mask number and text sequence at the same time like in {x}."
            )

        self.validate_input_molecule(text_sequence)
        self.validate_input_numerical(number_sequence)

    def validate_input_molecule(self, sequence: str, smiles: bool = False) -> None:
        """
        Verifies that the non-numerical part of the input is a proper sequence.

        Args:
            sequence: input sequence to be validated.
        """
        raise NotImplementedError

    def validate_input_numerical(self, sequence: str) -> None:
        """
        Verifies that the numeric part of the input sequence is valid.

        Args:
            sequence: input sequence to be validated.
        """
        if self.tokenizer.mask_token in sequence:
            # Task is most likely regression
            if sequence.count(self.tokenizer.mask_token) != sum(
                self.property_mask_lengths
            ):
                raise ValueError(
                    f"To predict {self.properties} you have to mask exactly "
                    f"{self.property_mask_lengths} times respectively."
                )
        else:
            # Task is most likely generation
            tokens = []
            for toks in sequence.split(self.tokenizer.expression_separator):
                tokens.extend(self.tokenizer.property_tokenizer.tokenize(toks))
                tokens.append(self.tokenizer.expression_separator)
            idx = list(
                np.where(np.array(tokens) == self.tokenizer.expression_separator)[0]
            )
            if len(idx) < len(self.properties):
                raise ValueError(
                    "Please append all properties with separator token:"
                    f"{self.tokenizer.expression_separator}."
                )
            if idx[0] != self.property_mask_lengths[0] + 1:
                raise ValueError(
                    f"To generate samples, please describe {self.properties[0]} with "
                    f"{self.property_mask_lengths[0]} tokens."
                )
            if len(self.properties) > 1:
                for i in range(len(self.properties) - 1):
                    if idx[i + 1] - idx[i] - 2 != self.property_mask_lengths[i + 1]:
                        raise ValueError(
                            f"To generate samples, describe {self.properties[i + 1]}"
                            f" with {self.property_mask_lengths[i+1]} samples."
                        )

    def safely_determine_task(self, x: str) -> str:
        """
        Determines whether the passed sequence adheres to regression or generation task.

        Args:
            x: the user-provided input sequence for the model, inluding mask tokens.

        Raises:
            ValueError: if the sequence does not adhere to the formatting rules.

        Returns:
            str: the task, either 'regression' or 'generation'.
        """

        self.validate_input(x)
        if (
            self.tokenizer.mask_token
            in x.split(self.tokenizer.expression_separator)[-1]
        ):

            return "generation"

        return "regression"

    def generate_batch_regression(self, context: Sequence) -> List[Sequence]:
        """
        Predict the property of a sample.

        Args:
            context: a string with a masked property, a separator and an
                entity. E.g. <stab>[MASK][MASK][MASK][MASK]|GSQEVNSGTQTYKNASPEEAERIARKAGATTWTEKGNKWEIRI.

        Returns:
            List[Sequence]: a list of (denormalized) predicted properties for the entity.
                Stored as a Sequence (str), e.g., '<qed>0.727'.
        """
        logger.info(f"Starting prediction for sequence {context}")

        # Prepare the batch
        inputs = self.collator([self.tokenizer(context)])
        input_ids = inputs["input_ids"].cpu()

        # Forward pass
        outputs = self.model(map_tensor_dict(inputs, self.device))

        # Obtain the singular predictions
        prediction = self.search(outputs["logits"].detach())

        return self.compile_regression_result(input_ids, prediction)

    def compile_regression_result(
        self, input_ids: torch.Tensor, prediction: torch.Tensor
    ) -> List[Sequence]:
        """
        Postprocesses the prediction from the property task to obtain a float.

        Args:
            input_ids: 2D Tensor of shape (batch_size, sequence_length).
            prediction: 2D Tensor of shape (batch_size, sequence_length).

        Returns:
            List[Sequence]: list of property sequences (one per sample). Can contain
                multiple properties but have to be hashable, therefore we use Sequences,
                e.g., '<qed>0.727' or '<logp>6.65<scscore>3.82'.
        """
        properties = []
        for inp, pred in zip(input_ids, prediction):
            in_tokens = self.tokenizer.decode(
                inp, clean_up_tokenization_spaces=False
            ).split(" ")
            out_tokens = self.tokenizer.decode(
                pred, clean_up_tokenization_spaces=False
            ).split(" ")
            joined = self.tokenizer.get_sample_prediction(out_tokens, in_tokens)
            _, gen_prop = self.tokenizer.aggregate_tokens(joined, label_mode=False)
            properties.append(
                "".join(
                    [
                        f"<{k}>{self.denormalize(v, pidx)}"
                        for pidx, (k, v) in enumerate(gen_prop.items())
                    ]
                )
            )
        return properties

    def generate_batch_generation(self, sequence: Sequence) -> Tuple:
        """
        Conditionally generate sequences given a continuous property value and a fixed
        sequence. This function first conditionally generates the novel sequences and
        then predicts their properties using the RT again. Only if the predicted
        property is within the tolerance range, the novel sequence is returned.

        Args:
            sequence: the input sequence with masked tokens on the text.

        Returns:
            Tuple[Tuple[str, float]]: a tuple of tuples, each containing the generated
                sequence alongside its predicted property value.
        """

        # If we are in sampling mode, the sampled sequence changes per iteration.
        if self.sampling_wrapper != {}:
            sequence = self.sample_sequence(sequence)

        # The property score has to be in the range understood by the model
        sequence = self.normalize_sequence(sequence)

        logger.warning(f"Starting prediction for sequence {sequence}")

        # Prepare the batch
        tokens = self.tokenizer(sequence)
        inputs = self.collator([tokens] * self.batch_size)
        input_ids = inputs["input_ids"].clone()

        # Forward pass
        outputs = self.model(map_tensor_dict(inputs, self.device))
        # Obtain model predictions via the search method
        predictions = self.search(outputs["logits"].detach()).squeeze().cpu()
        # Combine predictions with the static part to obtain the full sequences
        generations = input_ids.cpu()
        generations[generations == self.tokenizer.mask_token_id] = predictions[
            generations == self.tokenizer.mask_token_id
        ]

        # Second part: Predict the properties of the just generated sequence
        _input = self.property_collator.mask_tokens(generations)
        prediction_input = {
            "input_ids": _input[0],
            "perm_mask": _input[1],
            "target_mapping": _input[2],
            "attention_mask": self.property_collator.attention_mask(generations),
        }

        # Pass through model
        property_outputs = self.model(map_tensor_dict(prediction_input, self.device))
        # It's a design choice to go with greedy predictions here
        predictions = torch.argmax(property_outputs["logits"].detach(), dim=-1)
        # Obtain floating predictions
        properties = self.compile_regression_result(generations, predictions)
        # Obtain the sequences (AAS or SELFIES)
        sequences = [
            self.tokenizer.to_readable(
                "".join(
                    self.tokenizer.decode(seq, skip_special_tokens=True).split(" ")
                ).split(self.tokenizer.expression_separator)[-1]
            )
            for seq in generations
        ]
        # Filter out all sequences that do not satisfy property constraints within
        # tolerance range.
        logger.debug(f"Sequences {sequences}, properties: {properties}")
        successes: Tuple = tuple(
            filter(
                lambda x: all(
                    [
                        abs(
                            float(x[1].split(p)[-1].split("<")[0])
                            - self.target_values[i]
                        )
                        < self.tolerances[i]
                        for i, p in enumerate(self.properties)
                    ]
                ),
                zip(sequences, properties),
            )
        )  # type: ignore
        logger.info(f"Successes: {successes}")
        return successes

    def normalize_sequence(self, context: Sequence) -> Sequence:
        """
        Take a sequence with unnormalized property score(s) and convert it to a
        sequence with a normalized score.

        Args:
            context: sequence with unnormalized property.

        Returns:
            sequence with normalized property.
        """

        # Tokenize sequence and extract positions of separator tokens
        tokens = self.tokenizer.tokenize(context)
        final_tokens = ""
        sep_idxs = [
            i for i, t in enumerate(tokens) if t == self.tokenizer.expression_separator
        ]

        # Declard as class variable since used by other methods
        self.target_values = []

        # Loop over properties and normalize them
        for idx, prop in enumerate(self.properties):
            numerical_tokens = tokens[tokens.index(prop) + 1 : sep_idxs[idx]]
            final_tokens += prop

            unnorm = self.tokenizer.floating_tokens_to_float(numerical_tokens)
            self.target_values.append(unnorm)
            target = self.normalize(unnorm, idx)
            norm = str(target + 1e-10)[: self.property_mask_lengths[idx]]
            final_tokens += norm + self.tokenizer.expression_separator

        # Append rest of sequence
        final_tokens += "".join(tokens[sep_idxs[-1] + 1 :])
        return final_tokens

    @staticmethod
    def isfloat(sequence: str) -> bool:
        """Safely determine whether a string can be converted to a float

        Args:
            sequence: A string

        Returns:
            Whether it can be converted to a float
        """
        try:
            float(sequence)
            return True
        except ValueError:
            return False

    def validate_numerical(
        self, sequences: List[Any]
    ) -> Tuple[List[Sequence], List[int]]:
        """
        Validate whether a list of sequences contains only numerical values.

        Args:
            sequences: a list of hopefully only numerical values.

        Returns:
            A tuple of two lists for the validated Sequences and their respective
            indices.
        """
        items = []
        idxs = []
        for idx, item in enumerate(sequences):
            if (
                isinstance(item, str)
                and item.startswith("<")
                and self.isfloat(item.split(">")[-1])
            ):
                items.append(item)
                idxs.append(idx)
        return items, idxs

    def validate_sampling_wrapper(
        self,
        context: str,
        property_goal: Dict[str, Any] = {},
        fraction_to_mask: float = 0.2,
        tokens_to_mask: List = [],
    ) -> None:
        """
        Validating whether the wrapper can be used for conditional generation of samples.

        Args:
            context: A string that is used as a seed. Has to be a SELFIES
                (RegressionTransformerMolecules) or AAS (RegressionTransformerProteins).
            property_goal: Specifies the property conditions for the targeted generation.
               The keys are the properties and have to be aligned with the
               algorithm_version. For example, for the solubility model use:
               {'<esol>': 1.23}
               or for the logp_and_synthesizability model use:
               {'<logp>': 1.23, '<synthesizability>': 2.34}
                Defaults to {}, but it has to be specified.
            fraction_to_mask: The fraction of tokens that can be changed. Defaults to 0.2.
            tokens_to_mask: A list of atoms (or amino acids) that can be considered for masking.
                Defaults to [] meaning that all tokens can be masked. E.g., use ['F'] to
                only mask fluorine atoms
        """
        self.validate_input_molecule(context, smiles=True)

        self.seed_molecule = context

        if property_goal == {}:
            raise ValueError("Please specify the target properties with a dictionary.")
        if not all([k in property_goal.keys() for k in self.properties]):
            raise ValueError(
                f"Please provide property goals for all properties: {self.properties}."
            )
        self.property_goal = property_goal

        if not isinstance(fraction_to_mask, float):
            raise TypeError(
                f"The fraction_to_mask {fraction_to_mask} has to be a float."
            )
        if fraction_to_mask < 0 or fraction_to_mask > 1:
            raise ValueError(
                f"The fraction_to_mask {fraction_to_mask} has to be between 0 and 1."
            )
        self.fraction_to_mask = fraction_to_mask

        if not isinstance(tokens_to_mask, list):
            raise TypeError(f"The tokens_to_mask {tokens_to_mask} has to be a list.")
        self.maskable_tokens = self.get_maskable_tokens(tokens_to_mask)

        logger.info(
            f"Will start sampling molecules similar to {context} with goal: "
            f"{property_goal} and masking {fraction_to_mask} of the tokens."
        )

    def sample_sequence(self, seq: str) -> str:
        """
        Assembling a RT-sequence from a seed SMILES/AAS sequence.

        Args:
            seq: A SMILES/AAS string used as seed.

        Returns:
            A RT-sequence that uses SELFIES/AAS and incorporates the properties, e.g.:
            `<logp>1.234|<synthesizability>1.234|[C][C][O]`
        """

        if self.sampling_wrapper == {}:
            return seq

        # Convert SMILES/AAS to list of SELFIES/AAS tokens
        language_seq = self.language_encoding(seq)
        tokens = self.tokenizer.text_tokenizer.tokenize(language_seq)

        # Determine which tokens can be masked
        if self.maskable_tokens == []:
            # All tokens can be considered for masking
            maskable_tokens = set(tokens)
            num_to_mask = round(len(tokens) * self.fraction_to_mask)
        else:
            # Mask only tokens that are specified by the user
            maskable_tokens = set(self.maskable_tokens)
            num_to_mask = round(
                sum([tokens.count(t) for t in maskable_tokens]) * self.fraction_to_mask
            )

        maskable_idxs = [i for i, t in enumerate(tokens) if t in maskable_tokens]

        # Mask the tokens
        mask_idxs = np.random.choice(maskable_idxs, num_to_mask, replace=False)
        sequence_tokens = [
            t if i not in mask_idxs else self.tokenizer.mask_token
            for i, t in enumerate(tokens)
        ]

        # Extract the property tokens
        if not set(self.properties) == set(self.property_goal.keys()):
            raise ValueError(
                f"Please provide property goals exactly for: {self.properties}."
            )

        # Define property tokens (this has unnormalized property values)
        prop_tokens = ""
        for i, p in enumerate(self.properties):
            numerical = str(self.property_goal[p] + 1e-10)[
                : self.property_mask_lengths[i]
            ]
            prop_tokens += p + numerical + self.tokenizer.expression_separator

        # Return new sequence
        sequence = prop_tokens + "".join(sequence_tokens)
        return sequence

    def get_maskable_tokens(self, tokens_to_mask: List[str]):
        raise NotImplementedError

    def language_encoding(self, seq: str):
        raise NotImplementedError


class ChemicalLanguageRT(ConditionalGenerator):
    """
    Hybrid regression and conditional molecular generation model as implemented in
    https://arxiv.org/abs/2202.01338.

    Attributes:
        resources_path: path to the model.
        context: user-specified input text for the model.
        search: search key to instantiate a search via terminator.search.SEARCH_FACTORY.
        temperature: the temperature parameter in case of a `sample` search.
        batch_size: the batch size for the model, applicable only to generative task.
        tolerance: the tolerance for the property of the generated molecules.
    """

    def __init__(
        self,
        resources_path: str,
        context: str,
        search: str = "sample",
        temperature: float = 1.4,
        batch_size: int = 8,
        tolerance: float = 20.0,
        sampling_wrapper: Dict[str, Any] = {},
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """
        Initialize the molecule generator.

        Args:
            resources_path: directory where to find models and parameters.
            context: user-specified input text for the model.
            search: search key to instantiate a search, defaults to `sample`.
            temperature: temperature for the sampling. Defaults to 1.4.
            batch_size: number of points sampled per call. Defaults to 8.
            tolerance: the tolerance for the property of the generated molecules.
                Given in percent. Defaults to 20.
            sampling_wrapper: A high-level entry point that allows specifying a seed
                SMILES alongside some target conditions.
                NOTE: If this is used, the `target` needs to be a single SMILES string.
                Example: {
                    'fraction_to_mask': 0.5,
                    'tokens_to_mask': [],
                    'property_goal': {'<qed>': 0.85}
                }
            device: device where the inference s running either as a dedicated class
                or a string. If not provided is inferred.
        """
        super().__init__(device=device, resources_path=resources_path)

        if sampling_wrapper == {}:
            # Validate input and determine task
            self.task = self.safely_determine_task(context)
        else:
            self.validate_sampling_wrapper(context=context, **sampling_wrapper)
            self.task = "generation"
        self.sampling_wrapper = sampling_wrapper

        # Console outputs for usage of search methods
        if search == "sample" and self.task == "regression":
            logger.warning("For regression task, greedy search is recommended")
        elif search == "greedy" and self.task == "generation":
            logger.warning("For generation task, sample search is recommended")

        if search not in SEARCH_FACTORY.keys():
            raise KeyError(f"Pick a search of {SEARCH_FACTORY.keys()} not: {search}.")
        self.search = SEARCH_FACTORY[search](temperature=temperature)

        # Based on task, set the correct generate_batch method
        if self.task == "regression":
            self.generate_batch = self.generate_batch_regression
        elif self.task == "generation":
            self.generate_batch = self.generate_batch_generation  # type: ignore
            self.batch_size = batch_size
            self.property_collator = PropertyCollator(
                tokenizer=self.tokenizer,
                property_tokens=self.properties,
                num_tokens_to_mask=[-1] * len(self.properties),
                ignore_errors=False,
            )

        else:
            raise ValueError(f"Unknown task: {self.task}")

        self.small_mol = True

    def validate_input_molecule(self, sequence: str, smiles: bool = False) -> None:
        """
        Verifies that the non-numerical part of the input sequence is a molecule.

        Args:
            sequence: input sequence to be validated.
            smiles: whether the input is validated to be a SELFIES (default) or SMILES.
        """
        if smiles:
            _, idxs = validate_molecules([sequence])
            if len(idxs) != 1:
                raise ValueError(
                    f"The context {sequence} is not a valid SMILES string."
                )
        else:
            # Fractional molecules based on non-masked parts of the SELFIES sequence
            smis = list(map(decoder, sequence.split(self.tokenizer.mask_token)))
            if -1 in smis:
                raise ValueError(f"Invalid sequence: {sequence}")

    def validate_output(self, sequences: List[Any]) -> Tuple[List[Any], List[int]]:
        """
        Validate the output of the RT model.

        Args:
            sequences: list of sequences to be validated.

        Returns:
            A tuple of validated items:
                - the validate items, a list of either:
                    - Chem.rdchem.Mol (generation task)
                    - Sequence denoting the predicted properties (regression task)
                - list of valid indexes.
        """

        if self.task == "regression":
            return self.validate_numerical(sequences)
        else:
            # Convert SELFIES to SMILES
            smiles_list = list(
                filter(lambda x: x is not None, list(zip(*sequences))[0])
            )
            if smiles_list == []:
                return ([None], [-1])
            return validate_molecules(smiles_list=smiles_list)  # type: ignore

    def get_maskable_tokens(self, tokens_to_mask: List[str]) -> List[str]:
        """
        Convert a user-defined list of maskable tokens into a RT model-friendly format.

        Args:
            tokens_to_mask: List of atoms specified in SMILES notation.

        Returns:
            List of atoms in SELFIES notation.
        """
        # To reflect double bonds
        tokens_to_mask.extend([f"={t}" for t in tokens_to_mask])
        return [encoder(a) for a in tokens_to_mask]  # type: ignore

    def language_encoding(self, seq: str) -> str:
        selfie = encoder(seq)
        if not isinstance(selfie, str):
            raise TypeError(f"{seq} is not a valid SELFIES sequence.")
        return selfie


class ProteinLanguageRT(ConditionalGenerator):
    """
    Hybrid regression and conditional protein generation model as implemented in
    https://arxiv.org/abs/2202.01338. It generates peptides with a desired stability
    score or predicts the stability score of a given molecule.
    For details on the stability task see: https://doi.org/10.1126/science.aan0693

    Attributes:
        resources_path: path to the model.
        context: user-specified input text for the model.
        search: search key to instantiate a search via terminator.search.SEARCH_FACTORY.
        temperature: the temperature parameter in case of a `sample` search.
        batch_size: the batch size for the model, applicable only to generative task.
        tolerance: the tolerance for the property of the generated molecules.
    """

    def __init__(
        self,
        resources_path: str,
        context: str,
        search: str = "sample",
        temperature: float = 1.4,
        batch_size: int = 32,
        tolerance: float = 20.0,
        sampling_wrapper: Dict[str, Any] = {},
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """
        Initialize the protein generator.

        Args:
            resources_path: directory where to find models and parameters.
            search: search key to instantiate a search, defaults to `sample`.
            temperature: temperature for the sampling. Defaults to 1.4.
            batch_size: number of points sampled per call. Defaults to 8.
            tolerance: the tolerance for the property of the generated molecules.
                Given in percent. Defaults to 20.
            sampling_wrapper: A high-level entry point that allows specifying a seed
                SMILES alongside some target conditions.
                NOTE: If this is used, the `target` needs to be a single SMILES string.
                Example: {
                    'fraction_to_mask': 0.5,
                    'tokens_to_mask': [],
                    'property_goal': {'<stab>': 0.85}
                }
            device: device where the inference s running either as a dedicated class
                or a string. If not provided is inferred.
        """
        super().__init__(device=device, resources_path=resources_path)

        if sampling_wrapper == {}:
            # Validate input and determine task
            self.task = self.safely_determine_task(context)
        else:
            self.task = "generation"
            self.validate_sampling_wrapper(context=context, **sampling_wrapper)
        self.sampling_wrapper = sampling_wrapper

        # Console outputs for usage of search methods
        if search == "sample" and self.task == "regression":
            logger.warning("For regression task, greedy search is recommended")
        elif search == "greedy" and self.task == "generation":
            logger.warning("For generation task, sample search is recommended")
        if search not in SEARCH_FACTORY.keys():
            raise KeyError(f"Pick a search of {SEARCH_FACTORY.keys()} not: {search}.")
        self.search = SEARCH_FACTORY[search](temperature=temperature)

        # Based on task, set the correct generate_batch method
        if self.task == "regression":
            self.generate_batch = self.generate_batch_regression
        elif self.task == "generation":
            self.generate_batch = self.generate_batch_generation  # type: ignore
            self.batch_size = batch_size
            self.property_collator = PropertyCollator(
                tokenizer=self.tokenizer,
                property_tokens=self.properties,
                num_tokens_to_mask=[-1] * len(self.properties),
                ignore_errors=False,
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")

        self.small_mol = False

    def validate_input_molecule(self, sequence: str, smiles: bool = False) -> None:
        """
        Verifies that the non-numerical part of the input sequence is a valid AAS.

        Args:
            sequence: input sequence to be validated.
            smiles: boolean argument that is ignored but needed for sibling class.
        """
        if sequence != sequence.upper():
            raise ValueError(
                f"Sequence {sequence} does not follow IUPAC convention for AAS"
            )

    def validate_output(self, sequences: List[Any]) -> Tuple[List[Any], List[int]]:
        """
        Validate the output of the RT model.

        Args:
            sequences: list of sequences to be validated.

        Returns:
            A tuple of validated items:
                - List of validated items, either:
                    - Amino acid sequences (generation task)
                    - Sequence denoting the predicted properties (regression task)
                - a list of valid indexes.
        """

        if self.task == "regression":
            return self.validate_numerical(sequences)
        else:
            items = [
                item
                if (
                    (
                        item[0] == item[0].upper()
                        and self.tokenizer.mask_token not in item[0]
                        and not any([s.isdigit() for s in item[0]])
                    )
                    and self.validate_numerical(item[1])
                )
                else None
                for item in sequences
            ]
            idxs = [i for i, item in enumerate(sequences) if item in items]
            return items, idxs

    def get_maskable_tokens(self, tokens_to_mask: List[str]) -> List[str]:
        """
        Convert a user-defined list of maskable tokens (amino acids) into a RT
            model-friendly format. Nothing has to be done for proteins, but the function
            is more complex for sister-classes.

        Args:
            tokens_to_mask: List of amino acids specified in IUPAC convention.

        Returns:
            The same.
        """
        return tokens_to_mask

    def language_encoding(self, seq: str) -> str:
        return seq
