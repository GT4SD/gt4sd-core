"""Implementation of Regression Transformer conditional generators."""
import json
import logging
import os
from typing import Any, List, Optional, Tuple, Union

import torch
from terminator.collators import MaskedTextCollator, PropertyCollator
from terminator.inference import InferenceRT
from terminator.search import SEARCH_FACTORY, Search
from terminator.selfies import decoder
from terminator.tokenization import InferenceBertTokenizer
from transformers import AutoConfig, AutoModelWithLMHead, XLNetLMHeadModel

from ....domains.materials import Property, Sequence, validate_molecules
from ....frameworks.torch import device_claim

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ConditionalGenerator:
    """Main interface for a regression transformer."""

    # device where the inference is running.
    device: torch.device

    # The task the RT is currently performing. Either 'regression' or 'generation'
    task: str

    # method to convert logits to predictions. Either GreedySearch or SamplingSearch
    search: Search

    # ***** Additional attributes for text generation *****
    # data collator for property prediction of self-generated items
    property_collator: PropertyCollator

    # percentage of tolerated deviation between desired and obtained property
    tolerance: float = 20

    # number of samples obtained per call
    batch_size: int = 8

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
            self.property = data["property_token"]
            self.property_mask_length = data["property_mask_length"][self.property]
            self.min_ = data.get("property_ranges", {}).get(self.property, [0, 1])[0]
            self.max_ = data.get("property_ranges", {}).get(self.property, [0, 1])[1]
            self.metadata = data
        except Exception:
            raise ValueError(
                f"Could not restore inference parameters from {resources_path}"
            )

    def denormalize(self, x: float, precision: int = 4) -> float:
        """
        Denormalize from [0,1] scale to original scale.

        Args:
            x: normalized value.
            precision: optional rounding precision. Defaults to 4.

        Returns:
            float: Value in regular scale.
        """
        return round(x * (self.max_ - self.min_) + self.min_, precision)

    def normalize(self, x: float, precision: int = 3) -> float:
        """
        Normalize from original scale to [0,1] scale.

        Args:
            x: unnormalized input.
            precision: optional rounding precision.

        Returns:
            float: Normalized value.
        """
        return round((x - self.min_) / (self.max_ - self.min_), precision)

    def validate_input(self, x: str) -> None:

        if self.tokenizer.expression_separator not in x:
            raise ValueError(
                f"Expression separator {self.tokenizer.expression_separator} not "
                f"found in input {x}."
            )
        if self.tokenizer.mask_token not in x:
            raise ValueError(
                f"Nothing to do, no mask to fill ({self.tokenizer.mask_token}) found"
                f"in input {x}."
            )
        if self.property not in x:
            raise ValueError(f"No property token ({self.property}) found in input")

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

    def validate_input_molecule(self, sequence: str) -> None:
        """
        Verifies that the non-numerical part of the input is a proper sequence.

        Args:
            sequence: input sequence to be validated.
        """
        raise NotImplementedError

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

        if x.count(self.tokenizer.mask_token) != self.property_mask_length:
            raise ValueError(
                f"To predict {self.property} you have to mask {self.property_mask_length} times"
            )

        return "regression"

    def generate_batch_regression(self, context: Sequence) -> List[Property]:
        """
        Predict the property of a sample.

        Args:
            context: a string with a masked property, a separator and an
                entity. E.g. <stab>[MASK][MASK][MASK][MASK]|GSQEVNSGTQTYKNASPEEAERIARKAGATTWTEKGNKWEIRI.

        Returns:
            List[Property]: a list of (denormalized) predicted properties for the entity.
        """
        logger.info(f"Starting prediction for sequence {context}")

        # Prepare the batch
        inputs = self.collator([self.tokenizer(context)])
        input_ids = inputs["input_ids"].cpu()

        # Forward pass
        outputs = self.model(inputs)

        # Obtain the singular predictions
        prediction = self.search(outputs["logits"].detach())

        return self.compile_regression_result(input_ids, prediction)

    def compile_regression_result(
        self, input_ids: torch.Tensor, prediction: torch.Tensor
    ) -> List[Property]:
        """
        Postprocesses the prediction from the property task to obtain a float.

        Args:
            input_ids: 2D Tensor of shape (batch_size, sequence_length).
            prediction: 2D Tensor of shape (batch_size, sequence_length).

        Returns:
            List[Property]: list of property values.
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
            properties.append(self.denormalize(gen_prop[self.property[1:-1]]))
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

        # The property score has to be in the range [0, 1]
        sequence = self.normalize_sequence(sequence)

        logger.warning(f"Starting prediction for sequence {sequence}")

        # Prepare the batch
        tokens = self.tokenizer(sequence)
        inputs = self.collator([tokens] * self.batch_size)
        input_ids = inputs["input_ids"].clone()
        # Forward pass
        outputs = self.model(inputs)
        # Obtain model predictions via the search method
        predictions = self.search(outputs["logits"].detach()).squeeze()
        # Combine predictions with the static part to obtain the full sequences
        generations = input_ids
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
        property_outputs = self.model(prediction_input)
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
        successes: Tuple = tuple(
            filter(
                lambda x: abs(self.normalize(x[1]) - self.target_value)
                < self.tolerance,
                zip(sequences, properties),
            )
        )  # type: ignore
        logger.info(f"Successes: {successes}")
        return successes

    def normalize_sequence(self, context: Sequence) -> Sequence:
        """
        Take a sequence with a unnormalized property score and convert it to a
        sequence with a normalized score.

        Args:
            context: sequence with unnormalized property.

        Returns:
            Sequence: sequence with normalized property.
        """
        tokens = self.tokenizer.tokenize(context)
        numerical_tokens = tokens[
            tokens.index(self.property)
            + 1 : tokens.index(self.tokenizer.expression_separator)
        ]

        unnorm = self.tokenizer.floating_tokens_to_float(numerical_tokens)
        # Declard as class variable since used by other methods
        self.target_value = self.normalize(unnorm)
        norm = str(self.target_value)[: self.property_mask_length]

        tokens = (
            "".join(tokens[: tokens.index(self.property) + 1])
            + norm
            + "".join(tokens[tokens.index(self.tokenizer.expression_separator) :])
        )
        return "".join(tokens)

    @staticmethod
    def validate_numerical(sequences: List[Any]):
        """
        Validate whether a list of sequences contains only numerical values.

        Args:
            sequences: a list of hopefully only numerical values.

        Returns:
             List[Any]: a tuple containing of the validated floats and valid indexes.
        """

        items = [item if isinstance(item, float) else None for item in sequences]
        idxs = [i for i, item in enumerate(sequences) if isinstance(item, float)]
        return items, idxs


class ChemicalLanguageRT(ConditionalGenerator):
    """
    Hybrid regression and conditional molecular generation model as implemented in
    https://arxiv.org/abs/2202.01338. It generates molecules with a desired solubility
    (ESOL) score or predicts the ESOL of a given molecule.
    For details on the ESOL task see: https://pubs.acs.org/doi/10.1021/ci034243x

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
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """
        Initialize the molecule generator.

        Args:
            resources_path: directory where to find models and parameters.
            search: search key to instantiate a search, defaults to `sample`.
            temperature: temperature for the sampling. Defaults to 1.4.
            batch_size: number of points sampled per call. Defaults to 8.
            tolerance: the tolerance for the property of the generated molecules.
                Given in percent. Defaults to 20.
            device: device where the inference s running either as a dedicated class
                or a string. If not provided is inferred.
        """
        super().__init__(device=device, resources_path=resources_path)

        # Validate input and determine task
        self.task = self.safely_determine_task(context)

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
                property_tokens=[self.property],
                num_tokens_to_mask=[-1],
                ignore_errors=False,
            )
            # Tolerance on [0,1] scale
            self.tolerance = tolerance / 100.0
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def validate_input_molecule(self, sequence: str) -> None:
        """
        Verifies that the non-numerical part of the input sequence is a SELFIES.

        Args:
            sequence: input sequence to be validated.
        """
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
            A tuple of validated items (Chem.rdchem.Mol in the case of a generation task
                floating values otherwise) and a list of valid indexes.
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
            device: device where the inference s running either as a dedicated class
                or a string. If not provided is inferred.
        """
        super().__init__(device=device, resources_path=resources_path)

        # Validate input and determine task
        self.task = self.safely_determine_task(context)

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
                property_tokens=[self.property],
                num_tokens_to_mask=[-1],
                ignore_errors=False,
            )
            # Tolerance on [0,1] scale
            self.tolerance = tolerance / 100.0
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def validate_input_molecule(self, sequence: str) -> None:
        """
        Verifies that the non-numerical part of the input sequence is a valid AAS.

        Args:
            sequence: input sequence to be validated.
        """
        if sequence != sequence.upper():
            raise ValueError(
                f"Sequence {sequence} does not follow IUPAC convention for AAS"
            )

    def validate_output(self, sequences: Any) -> Tuple[List[Any], List[int]]:
        """
        Validate the output of the RT model.

        Args:
            sequences: list of sequences to be validated.

        Returns:
            A tuple of validated items and a list of valid indexes.
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
                    and isinstance(item[1], float)
                )
                else None
                for item in sequences
            ]
            idxs = [i for i, item in enumerate(sequences) if item in items]
            return items, idxs
