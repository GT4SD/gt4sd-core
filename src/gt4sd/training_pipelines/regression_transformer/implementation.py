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
"""Regression Transformer training implementation."""
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from terminator.collators import TRAIN_COLLATORS
from terminator.tokenization import ExpressionBertTokenizer
from terminator.trainer import CustomTrainer, get_trainer_dict
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    DataCollatorForPermutationLanguageModeling,
    LineByLineTextDataset,
    set_seed,
)

from ..core import TrainingPipeline, TrainingPipelineArguments
from .utils import (
    get_hf_training_arg_object,
    get_train_config_dict,
    prepare_datasets_from_files,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())  # type: ignore
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class RegressionTransformerTrainingPipeline(TrainingPipeline):
    """RegressionTransformer training pipeline."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
    ) -> None:
        """Generic training function for training a Regression Transformer (RT) model.
           For details see:
            Born, J., & Manica, M. (2022). Regression Transformer: Concurrent Conditional
            Generation and Regression by Blending Numerical and Textual Tokens.
            `ICLR Workshop on Machine Learning for Drug Discovery`.

        Args:
            training_args: training arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.
        """
        try:

            params = {**training_args, **dataset_args, **model_args}
            # Setup logging
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
            )
            if not params["do_train"]:
                logger.info("Nothing to do.")
                return

            training_name = params.get("training_name", "rt_training")

            logger.info(f"Model with name {training_name} starts.")
            self.setup_model(params)

            # Register training_dataset and eval_dataset
            self.train_dataset, self.test_dataset = self.setup_dataset(**params)
            logger.info(
                f"# samples: {len(self.train_dataset)}, {len(self.test_dataset)}."
            )

            # Model logging
            num_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            typ = type(self.model)
            logger.info(f"# parameters: {num_params}. Model: {typ}")
            if typ != "xlnet":
                logger.warning(f"Full functionality only with XLNet; not {typ}")

            # Setup training configuration
            self.model.resize_token_embeddings(len(self.tokenizer))
            if training_args["alternate_steps"] <= 0:
                # No alternation of training objectives means: Vanilla PLM training
                data_collator = DataCollatorForPermutationLanguageModeling(
                    tokenizer=self.tokenizer,
                    plm_probability=training_args["plm_probability"],
                    max_span_length=training_args["max_span_length"],
                )
                alternating_collator = None
            else:
                data_collator = TRAIN_COLLATORS["property"](
                    tokenizer=self.tokenizer,
                    property_tokens=self.properties,
                    num_tokens_to_mask=None,
                    mask_token_order=None,
                )
                alternating_collator = TRAIN_COLLATORS[training_args["cg_collator"]](
                    tokenizer=self.tokenizer,
                    property_tokens=self.properties,
                    plm_probability=training_args["plm_probability"],
                    max_span_length=training_args["max_span_length"],
                    do_sample=False,
                    entity_separator_token=training_args["entity_separator_token"],
                    mask_entity_separator=training_args["mask_entity_separator"],
                    entity_to_mask=training_args["entity_to_mask"],
                )

            # Initialize our Trainer
            train_config = get_train_config_dict(training_args, set(self.properties))
            os.makedirs(params["output_dir"], exist_ok=True)
            with open(
                os.path.join(params["output_dir"], "training_config.json"), "w"
            ) as f:
                json.dump(train_config, f, indent="\t")

            # Create the inference.json
            inference_dict = {
                "property_token": self.properties,
                "example": self.example_sample,
                "property_ranges": {
                    p.name: [p.minimum, p.maximum]
                    for p in self.property_objects.values()
                },
                "normalize": [False] * len(self.properties),  # True not supported atm
                "max_span_length": training_args["max_span_length"],
                "property_mask_length": {
                    p.name: p.mask_length for p in self.property_objects.values()
                },
            }
            with open(os.path.join(params["output_dir"], "inference.json"), "w") as f:
                json.dump(inference_dict, f, indent="\t")

            custom_trainer_params = get_trainer_dict(self.model_params)
            hf_train_object = get_hf_training_arg_object(training_args)

            trainer = CustomTrainer(
                model=self.model,
                args=hf_train_object,
                data_collator=data_collator,
                train_dataset=self.train_dataset,
                eval_dataset=self.test_dataset,
                tokenizer=self.tokenizer,
                alternating_collator=alternating_collator,
                train_config=train_config,
                **custom_trainer_params,
            )

            trainer.train(model_path=params["output_dir"])
            trainer.save_model()  # type: ignore

        except Exception:
            logger.exception(
                "Exception occurred while running RegressionTransformerTrainingPipeline."
            )

    def setup_model(self, params: Dict[str, Any]):
        """
        Error handling and training setup routine.

        Args:
            params: A dictionary with all parameters to launch training.

        Raises:
            ValueError: If flawed values are passed.
        """

        if params.get("test_data_path") is None and params["do_eval"]:
            raise ValueError(
                "Cannot do evaluation without an evaluation data file. Either supply a "
                "file to --test_data_path or remove the --do_eval argument."
            )
        if params["output_dir"] is None:
            raise ValueError(
                "You have to specify an output directory for the trained model"
            )
        if (
            os.path.exists(params["output_dir"])
            and os.listdir(params["output_dir"])
            and not params["overwrite_output_dir"]
        ):
            raise ValueError(
                f"Output directory ({params['output_dir']}) exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

        # Set seed
        set_seed(params["seed"])

        # Load model configuration.
        # Priorities:
        #   1) If provided, take it from the model path: `model_path`.
        #   2) If provided, take model configuration from `config_name`.
        #   3) Instantiate a fresh model.

        if params["model_path"] is None and params["model_type"] is None:
            raise ValueError(
                "Either pass pretrained model via `model_path` or specify"
                "which model to use via `model_typ`."
            )
        if params["model_path"]:
            config = AutoConfig.from_pretrained(
                params["model_path"],
                cache_dir=params["cache_dir"],
            )
            self.model_params = config.__dict__

            self.model = AutoModelWithLMHead.from_pretrained(
                params["model_path"],
                from_tf=False,
                config=config,
                cache_dir=params["cache_dir"],
            )
            logger.info(f"Model restored from {params['model_path']}")

        elif params["config_name"] is not None:
            with open(params["config_name"], "r") as f:
                self.model_params = json.load(f)
            config = AutoConfig.from_pretrained(
                params["config_name"],
                cache_dir=params["cache_dir"],
                mem_len=self.model_params.get("mem_len", 1024),
            )

        else:
            config = CONFIG_MAPPING[params["model_type"]]()
            self.model_params = config.__dict__
            logger.warning(
                f"Instantiating a new config instance: {params['model_type']}."
            )

        # Load tokenizer
        if params["model_path"]:
            # If model_path was provided we load tokenizer from there
            self.tokenizer = ExpressionBertTokenizer.from_pretrained(
                params["model_path"], cache_dir=params["cache_dir"]
            )
        elif params["tokenizer_name"]:
            self.tokenizer = ExpressionBertTokenizer.from_pretrained(
                params["tokenizer_name"], cache_dir=params["cache_dir"]
            )
        else:
            raise ValueError(
                f"No support for creating new tokenizer for: {params['model_type']}."
            )

        if not params["model_path"]:
            logger.info("Training new model from scratch")
            self.model = AutoModelWithLMHead.from_config(config)

    def setup_dataset(
        self,
        train_data_path: str,
        test_data_path: str,
        augment: int = 0,
        save_datasets: bool = False,
        *args,
        **kwargs,
    ):
        """
        Constructs the dataset objects.


        Args:
            train_data_path: Path to `.csv` file. Has to have a `text` column and
                at least one column of numerical properties.
            train_data_path: Path to `.csv` file. Has to have a `text` column and
                at least one column of numerical properties.
            augment: How many times each training sample is augmented.
            save_datasets: Whether to save the datasets to disk (will be stored in
                same location as `train_data_path` and `test_data_path`).

        Returns:
            A tuple of train and test dataset.
        """

        logger.info("Preparing/reading data...")

        tokenizer, properties, train_data, test_data = prepare_datasets_from_files(
            self.tokenizer, train_data_path, test_data_path, augment=augment
        )
        self.tokenizer, self.property_objects = tokenizer, properties
        self.properties = list(properties.keys())
        self.example_sample = train_data[0]

        train_dataset = self.create_dataset_from_list(
            train_data,
            save_path=train_data_path.replace(".csv", ".txt")
            if save_datasets
            else None,
        )
        test_dataset = self.create_dataset_from_list(
            test_data,
            save_path=test_data_path.replace(".csv", ".txt") if save_datasets else None,
        )
        logger.info("Finished data setup.")
        return train_dataset, test_dataset

    def create_dataset_from_list(
        self, data: List[str], save_path: Optional[str] = None
    ) -> LineByLineTextDataset:
        """
        Creates a LineByLineTextDataset from a List of strings.

        Args:
            data: List of strings with the samples.
            save_path: Path to save the dataset to. Defaults to None, meaning
                the dataset will not be saved.
        Returns:
            The dataset.
        """
        # Write files to temporary location and create data
        with tempfile.TemporaryDirectory() as temp:
            f_name = os.path.join(temp, "tmp_data.txt")

            # Write file
            with open(f_name, "w") as f:
                for line in data:
                    f.write(line + "\n")

            # Create dataset
            dataset = LineByLineTextDataset(
                file_path=f_name, tokenizer=self.tokenizer, block_size=2**64
            )
            if save_path:
                shutil.copyfile(f_name, save_path)
        return dataset


@dataclass
class RegressionTransformerModelArguments(TrainingPipelineArguments):
    """Arguments pertaining to model instantiation."""

    __name__ = "model_args"

    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path where the pretrained model artifacts are stored."},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path. If not provided, will be "
            "inferred from `model_path`. If `model_path` is not provided either you "
            "have to pass a tokenizer."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path. But `model_path` takes preference."
        },
    )
    model_type: Optional[str] = field(
        default="xlnet",
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            f"{', '.join(MODEL_TYPES)}. If `model_path` is also provided, `model_path` "
            "takes preference."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
