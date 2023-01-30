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
"""Crystals crf training utilities."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ...frameworks.crystals_rfc.rf_classifier import RFC
from ..core import TrainingPipeline, TrainingPipelineArguments

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class CrystalsRFCTrainingPipeline(TrainingPipeline):
    """Crystals RFC training pipelines for crystals."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
    ) -> None:
        """Generic training function for Crystals RFC models.

        Args:
            training_args: training arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.

        Raises:
            NotImplementedError: the generic trainer does not implement the pipeline.
        """

        rfc = RFC(crystal_sys=model_args["sym"])
        df = rfc.load_data(file_name=dataset_args["datapath"])
        train_x, test_x, train_y, test_y = rfc.split_data(
            df, test_size=dataset_args["test_size"]
        )
        train_x, test_x, train_y, test_y = rfc.normalize_data(
            train_x, test_x, train_y, test_y
        )

        rfc.train(train_x, train_y)

        rfc.save(training_args["output_path"])


@dataclass
class CrystalsRFCDataArguments(TrainingPipelineArguments):
    """Data arguments related to crystals RFC trainer."""

    __name__ = "dataset_args"

    datapath: str = field(
        metadata={
            "help": "Path to the dataset."
            "The dataset should follow the directory structure as described in https://github.com/dilangaem/SemiconAI."
        },
    )
    test_size: Optional[int] = field(
        default=None, metadata={"help": "Testing set percentage."}
    )


@dataclass
class CrystalsRFCModelArguments(TrainingPipelineArguments):
    """Model arguments related to crystals RFC trainer."""

    __name__ = "model_args"

    sym: str = field(
        default="all",
        metadata={
            "help": "Crystal systems to be used. 'all' for all the crystal systems. Other seven options are: 'monoclinic', 'triclinic', 'orthorhombic', 'trigonal', 'hexagonal', 'cubic', 'tetragonal'"
        },
    )


@dataclass
class CrystalsRFCTrainingArguments(TrainingPipelineArguments):
    """Training arguments related to crystals RFC trainer."""

    __name__ = "training_args"

    output_path: str = field(
        default=".",
        metadata={"help": "Path to the store the checkpoints."},
    )


@dataclass
class CrystalsRFCSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to crystals RFC trainer."""

    __name__ = "saving_args"
