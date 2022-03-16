"""Module initialization for gt4sd traning pipelines."""

import json
import logging
from typing import Any, Dict

import pkg_resources

from ..cli.load_arguments_from_dataclass import extract_fields_from_class
from .paccmann.core import PaccMannDataArguments, PaccMannTrainingArguments
from .paccmann.vae.core import PaccMannVAEModelArguments, PaccMannVAETrainingPipeline
from .guacamol_baselines.smiles_lstm_hc.core import (
    SMILESLSTMHCTrainingPipeline,
    SMILESLSTMHCModelArguments,
    SMILESLSTMHCDataArguments,
)
from .guacamol_baselines.smiles_lstm_ppo.core import (
    SMILESLSTMPPOTrainingPipeline,
    SMILESLSTMPPOModelArguments,
)
from .moses.core import MosesCommonArguments
from .moses.aae.core import (
    MosesAAETrainingPipeline,
    MosesAAEModelArguments,
    MosesAAETrainingArguments,
)
from .moses.vae.core import (
    MosesVAETrainingPipeline,
    MosesVAEModelArguments,
    MosesVAETrainingArguments,
)
from .moses.organ.core import (
    MosesOrganTrainingPipeline,
    MosesOrganModelArguments,
)
from .pytorch_lightning.core import PytorchLightningTrainingArguments
from .pytorch_lightning.granular.core import (
    GranularDataArguments,
    GranularModelArguments,
    GranularTrainingPipeline,
)
from .pytorch_lightning.language_modeling.core import (
    LanguageModelingDataArguments,
    LanguageModelingModelArguments,
    LanguageModelingTrainingPipeline,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

TRAINING_PIPELINE_NAME_METADATA_MAPPING = {
    "mock_training_pipeline": "mock_training_pipeline.json",
    "Terminator training": "terminator_training.json",
}

TRAINING_PIPELINE_ARGUMENTS_MAPPING = {
    "language-modeling-trainer": (
        PytorchLightningTrainingArguments,
        LanguageModelingDataArguments,
        LanguageModelingModelArguments,
    ),
    "paccmann-vae-trainer": (
        PaccMannTrainingArguments,
        PaccMannDataArguments,
        PaccMannVAEModelArguments,
    ),
    "granular-trainer": (
        PytorchLightningTrainingArguments,
        GranularDataArguments,
        GranularModelArguments,
    ),
    "smiles-lstm-hc-trainer": (
        SMILESLSTMHCModelArguments,
        SMILESLSTMHCDataArguments,
    ),
    "smiles-lstm-ppo-trainer": (SMILESLSTMPPOModelArguments),
    "moses-vae-trainer": (
        MosesVAETrainingArguments,
        MosesVAEModelArguments,
        MosesCommonArguments,
    ),
    "moses-aae-trainer": (
        MosesAAETrainingArguments,
        MosesAAEModelArguments,
        MosesCommonArguments,
    ),
    "moses-organ-trainer": (
        MosesOrganModelArguments,
        MosesCommonArguments,
    ),
}

TRAINING_PIPELINE_MAPPING = {
    "language-modeling-trainer": LanguageModelingTrainingPipeline,
    "paccmann-vae-trainer": PaccMannVAETrainingPipeline,
    "granular-trainer": GranularTrainingPipeline,
    "smiles-lstm-hc-trainer": SMILESLSTMHCTrainingPipeline,
    "smiles-lstm-ppo-trainer": SMILESLSTMPPOTrainingPipeline,
    "moses-aae-trainer": MosesAAETrainingPipeline,
    "moses-vae-trainer": MosesVAETrainingPipeline,
    "moses-organ-trainer": MosesOrganTrainingPipeline,
}


def training_pipeline_name_to_metadata(name: str) -> Dict[str, Any]:
    """Retrieve training pipeline metadata from the name.

    Args:
        name: name of the pipeline.

    Returns:
        dictionary describing the parameters of the pipeline. If the pipeline is not found, no metadata (a.k.a., an empty dictionary is returned).
    """
    metadata: Dict[str, Any] = {"training_pipeline": name, "parameters": {}}
    if name in TRAINING_PIPELINE_NAME_METADATA_MAPPING:
        try:
            with open(
                pkg_resources.resource_filename(
                    "gt4sd",
                    f"training_pipelines/{TRAINING_PIPELINE_NAME_METADATA_MAPPING[name]}",
                ),
                "rt",
            ) as fp:
                metadata["parameters"] = json.load(fp)
        except Exception:
            logger.exception(
                f'training pipeline "{name}" metadata fetching failed, returning an empty metadata dictionary'
            )

    elif name in TRAINING_PIPELINE_ARGUMENTS_MAPPING:

        for training_argument_class in TRAINING_PIPELINE_ARGUMENTS_MAPPING[name]:
            field_types = extract_fields_from_class(training_argument_class)
            metadata["parameters"].update(field_types)

    else:
        logger.warning(
            f'training pipeline "{name}" metadata not found, returning an empty metadata dictionary'
        )
    metadata["description"] = metadata["parameters"].pop(
        "description", "A training pipeline."
    )
    return metadata
