"""Module initialization for gt4sd traning pipelines."""

import json
import logging
from typing import Any, Dict

import pkg_resources

from ..cli.load_arguments_from_dataclass import extract_fields_from_class
from .guacamol_baselines.smiles_lstm_hc.core import (
    GuacamolLSTMHCDataArguments,
    GuacamolLSTMHCModelArguments,
    GuacamolLSTMHCSavingArguments,
    GuacamolLSTMHCTrainingArguments,
    GuacamolLSTMHCTrainingPipeline,
)
from .guacamol_baselines.smiles_lstm_ppo.core import (
    GuacamolLSTMPPOModelArguments,
    GuacamolLSTMPPOSavingArguments,
    GuacamolLSTMPPOTrainingArguments,
    GuacamolLSTMPPOTrainingPipeline,
)
from .moses.core import MosesCommonArguments, MosesSavingArguments
from .moses.organ.core import (
    MosesOrganModelArguments,
    MosesOrganTrainingArguments,
    MosesOrganTrainingPipeline,
)
from .moses.vae.core import (
    MosesVAEModelArguments,
    MosesVAETrainingArguments,
    MosesVAETrainingPipeline,
)
from .paccmann.core import (
    PaccMannDataArguments,
    PaccMannSavingArguments,
    PaccMannTrainingArguments,
)
from .paccmann.vae.core import PaccMannVAEModelArguments, PaccMannVAETrainingPipeline
from .pytorch_lightning.core import PytorchLightningTrainingArguments
from .pytorch_lightning.granular.core import (
    GranularDataArguments,
    GranularModelArguments,
    GranularSavingArguments,
    GranularTrainingPipeline,
)
from .pytorch_lightning.language_modeling.core import (
    LanguageModelingDataArguments,
    LanguageModelingModelArguments,
    LanguageModelingSavingArguments,
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
    "guacamol-lstm-hc-trainer": (
        GuacamolLSTMHCModelArguments,
        GuacamolLSTMHCTrainingArguments,
        GuacamolLSTMHCDataArguments,
    ),
    "guacamol-lstm-ppo-trainer": (
        GuacamolLSTMPPOModelArguments,
        GuacamolLSTMPPOTrainingArguments,
    ),
    "moses-vae-trainer": (
        MosesVAETrainingArguments,
        MosesVAEModelArguments,
        MosesCommonArguments,
    ),
    "moses-organ-trainer": (
        MosesOrganTrainingArguments,
        MosesOrganModelArguments,
        MosesCommonArguments,
    ),
}

TRAINING_PIPELINE_MAPPING = {
    "language-modeling-trainer": LanguageModelingTrainingPipeline,
    "paccmann-vae-trainer": PaccMannVAETrainingPipeline,
    "granular-trainer": GranularTrainingPipeline,
    "guacamol-lstm-hc-trainer": GuacamolLSTMHCTrainingPipeline,
    "guacamol-lstm-ppo-trainer": GuacamolLSTMPPOTrainingPipeline,
    "moses-organ-trainer": MosesOrganTrainingPipeline,
    "moses-vae-trainer": MosesVAETrainingPipeline,
}

TRAINING_PIPELINE_ARGUMENTS_FOR_MODEL_SAVING = {
    "paccmann-vae-trainer": PaccMannSavingArguments,
    "granular-trainer": GranularSavingArguments,
    "language-modeling-trainer": LanguageModelingSavingArguments,
    "guacamol-lstm-hc-trainer": GuacamolLSTMHCSavingArguments,
    "guacamol-lstm-ppo-trainer": GuacamolLSTMPPOSavingArguments,
    "moses-organ-trainer": MosesSavingArguments,
    "moses-vae-trainer": MosesSavingArguments,
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
