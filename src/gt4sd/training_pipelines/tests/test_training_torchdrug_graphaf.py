"""TorchDrug GraphAF trainer unit tests."""

import shutil
import tempfile
from typing import Any, Dict, cast

import pkg_resources

from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_MAPPING,
    TorchDrugGCPNTrainingPipeline,
)

template_config = {
    "model_args": {
        "hidden_dims": "[128, 128]",
        "batch_norm": True,
        "short_cut": True,
        "concat_hidden": True,
        "readout": "mean",
        "hidden_dim_mlp": 128,
        "agent_update_interval": 16,
        "gamma": 0.95,
        "reward_temperature": 1.2,
        "criterion": "{'nll': 1}",
    },
    "dataset_args": {
        "dataset_name": "freesolv",
        "lazy": True,
        "no_kekulization": False,
    },
    "training_args": {
        "model_path": "/tmp/torchdrug-graphaf",
        "training_name": "torchdrug-graphaf-test",
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.0005,
        "log_interval": 2,
        "gradient_interval": 2,
    },
}


def test_train():

    pipeline = TRAINING_PIPELINE_MAPPING.get("torchdrug-graphaf-trainer")

    assert pipeline is not None

    TEMPORARY_DIRECTORY = tempfile.mkdtemp()

    test_pipeline = cast(TorchDrugGCPNTrainingPipeline, pipeline())

    config: Dict[str, Any] = template_config.copy()
    config["training_args"]["model_path"] = TEMPORARY_DIRECTORY
    config["training_args"]["dataset_path"] = TEMPORARY_DIRECTORY

    test_pipeline.train(**config)
    shutil.rmtree(TEMPORARY_DIRECTORY)

    # Now test with a custom dataset
    file_path = pkg_resources.resource_filename(
        "gt4sd",
        "training_pipelines/tests/molecules.csv",
    )
    config["dataset_args"]["dataset_name"] = "custom"
    config["dataset_args"]["file_path"] = file_path
    config["dataset_args"]["smiles_field"] = "smiles"
    config["dataset_args"]["target_field"] = "qed"

    test_pipeline.train(**config)
    shutil.rmtree(TEMPORARY_DIRECTORY)

    # Test the property optimization
    """
    Disabled for now due to multiple downstream torchdrug issues, most importantly:
    - https://github.com/DeepGraphLearning/torchdrug/issues/83
    """
    # config["dataset_args"]["target_field"] = 'qed'
    # config['training_args']['task'] = 'qed'
    # config['data_args']['node_feature'] = 'symbol'
    # config['model_args']['criterion'] = "{'ppo': 1}"

    # test_pipeline.train(**config)
    # shutil.rmtree(TEMPORARY_DIRECTORY)
