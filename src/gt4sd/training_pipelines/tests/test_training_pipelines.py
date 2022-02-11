"""Exceptions tests."""

from gt4sd.training_pipelines import (
    TRAINING_PIPELINE_ARGUMENTS_MAPPING,
    TRAINING_PIPELINE_NAME_METADATA_MAPPING,
    training_pipeline_name_to_metadata,
)


def test_metadata_retrieval_for_registered_pipelines_from_json():
    for name, filename in TRAINING_PIPELINE_NAME_METADATA_MAPPING.items():
        pipeline_metadata = training_pipeline_name_to_metadata(name)
        assert pipeline_metadata["training_pipeline"] == name
        assert "description" in pipeline_metadata
        assert "parameters" in pipeline_metadata
        assert "description" not in pipeline_metadata["parameters"]


def test_metadata_retrieval_for_registered_pipelines_from_dataclass():
    for name, filename in TRAINING_PIPELINE_ARGUMENTS_MAPPING.items():
        pipeline_metadata = training_pipeline_name_to_metadata(name)
        assert pipeline_metadata["training_pipeline"] == name
        assert "description" in pipeline_metadata
        assert "parameters" in pipeline_metadata
        assert "description" not in pipeline_metadata["parameters"]

        for parameter in pipeline_metadata["parameters"]:
            assert "description" in pipeline_metadata["parameters"][parameter]
            assert "type" in pipeline_metadata["parameters"][parameter]

            assert len(pipeline_metadata["parameters"][parameter]) <= 3

            if len(pipeline_metadata["parameters"][parameter]) == 3:
                assert "default" in pipeline_metadata["parameters"][parameter]


def test_metadata_retrieval_for_unregistered_pipelines():
    name = "this pipeline does not exists and can't be registered"
    pipeline_metadata = training_pipeline_name_to_metadata(name)
    assert pipeline_metadata["training_pipeline"] == name
    assert pipeline_metadata["description"] == "A training pipeline."
    assert pipeline_metadata["parameters"] == {}
