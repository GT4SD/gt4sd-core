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
