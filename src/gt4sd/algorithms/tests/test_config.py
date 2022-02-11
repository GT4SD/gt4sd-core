"""Tests for AlgorithmConfiguration."""

import os
import shutil
from typing import ClassVar

import pytest

from gt4sd.algorithms.core import AlgorithmConfiguration
from gt4sd.configuration import GT4SDConfiguration

gt4sd_configuration_instance = GT4SDConfiguration.get_instance()


@pytest.fixture()
def development_version_path():
    # setup
    path = os.path.join(
        gt4sd_configuration_instance.gt4sd_local_cache_path,
        gt4sd_configuration_instance.gt4sd_local_cache_path_algorithms,
        "dummy",
        "algorithm",
        "config",
        "development",
    )
    os.makedirs(path, exist_ok=False)
    # test
    yield path
    # teardown
    shutil.rmtree(
        os.path.join(
            gt4sd_configuration_instance.gt4sd_local_cache_path,
            gt4sd_configuration_instance.gt4sd_local_cache_path_algorithms,
            "dummy",
        )
    )


def test_list_versions_local_only(development_version_path):
    class Config(AlgorithmConfiguration):
        algorithm_type: ClassVar[str] = "dummy"
        domain: ClassVar[str] = ""
        algorithm_name: ClassVar[str] = "algorithm"
        algorithm_application: ClassVar[str] = "config"
        algorithm_version: str = "development"

    assert "development" in Config.list_versions()
