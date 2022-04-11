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
