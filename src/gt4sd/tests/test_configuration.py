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
"""Configuration tests."""

import os

from gt4sd.configuration import (
    GT4SDConfiguration,
    get_algorithm_subdirectories_in_cache,
    get_algorithm_subdirectories_with_s3,
)

gt4sd_configuration_instance = GT4SDConfiguration.get_instance()


def test_default_local_cache_path():
    if "GT4SD_LOCAL_CACHE_PATH" not in os.environ:
        assert os.path.dirname(
            gt4sd_configuration_instance.gt4sd_local_cache_path
        ) == os.path.expanduser("~")
        assert (
            os.path.basename(gt4sd_configuration_instance.gt4sd_local_cache_path)
            == ".gt4sd"
        )
    else:
        assert (
            gt4sd_configuration_instance.gt4sd_local_cache_path
            == os.environ["GT4SD_LOCAL_CACHE_PATH"]
        )


def test_get_algorithm_subdirectories_with_s3():
    assert isinstance(get_algorithm_subdirectories_with_s3(), set)


def test_get_algorithm_subdirectories_in_cache():
    assert isinstance(get_algorithm_subdirectories_in_cache(), set)
