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
"""Make pytest fixtures available to multiple test directories."""
import atexit
from contextlib import ExitStack
from pathlib import PosixPath

import importlib_resources
import pytest


@pytest.fixture
def mock_wrong_s3_env(monkeypatch):
    """Changes an environment variable to break the s3 connection."""
    monkeypatch.setenv("GT4SD_S3_SECRET_KEY", "(╯°□°）╯︵ ┻━┻")


def exitclose_file_creator(file_path: str) -> PosixPath:
    """
    Creates an absolute filepath that is closed at exit time.

    Args:
        file_path: A relative path to a file for which the context handler is created.

    Returns:
        PosixPath: An absolute filepath.
    """

    file_manager = ExitStack()
    atexit.register(file_manager.close)
    ref = importlib_resources.files("gt4sd") / file_path
    absolute_path = file_manager.enter_context(importlib_resources.as_file(ref))
    return absolute_path
