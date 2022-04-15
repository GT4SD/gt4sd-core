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

import pytest

from gt4sd.exceptions import InvalidAlgorithmConfiguration, InvalidItem, S3SyncError


def test_s3_sync_error():
    error = S3SyncError("GenericSyncError", "my message")
    assert error.type == "S3SyncError"
    assert error.title == "GenericSyncError"
    assert error.detail == "my message"
    with pytest.raises(RuntimeError):
        str(error) == error.detail
        raise error


def test_invalid_item():
    error = InvalidItem("GenericInvaliItemError", "my message")
    assert error.type == "InvalidItem"
    assert error.title == "GenericInvaliItemError"
    assert error.detail == "my message"
    with pytest.raises(ValueError):
        str(error) == error.detail
        raise error


def test_invalid_algorithm_configuration():
    error = InvalidAlgorithmConfiguration(
        "GenericAlgorithmConfigurationError", "my message"
    )
    assert error.type == "InvalidAlgorithmConfiguration"
    assert error.title == "GenericAlgorithmConfigurationError"
    assert error.detail == "my message"
    with pytest.raises(ValueError):
        str(error) == error.detail
        raise error
