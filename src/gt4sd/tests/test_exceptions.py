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
