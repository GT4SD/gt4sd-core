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
