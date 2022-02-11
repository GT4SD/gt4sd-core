"""Module configuration."""

import logging
import os
from functools import lru_cache
from typing import Optional, Set

from pydantic import BaseSettings

from .s3 import GT4SDS3Client, S3SyncError, sync_folder_with_s3

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GT4SDConfiguration(BaseSettings):
    """GT4SDConfiguration settings from environment variables.

    Default configurations for gt4sd including a read-only COS for algorithms' artifacts.
    """

    gt4sd_local_cache_path: str = os.path.join(os.path.expanduser("~"), ".gt4sd")
    gt4sd_local_cache_path_algorithms: str = "algorithms"
    gt4sd_max_number_of_stuck_calls: int = 50
    gt4sd_max_number_of_samples: int = 1000000
    gt4sd_max_runtime: int = 86400
    gt4sd_s3_host: str = "s3.mil01.cloud-object-storage.appdomain.cloud"
    gt4sd_s3_access_key: str = "a19f93a1c67949f1a31db38e58bcb7e8"
    gt4sd_s3_secret_key: str = "5748375c761a4f09c30a68cd15e218e3b27ca3e2aebd7726"
    gt4sd_s3_secure: bool = True
    gt4sd_s3_bucket: str = "algorithms"

    class Config:
        # immutable and in turn hashable, that is required for lru_cache
        frozen = True

    @staticmethod
    @lru_cache(maxsize=None)
    def get_instance() -> "GT4SDConfiguration":
        return GT4SDConfiguration()


gt4sd_configuration_instance = GT4SDConfiguration.get_instance()
logger.info(
    f"using as local cache path: {gt4sd_configuration_instance.gt4sd_local_cache_path}"
)
try:
    os.makedirs(gt4sd_configuration_instance.gt4sd_local_cache_path)
except FileExistsError:
    logger.debug("local cache path already exists")


def sync_algorithm_with_s3(prefix: Optional[str] = None) -> str:
    """Sync an algorithm in the local cache using environment variables.

    Args:
        prefix: the relative path in the bucket (both
            on S3 and locally) to match files to download. Defaults to None.

    Returns:
        str: local path using the prefix.
    """
    folder_path = os.path.join(
        gt4sd_configuration_instance.gt4sd_local_cache_path,
        gt4sd_configuration_instance.gt4sd_local_cache_path_algorithms,
    )
    try:
        sync_folder_with_s3(
            host=gt4sd_configuration_instance.gt4sd_s3_host,
            access_key=gt4sd_configuration_instance.gt4sd_s3_access_key,
            secret_key=gt4sd_configuration_instance.gt4sd_s3_secret_key,
            bucket=gt4sd_configuration_instance.gt4sd_s3_bucket,
            folder_path=folder_path,
            prefix=prefix,
            secure=gt4sd_configuration_instance.gt4sd_s3_secure,
        )
    except S3SyncError:
        logger.exception("error in syncing the cache with S3")
    return os.path.join(folder_path, prefix) if prefix is not None else folder_path


def get_cached_algorithm_path(prefix: Optional[str] = None) -> str:
    return (
        os.path.join(
            gt4sd_configuration_instance.gt4sd_local_cache_path,
            gt4sd_configuration_instance.gt4sd_local_cache_path_algorithms,
            prefix,
        )
        if prefix is not None
        else os.path.join(
            gt4sd_configuration_instance.gt4sd_local_cache_path,
            gt4sd_configuration_instance.gt4sd_local_cache_path_algorithms,
        )
    )


def get_algorithm_subdirectories_with_s3(prefix: Optional[str] = None) -> Set[str]:

    try:
        host = gt4sd_configuration_instance.gt4sd_s3_host
        access_key = gt4sd_configuration_instance.gt4sd_s3_access_key
        secret_key = gt4sd_configuration_instance.gt4sd_s3_secret_key
        secure = gt4sd_configuration_instance.gt4sd_s3_secure
        client = GT4SDS3Client(
            host=host, access_key=access_key, secret_key=secret_key, secure=secure
        )
        bucket = gt4sd_configuration_instance.gt4sd_s3_bucket
        return client.list_directories(bucket=bucket, prefix=prefix)
    except Exception:
        logger.exception("generic syncing error")
        raise S3SyncError(
            "CacheSyncingError",
            f"error in getting directories of prefix={prefix} with host={host} access_key={access_key} secret_key={secret_key} secure={secure} bucket={bucket}",
        )


def get_algorithm_subdirectories_in_cache(prefix: Optional[str] = None) -> Set[str]:
    path = get_cached_algorithm_path(prefix=prefix)
    try:
        _, dirs, _ = next(iter(os.walk(path)))
        return set(dirs)
    except StopIteration:
        return set()
