"""Utilities used in the tests."""

from functools import lru_cache

from pydantic import BaseSettings


class GT4SDTestSettings(BaseSettings):
    """Utility variables for the tests setup."""

    gt4sd_s3_host: str = "s3.mil01.cloud-object-storage.appdomain.cloud"
    gt4sd_s3_access_key: str = "a19f93a1c67949f1a31db38e58bcb7e8"
    gt4sd_s3_secret_key: str = "5748375c761a4f09c30a68cd15e218e3b27ca3e2aebd7726"
    gt4sd_s3_secure: bool = True
    gt4sd_ci: bool = False

    class Config:
        # immutable and in turn hashable, that is required for lru_cache
        frozen = True

    @staticmethod
    @lru_cache(maxsize=None)
    def get_instance() -> "GT4SDTestSettings":
        return GT4SDTestSettings()
