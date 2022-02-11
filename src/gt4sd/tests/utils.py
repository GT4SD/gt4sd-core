"""Utilities used in the tests."""

from functools import lru_cache

from pydantic import BaseSettings


class GT4SDTestSettings(BaseSettings):
    """Utility variables for the tests setup."""

    gt4sd_s3_host: str = "localhost:9000"
    gt4sd_s3_access_key: str = "access-key"
    gt4sd_s3_secret_key: str = "secret-key"
    gt4sd_s3_secure: bool = False
    gt4sd_ci: bool = False

    class Config:
        # immutable and in turn hashable, that is required for lru_cache
        frozen = True

    @staticmethod
    @lru_cache(maxsize=None)
    def get_instance() -> "GT4SDTestSettings":
        return GT4SDTestSettings()
