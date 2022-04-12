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
