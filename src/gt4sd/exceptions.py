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
"""Custom exception definitions."""


class S3SyncError(RuntimeError):
    """Error in syncing the cache with S3."""

    def __init__(self, title: str, detail: str) -> None:
        """Initialize S3SyncError.

        Args:
            title: title of the error.
            detail: description of the error.
        """
        self.type = "S3SyncError"
        self.title = title
        self.detail = detail
        super().__init__(detail)


class InvalidItem(ValueError):
    """Error in validating an item."""

    def __init__(self, title: str, detail: str) -> None:
        """Initialize InvalidItem.

        Args:
            title: title of the error.
            detail: description of the error.
        """
        self.type = "InvalidItem"
        self.title = title
        self.detail = detail
        super().__init__(detail)


class InvalidAlgorithmConfiguration(ValueError):
    """Error in validating an algorithm configuration."""

    def __init__(self, title: str, detail: str) -> None:
        """Initialize InvalidAlgorithmConfiguration.

        Args:
            title: title of the error.
            detail: description of the error.
        """
        self.type = "InvalidAlgorithmConfiguration"
        self.title = title
        self.detail = detail
        super().__init__(detail)


class DuplicateApplicationRegistration(ValueError):
    """Error when identifier for a registration is not unique."""

    def __init__(self, title: str, detail: str) -> None:
        """Initialize DuplicateApplicationRegistration.

        Args:
            title: title of the error.
            detail: description of the error.
        """
        self.type = "InvalidAlgorithmConfiguration"
        self.title = title
        self.detail = detail
        super().__init__(detail)


class SamplingError(TimeoutError):
    """Error when inference takes too long."""

    def __init__(self, title: str, detail: str) -> None:
        """Initialize SamplingError.

        Args:
            title: title of the error.
            detail: description of the error.
        """
        self.type = "SamplingError"
        self.title = title
        self.detail = detail
        super().__init__(detail)


class GT4SDTimeoutError(TimeoutError):
    """Error for timeouts in gt4sd."""

    def __init__(self, title: str, detail: str) -> None:
        """Initialize SamplingError.

        Args:
            title: title of the error.
            detail: description of the error.
        """
        self.type = "GT4SDTimeoutError"
        self.title = title
        self.detail = detail
        super().__init__(detail)
