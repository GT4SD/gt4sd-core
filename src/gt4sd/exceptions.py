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
