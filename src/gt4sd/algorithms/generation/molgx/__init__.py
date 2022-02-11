"""MolGX initialization."""
import logging

from ....extras import EXTRAS_ENABLED

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if EXTRAS_ENABLED:
    from .core import MolGX, MolGXQM9Generator

    __all__ = [
        "MolGX",
        "MolGXQM9Generator",
    ]
else:
    logger.warning("install AMD_analytcs extras to use MolGX")
