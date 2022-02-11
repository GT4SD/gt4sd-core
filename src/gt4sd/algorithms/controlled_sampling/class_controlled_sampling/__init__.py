"""Controlled Latent attribute Space Sampling initialization."""
import logging

from ....extras import EXTRAS_ENABLED

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if EXTRAS_ENABLED:
    from .core import PAG, CLaSS, CogMol

    __all__ = ["CLaSS", "CogMol", "PAG"]
else:
    logger.warning("install cogmol-inference extras to use CLaSS")
