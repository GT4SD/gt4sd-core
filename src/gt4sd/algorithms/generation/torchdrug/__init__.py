"""TorchDrugGenerationAlgorithm initialization."""
from .abc import torch
from .core import (
    TorchDrugGenerator,
    TorchDrugPlogpGAF,
    TorchDrugPlogpGCPN,
    TorchDrugQedGAF,
    TorchDrugQedGCPN,
    TorchDrugZincGAF,
    TorchDrugZincGCPN,
)

torch._C.has_openmp = True

__all__ = [
    "TorchDrugGenerator",
    "TorchDrugZincGCPN",
    "TorchDrugQedGCPN",
    "TorchDrugPlogpGCPN",
    "TorchDrugZincGAF",
    "TorchDrugQedGAF",
    "TorchDrugPlogpGAF",
]
