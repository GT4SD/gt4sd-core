"""TorchDrugGenerationAlgorithm initialization."""
from .abc import openmp, torch
from .core import TorchDrugGCPN, TorchDrugGenerator, TorchDrugGraphAF

# Re-enable openMP after torchdrug imports if applicable
if openmp:
    torch._C.has_openmp = True

__all__ = ["TorchDrugGenerator", "TorchDrugGCPN", "TorchDrugGraphAF"]
