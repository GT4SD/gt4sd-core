"""TorchDrugGenerationAlgorithm initialization."""
from .checks import TORCH_OPENMP_CHECK, torch
from .core import TorchDrugGCPN, TorchDrugGenerator, TorchDrugGraphAF

# re-enable openMP after torchdrug imports if applicable
if TORCH_OPENMP_CHECK:
    torch._C.has_openmp = True

__all__ = ["TorchDrugGenerator", "TorchDrugGCPN", "TorchDrugGraphAF"]
