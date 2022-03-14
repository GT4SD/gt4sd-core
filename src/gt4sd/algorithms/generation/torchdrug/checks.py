import torch

# disable openmp usage since this raises on MacOS when libomp has the wrong version.
TORCH_OPENMP_CHECK = torch._C.has_openmp
if TORCH_OPENMP_CHECK:
    torch._C.has_openmp = False
