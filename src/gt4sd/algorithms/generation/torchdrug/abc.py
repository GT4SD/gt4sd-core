import torch

# disable openmp usage since this raises on MacOS when libomp has the wrong version.
openmp = torch._C.has_openmp
if openmp:
    torch._C.has_openmp = False
