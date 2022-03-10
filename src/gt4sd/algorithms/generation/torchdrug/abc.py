import torch

# Disable openmp usage since this raises on MacOS when libomp has the wrong version.
torch._C.has_openmp = False
