from torchdrug.datasets import (
    BACE,
    BBBP,
    CEP,
    HIV,
    MOSES,
    MUV,
    OPV,
    PCQM4M,
    QM8,
    QM9,
    SIDER,
    ChEMBLFiltered,
    ClinTox,
    Delaney,
    FreeSolv,
    Lipophilicity,
    Malaria,
    PubChem110m,
    Tox21,
    ToxCast,
    ZINC2m,
    ZINC250k,
)

from .dataset import TorchDrugDataset

# isort: off
from torch import nn

"""
Necessary because torchdrug silently overwrites the default nn.Module. This is quite
invasive and causes significant side-effects in the rest of the code.
See: https://github.com/DeepGraphLearning/torchdrug/issues/77
"""
nn.Module = nn._Module  # type: ignore

DATASET_FACTORY = {
    "bace": BACE,
    "bbbp": BBBP,
    "custom": TorchDrugDataset,
    "cep": CEP,
    "chembl": ChEMBLFiltered,
    "clintox": ClinTox,
    "delaney": Delaney,
    "freesolv": FreeSolv,
    "hiv": HIV,
    "lipophilicity": Lipophilicity,
    "malaria": Malaria,
    "moses": MOSES,
    "muv": MUV,
    "opv": OPV,
    "pcqm4m": PCQM4M,
    "pubchem": PubChem110m,
    "qm8": QM8,
    "qm9": QM9,
    "sider": SIDER,
    "tox21": Tox21,
    "toxcast": ToxCast,
    "zinc250k": ZINC250k,
    "zinc2m": ZINC2m,
}
