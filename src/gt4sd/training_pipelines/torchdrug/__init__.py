#
# MIT License
#
# Copyright (c) 2022 GT4SD team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

from .unpatch import (  # isort:skip
    fix_datasets,
    sane_datasets,
    fix_schedulers,
    sane_schedulers,
)

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
fix_datasets(sane_datasets)
fix_schedulers(sane_schedulers)


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
