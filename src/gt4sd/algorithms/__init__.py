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
"""Module initialization for gt4sd."""
from ..extras import EXTRAS_ENABLED

# NOTE: here we import the applications to register them
from .conditional_generation.guacamol.core import (  # noqa: F401
    AaeGenerator,
    GraphGAGenerator,
    GraphMCTSGenerator,
    OrganGenerator,
    SMILESGAGenerator,
    SMILESLSTMHCGenerator,
    SMILESLSTMPPOGenerator,
    VaeGenerator,
)
from .conditional_generation.key_bert.core import KeyBERTGenerator  # noqa: F401
from .conditional_generation.paccmann_rl.core import (  # noqa: F401
    PaccMannRLOmicBasedGenerator,
    PaccMannRLProteinBasedGenerator,
)
from .conditional_generation.regression_transformer.core import (  # noqa: F401
    RegressionTransformerMolecules,
    RegressionTransformerProteins,
)
from .conditional_generation.reinvent.core import ReinventGenerator  # noqa: F401
from .conditional_generation.template.core import TemplateGenerator  # noqa: F401
from .controlled_sampling.advanced_manufacturing.core import (  # noqa: F401
    CatalystGenerator,
)
from .controlled_sampling.paccmann_gp.core import PaccMannGPGenerator  # noqa: F401
from .generation.hugging_face.core import (  # noqa: F401
    HuggingFaceCTRLGenerator,
    HuggingFaceGPT2Generator,
    HuggingFaceOpenAIGPTGenerator,
    HuggingFaceTransfoXLGenerator,
    HuggingFaceXLMGenerator,
    HuggingFaceXLNetGenerator,
)
from .generation.moler.core import MoLeRDefaultGenerator  # noqa: F401
from .generation.pgt.core import (  # noqa: F401
    PGTCoherenceChecker,
    PGTEditor,
    PGTGenerator,
)
from .generation.polymer_blocks.core import PolymerBlocksGenerator  # noqa: F401
from .generation.torchdrug.core import TorchDrugGCPN, TorchDrugGraphAF  # noqa: F401
from .prediction.paccmann.core import PaccMann  # noqa: F401
from .prediction.topics_zero_shot.core import TopicsPredictor  # noqa: F401

# extras requirements
if EXTRAS_ENABLED:
    from .controlled_sampling.class_controlled_sampling.core import (  # noqa: F401
        PAG,
        CogMol,
    )
    from .generation.molgx.core import MolGXQM9Generator  # noqa: F401
