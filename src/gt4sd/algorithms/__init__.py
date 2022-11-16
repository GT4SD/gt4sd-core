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
from .conditional_generation.guacamol import (  # noqa: F401
    AaeGenerator,
    GraphGAGenerator,
    GraphMCTSGenerator,
    OrganGenerator,
    SMILESGAGenerator,
    SMILESLSTMHCGenerator,
    SMILESLSTMPPOGenerator,
    VaeGenerator,
)
from .conditional_generation.key_bert import KeyBERTGenerator  # noqa: F401
from .conditional_generation.molgx import MolGXQM9Generator  # noqa: F401
from .conditional_generation.paccmann_rl import (  # noqa: F401
    PaccMannRLOmicBasedGenerator,
    PaccMannRLProteinBasedGenerator,
)
from .conditional_generation.regression_transformer import (  # noqa: F401
    RegressionTransformerMolecules,
    RegressionTransformerProteins,
)
from .conditional_generation.reinvent import ReinventGenerator  # noqa: F401
from .conditional_generation.template import TemplateGenerator  # noqa: F401
from .controlled_sampling.advanced_manufacturing import CatalystGenerator  # noqa: F401
from .controlled_sampling.paccmann_gp import PaccMannGPGenerator  # noqa: F401
from .generation.diffusion import (  # noqa: F401
    DDIMGenerator,
    DDPMGenerator,
    LDMGenerator,
    LDMTextToImageGenerator,
    ScoreSdeGenerator,
    StableDiffusionGenerator,
)
from .generation.hugging_face import (  # noqa: F401
    HuggingFaceCTRLGenerator,
    HuggingFaceGPT2Generator,
    HuggingFaceOpenAIGPTGenerator,
    HuggingFaceTransfoXLGenerator,
    HuggingFaceXLMGenerator,
    HuggingFaceXLNetGenerator,
)
from .generation.moler import MoLeRDefaultGenerator  # noqa: F401
from .generation.pgt import PGTCoherenceChecker, PGTEditor, PGTGenerator  # noqa: F401
from .generation.polymer_blocks import PolymerBlocksGenerator  # noqa: F401
from .generation.torchdrug import TorchDrugGCPN, TorchDrugGraphAF  # noqa: F401
from .prediction.paccmann import PaccMann  # noqa: F401
from .prediction.topics_zero_shot import TopicsPredictor  # noqa: F401

# extras requirements
if EXTRAS_ENABLED:
    from .controlled_sampling.class_controlled_sampling import PAG, CogMol  # noqa: F401
