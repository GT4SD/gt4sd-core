"""Module initialization for gt4sd."""
from ..extras import EXTRAS_ENABLED
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

# here we import the applications to register them
from .conditional_generation.paccmann_rl.core import (  # noqa: F401
    PaccMannRLOmicBasedGenerator,
    PaccMannRLProteinBasedGenerator,
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
from .generation.polymer_blocks.core import PolymerBlocksGenerator  # noqa: F401
from .prediction.topics_zero_shot.core import TopicsPredictor  # noqa: F401

# extras requirements
if EXTRAS_ENABLED:
    from .controlled_sampling.class_controlled_sampling.core import (  # noqa: F401
        PAG,
        CogMol,
    )
    from .generation.molgx.core import MolGXQM9Generator  # noqa: F401
