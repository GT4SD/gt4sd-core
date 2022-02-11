"""Model initialization module."""

from typing import Dict, Type

from .activation import ACTIVATION_FACTORY  # noqa: F401
from .base_model import GranularBaseModel, GranularEncoderDecoderModel
from .loss import LOSS_FACTORY  # noqa: F401
from .mlp_auto_encoder import MlpAutoEncoder
from .mlp_predictor import MlpPredictor
from .no_encoding import NoEncoding
from .vae_mlp import VaeMlp
from .vae_rnn import VaeRnn
from .vae_trans import VaeTrans

ARCHITECTURE_FACTORY: Dict[str, Type[GranularBaseModel]] = {
    "vae_rnn": VaeRnn,
    "vae_trans": VaeTrans,
    "mlp_predictor": MlpPredictor,
    "no_encoding": NoEncoding,
    "mlp_autoencoder": MlpAutoEncoder,
    "vae_mlp": VaeMlp,
}
AUTOENCODER_ARCHITECTURES = set(
    [
        model_type
        for model_type, model_class in ARCHITECTURE_FACTORY.items()
        if issubclass(model_class, GranularEncoderDecoderModel)
    ]
)
