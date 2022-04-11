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
