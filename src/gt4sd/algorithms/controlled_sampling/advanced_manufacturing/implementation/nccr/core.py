"""Catalyst design for NCCR project."""

import logging
import os
import re
from typing import List, Union, cast

import torch

from ......frameworks.granular.ml.models import (
    GranularEncoderDecoderModel,
    MlpPredictor,
)
from ......frameworks.granular.ml.module import GranularModule
from ......frameworks.granular.tokenizer.tokenizer import SmilesTokenizer
from ..core import (
    GaussianProcessRepresentationsSampler,
    Generator,
    Point,
    PropertyPredictor,
    Representation,
    point_to_tensor,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class CatalystVAE(Representation):
    """Catalyst VAE for suzuki reactions."""

    model: GranularEncoderDecoderModel

    def __init__(
        self,
        resources_path: str,
        padding_length: int = 127,
        maximum_length: int = 100,
        primer_smiles: str = "",
    ) -> None:
        """Constructs a CatalystVAE.

        Args:
            resources_path: directory where to find models and configurations.
            pading_length: size of the padded sequence. Defaults to 127.
            maximum_length: maximum length of the synthesis.
            primer_smiles: primer SMILES representation. Default to "", a.k.a., no primer.
        """
        self.vocabulary_filepath = os.path.join(resources_path, "vocab_combined.csv")
        self.checkpoint_filepath = os.path.join(
            resources_path, "epoch=199-step=5799.ckpt"
        )
        self.tokenizer = SmilesTokenizer(self.vocabulary_filepath)
        self.model = cast(
            GranularEncoderDecoderModel,
            GranularModule.load_from_checkpoint(self.checkpoint_filepath).autoencoders[
                0
            ],
        )
        self.model.eval()
        self.padding_length = padding_length
        self.z_dimension = self.model.latent_size
        self.maximum_length = maximum_length
        self.primer_smiles = primer_smiles
        if len(self.primer_smiles) > 0:
            self.primer_point = self.smiles_to_latent(self.primer_smiles)
        else:
            self.primer_point = torch.zeros(1, self.z_dimension)
        self.clean_regex = re.compile(
            r"{}|{}".format(self.tokenizer.sos_token, self.tokenizer.unk_token)
        )
        self.end_regex = re.compile(
            r"{}|{}".format(self.tokenizer.eos_token, self.tokenizer.pad_token)
        )

    def smiles_to_latent(self, smiles: str) -> Point:
        """Encode a SMILES into a latent point.

        Args:
            smiles: a SMILES representation of a molecule.

        Returns:
            the encoded latent space point.
        """
        return self.model.encode(  # type:ignore
            torch.tensor(
                [
                    self.tokenizer.add_padding_tokens(
                        self.tokenizer.convert_tokens_to_ids(
                            [self.tokenizer.sos_token]
                            + self.tokenizer.tokenize(smiles)
                            + [self.tokenizer.eos_token]
                        ),
                        length=self.padding_length,
                    )
                ]
            )
        )

    def decode(self, z: Point) -> str:
        """Decode a catalyst from the latent space.

        Args:
            z: a latent space point.

        Returns:
            a catalyst in SMILES format.
        """
        z = torch.unsqueeze(point_to_tensor(z), dim=0)
        reconstructed = self.model.decode(z, max_len=self.maximum_length)[0][
            0
        ]  # type:ignore
        reconstructed = self.clean_regex.sub("", reconstructed)
        match_ending = self.end_regex.search(reconstructed)
        if match_ending:
            reconstructed = reconstructed[: match_ending.start()]
        return reconstructed


class CatalystBindingEnergyPredictor(PropertyPredictor):
    """Catalyst binding energy predictor for suzuki reactions."""

    model: MlpPredictor

    def __init__(self, resources_path: str) -> None:
        """Constructs a CatalystBindingEnergyPredictor.

        Args:
            resources_path: directory where to find models and configurations.
        """
        self.vocabulary_filepath = os.path.join(resources_path, "vocab_combined.csv")
        self.checkpoint_filepath = os.path.join(
            resources_path, "epoch=199-step=5799.ckpt"
        )
        self.tokenizer = SmilesTokenizer(self.vocabulary_filepath)
        self.model = cast(
            MlpPredictor,
            GranularModule.load_from_checkpoint(self.checkpoint_filepath).latent_models[
                0
            ],
        )
        self.model.eval()

    def __call__(self, z: Point) -> float:
        """Predict binding energy.

        Args:
            z: a latent space point.

        Returns:
            the predicted binding energy.
        """
        z = point_to_tensor(z)
        return self.model(z)[0][0].item()


class CatalystGenerator(Generator):
    """Catalyst generator."""

    def __init__(
        self,
        resources_path: str,
        generated_length: int = 100,
        number_of_points: int = 10,
        number_of_steps: int = 50,
        primer_smiles: str = "",
    ):
        """Constructs catalyst generator.

        Args:
            resource_path: directory where to find models and configurations.
            generated_length: maximum lenght of the generated molecule. Defaults to 100.
            number_of_points: number of optimal points to return. Defaults to 10.
            number_of_steps: number of optimization steps. Defaults to 50.
            primer_smiles: primer SMILES representation. Default to "", a.k.a., no primer.
        """
        self.resources_path = resources_path
        self.generated_length = generated_length
        self.number_of_points = number_of_points
        self.number_of_steps = max(self.number_of_points, number_of_steps)
        self.primer_smiles = primer_smiles
        self.vae = CatalystVAE(
            resources_path,
            maximum_length=self.generated_length,
            primer_smiles=primer_smiles,
        )
        self.predictor = CatalystBindingEnergyPredictor(resources_path)
        self.minimum_latent_coordinate = -100.0
        self.maximum_latent_coordinate = 100.0

    def generate_samples(self, target_energy: Union[float, str]) -> List[str]:
        """Generate samples given a target energy.

        Args:
            target_energy: target energy value.

        Returns:
            catalysts sampled for the target value.
        """
        if isinstance(target_energy, str):
            logger.warning(
                f"target energy ({target_energy}) passed as string, casting to float"
            )
            target_energy = float(target_energy)
        sampler = GaussianProcessRepresentationsSampler(
            {"energy": target_energy},
            property_predictors={"energy": self.predictor},
            representations={"smiles": self.vae},
            bounds={
                "smiles": (
                    self.minimum_latent_coordinate,
                    self.maximum_latent_coordinate,
                )
            },
        )
        return [
            sample["smiles"]
            for sample in sampler.optimize(
                number_of_points=self.number_of_points,
                number_of_steps=self.number_of_steps,
            )
        ]
