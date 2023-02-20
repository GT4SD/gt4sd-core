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
"""PaccMannVAE Algorithm.

PaccMannVAE is an unconditional molecular generative model.
"""

import logging
from dataclasses import field
from typing import Any, ClassVar, Dict, Optional, Set, TypeVar

from ....domains.materials import SMILES
from ...conditional_generation.paccmann_rl.core import PaccMannRLProteinBasedGenerator
from ...core import AlgorithmConfiguration, GeneratorAlgorithm, Untargeted
from ...registry import ApplicationsRegistry
from .implementation import PaccMannVaeDefaultGenerator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = type(None)
S = TypeVar("S", bound=SMILES)


class PaccMannVAE(GeneratorAlgorithm[S, T]):
    """Molecular VAE as in the PaccMann\\ :superscript:`RL` paper."""

    def __init__(
        self, configuration: AlgorithmConfiguration[S, T], target: Optional[T] = None
    ):
        """Instantiate PaccMannVAE ready to generate molecules.

        Args:
            configuration: domain and application specification defining parameters,
                types and validations.
            target: unused since it is not a conditional generator.

        Example:
            An example for unconditional generation of small molecules::

                config = PaccMannVAEGenerator()
                algorithm = PaccMannVAE(configuration=config)
                items = list(algorithm.sample(10))
                print(items)
        """

        configuration = self.validate_configuration(configuration)
        super().__init__(configuration=configuration, target=None)  # type:ignore

    def get_generator(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ) -> Untargeted:
        """Get the function to sample batches via PaccMannVAE.

        Args:
            configuration: helps to set up specific application of PaccMannVAE.

        Returns:
            callable with target generating a batch of items.
        """
        implementation: PaccMannVaeDefaultGenerator = (
            configuration.get_conditional_generator()  # type: ignore
        )
        return implementation.generate


@ApplicationsRegistry.register_algorithm_application(PaccMannVAE)
class PaccMannVAEGenerator(AlgorithmConfiguration[SMILES, Any]):
    """
    Configuration to generate molecules with PaccMannVAE.

    Implementation from the paper: https://doi.org/10.1016/j.isci.2021.102269
    """

    algorithm_type: ClassVar[str] = "generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    batch_size: int = field(
        default=32,
        metadata=dict(description="Batch size used for the generative model sampling."),
    )
    temperature: float = field(
        default=1.4,
        metadata=dict(
            description="Temperature parameter for the softmax sampling in decoding."
        ),
    )
    generated_length: int = field(
        default=100,
        metadata=dict(
            description="Maximum length in tokens of the generated molcules (relates to the SMILES length)."
        ),
    )

    def get_target_description(self) -> Optional[Dict[str, str]]:
        """Get description of the target for generation.

        Returns:
            target description.
        """
        return None

    def get_conditional_generator(self) -> PaccMannVaeDefaultGenerator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.paccmann_rl.implementation.ConditionalGenerator.generate_batch>` method for targeted generation.
        """

        return PaccMannVaeDefaultGenerator(
            temperature=self.temperature,
            generated_length=self.generated_length,
            algorithm_version=self.algorithm_version,
            batch_size=self.batch_size,
        )

    @classmethod
    def list_versions(cls) -> Set[str]:
        """Get possible algorithm versions.

        S3 is searched as well as the local cache is searched for matching versions.

        Returns:
            viable values as :attr:`algorithm_version` for the environment.
        """

        return PaccMannRLProteinBasedGenerator().list_versions()
