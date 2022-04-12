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
"""CLaSS Algorithm: PAG and CogMol applications."""

import logging
from dataclasses import field
from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, TypeVar, Union

from ....extras import EXTRAS_ENABLED
from ...core import AlgorithmConfiguration, GeneratorAlgorithm  # type: ignore
from ...registry import ApplicationsRegistry  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if EXTRAS_ENABLED:
    from cog.core import CogMolGenerator
    from cog.sample_pipeline import CogMolFiles, read_artifacts_config
    from pag.core import PAGGenerator
    from pag.sample_pipeline import PAGFiles

    T = TypeVar("T")
    S = TypeVar("S")
    Targeted = Callable[[T], Iterable[Any]]
    Untargeted = Callable[[], Iterable[Any]]

    class CLaSS(GeneratorAlgorithm[S, T]):
        """Controlled Latent attribute Space Sampling (CLaSS) Algorithm."""

        def __init__(
            self,
            configuration: AlgorithmConfiguration[S, T],
            target: Optional[T] = None,
        ):
            """Instantiate CLaSS ready to generate items.

            Args:
                configuration: domain and application
                    specification, defining types and validations.
                target: Optional, in this inistance we will convert to a string.

            Example:
                An example for using the CogMol application with this Algorithm::

                    # target protein
                    MPRO = "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ"
                    configuration = CogMol()
                    algorithm = CLaSS(configuration=configuration, target=MPRO)
                    items = list(algorithm.sample(1))
                    print(items)

                We can also use the PAG application similarly::

                    configuration = PAG()
                    algorithm = CLaSS(configuration=configuration)
                    items = list(algorithm.sample(1))
                    print(items)
            """

            configuration = self.validate_configuration(configuration)
            # TODO there might also be a validation/check on the target input

            super().__init__(
                configuration=configuration,
                target=target,  # type:ignore
            )

        def get_generator(
            self,
            configuration: AlgorithmConfiguration[S, T],
            target: Optional[T],
        ) -> Union[Untargeted, Targeted[T]]:
            """Get the function to sample from generator.

            Args:
                configuration: helps to set up the application.
                target: target to generate molecules against.

            Returns:
                callable generating a list of molecules.
            """
            logger.info("ensure artifacts for the application are present.")
            self.local_artifacts = configuration.ensure_artifacts()
            implementation = configuration.get_class_instance(  # type: ignore
                resources_path=self.local_artifacts, target=target
            )
            return implementation.sample_accepted

        def validate_configuration(
            self, configuration: AlgorithmConfiguration
        ) -> AlgorithmConfiguration:
            # TODO raise InvalidAlgorithmConfiguration
            assert isinstance(configuration, AlgorithmConfiguration)
            return configuration

    @ApplicationsRegistry.register_algorithm_application(CLaSS)
    class CogMol(AlgorithmConfiguration[str, str]):
        """Configuration for CogMol: Target-Specific and Selective Drug Design."""

        algorithm_type: ClassVar[str] = "controlled_sampling"
        domain: ClassVar[str] = "materials"
        algorithm_version: str = "v0"

        samples_per_round: int = field(
            default=200,
            metadata=dict(
                description="Number of generated samples for acceptance/rejection per round."
            ),
        )
        max_length: int = field(
            default=100,
            metadata=dict(description="Maximal number of tokens in generated samples."),
        )
        temperature: float = field(
            default=1.0,
            metadata=dict(description="Temperature of softmax."),
        )
        num_proteins_selectivity: int = field(
            default=10,
            metadata=dict(
                description="Number of random samples for measuring off target selectivity for rejection."
            ),
        )

        def get_target_description(self) -> Dict[str, str]:
            """Get description of the target for generation.
            Returns:
                target description.
            """
            return {
                "title": "Protein",
                "description": "Primary structure of the target protein as sequence of amino acid characters.",
                "type": "string",
            }

        def get_class_instance(self, resources_path: str, target: str):
            try:
                config = read_artifacts_config(resources_path)
                bindingdb_date = config["cogmol version information"]["bindingdb_date"]
            except KeyError:
                bindingdb_date = None

            return CogMolGenerator(
                protein_sequence=target,
                model_files=CogMolFiles.from_directory_with_config(resources_path),
                n_samples_per_round=self.samples_per_round,
                device="cpu",
                num_proteins_selectivity=self.num_proteins_selectivity,
                temp=self.temperature,
                max_len=self.max_length,
                bindingdb_date=bindingdb_date,
            )

    @ApplicationsRegistry.register_algorithm_application(CLaSS)
    class PAG(AlgorithmConfiguration[str, str]):
        """Configuration for photoacid generator (PAG) design."""

        algorithm_type: ClassVar[str] = "controlled_sampling"
        domain: ClassVar[str] = "materials"
        algorithm_version: str = "v0"

        samples_per_round: int = field(
            default=200,
            metadata=dict(
                description="Number of generated samples for acceptance/rejection per round."
            ),
        )
        max_length: int = field(
            default=100,
            metadata=dict(description="Maximal number of tokens in generated samples."),
        )
        temperature: float = field(
            default=1.0,
            metadata=dict(description="Temperature of softmax."),
        )

        def get_target_description(self) -> None:
            """Untargeted sampling. Always returns None.

            Returns:
                None
            """
            return None

        def get_class_instance(self, resources_path: str, target: Optional[T] = None):
            if target is not None:
                raise NotImplementedError

            return PAGGenerator(
                model_files=PAGFiles.from_directory_with_config(resources_path),
                n_samples_per_round=self.samples_per_round,
                device="cpu",
                temp=self.temperature,
                max_len=self.max_length,
            )

else:
    logger.warning("install cogmol-inference extras to use CLaSS")
