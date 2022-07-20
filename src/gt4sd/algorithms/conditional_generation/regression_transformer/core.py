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
"""RegressionTransformer algorithm.

RegressionTransformer is a mutlitask regression and conditional generation model.
"""

import logging
from dataclasses import field
from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, TypeVar, Union

from typing_extensions import Protocol, runtime_checkable

from ....domains.materials import Molecule, Sequence
from ....exceptions import InvalidItem
from ....properties.core import PropertyValue
from ...core import AlgorithmConfiguration, GeneratorAlgorithm
from ...registry import ApplicationsRegistry
from .implementation import ChemicalLanguageRT, ConditionalGenerator, ProteinLanguageRT

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = TypeVar("T", bound=Sequence)
S = TypeVar("S", PropertyValue, Molecule)
Targeted = Callable[[T], Iterable[Any]]


class RegressionTransformer(GeneratorAlgorithm[S, T]):
    """RegressionTransformer Algorithm."""

    #: The maximum number of samples a user can try to run in one go
    max_samples: int = 50

    def __init__(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ) -> None:
        """Instantiate Regression Transformer ready to generate items.

        Args:
            configuration: domain and application
                specification defining parameters, types and validations.
            target: a target for which to generate items.

        Example:
            An example for generating small molecules (SMILES) with high affinity for a target protein::

                config = RegressionTransformerProteins(
                    search='sample', temperature=2.0, tolerance=10
                )
                target = "<stab>0.393|GSQEVNSGT[MASK][MASK][MASK]YKNASPEEAE[MASK][MASK]IARKAGATTWTEKGNKWEIRI"
                stability_generator = RegressionTransformer(configuration=config, target=target)
                items = list(stability_generator.sample(10))
                print(items)
        """

        configuration = self.validate_configuration(configuration)

        # No validation/check on the target input here, since model is not yet loaded.
        super().__init__(
            configuration=configuration,  # type:ignore
            target=target,  # type:ignore
        )

    def get_generator(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ) -> Targeted[T]:
        """Get the function to sample with the given configuration.

        Args:
            configuration: helps to set up specific application of PaccMannRL.
            target: context or condition for the generation.

        Returns:
            callable with target generating a batch of items.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: ConditionalGenerator = configuration.get_conditional_generator(  # type: ignore
            resources_path=self.local_artifacts, context=target
        )
        if implementation.task == "regression" and configuration.search == "greedy":  # type: ignore
            self.max_samples = 1
            logger.warning(
                "max_samples was set to 1 due to regression task and greedy search"
            )

        return implementation.generate_batch  # type: ignore

    def validate_configuration(
        self, configuration: AlgorithmConfiguration[S, T]
    ) -> AlgorithmConfiguration[S, T]:
        @runtime_checkable
        class AnyRegressionTransformerConfiguration(Protocol):
            """Protocol for RegressionTransformer configurations."""

            def get_conditional_generator(
                self, resources_path: str
            ) -> ConditionalGenerator:
                ...

            def validate_item(self, item: Any) -> S:
                ...

        # TODO raise InvalidAlgorithmConfiguration
        assert isinstance(configuration, AnyRegressionTransformerConfiguration)
        assert isinstance(configuration, AlgorithmConfiguration)
        return configuration


@ApplicationsRegistry.register_algorithm_application(RegressionTransformer)
class RegressionTransformerMolecules(AlgorithmConfiguration[Sequence, Sequence]):
    """
    Configuration to generate molecules given a continuous property target and a molecular sub-structure.

    Implementation from the paper: https://arxiv.org/abs/2202.01338.

    Examples:
        An example for generating a peptide around a desired property value::

            config = RegressionTransformerMolecules(
                algorithm_version='solubility', search='sample', temperature=2, tolerance=5
            )
            target = "<esol>-3.534|[Br][C][=C][C][MASK][MASK][=C][C][=C][C][=C][Ring1][MASK][MASK][Branch2_3][Ring1][Branch1_2]"
            solubility_generator = RegressionTransformer(
                configuration=config, target=target
            )
            list(solubility_generator.sample(5))

        An example for predicting the solubility of a molecule::

            config = RegressionTransformerMolecules(
                algorithm_version='solubility', search='greedy'
            )
            target = "<esol>[MASK][MASK][MASK][MASK][MASK]|[Cl][C][Branch1_2][Branch1_2][=C][Branch1_1][C][Cl][Cl][Cl]"
            solubility_generator = RegressionTransformer(
                configuration=config, target=target
            )
            list(solubility_generator.sample(1))
    """

    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = field(
        default="solubility",
        metadata=dict(
            description="The version of the algorithm to use.",
            options=["solubility", "qed", "logp_and_synthesizability"],
        ),
    )

    search: str = field(
        default="sample",
        metadata=dict(
            description="Search algorithm to use for the generation: sample or greedy",
            options=["sample", "greedy"],
        ),
    )

    temperature: float = field(
        default=1.4,
        metadata=dict(
            description="Temperature parameter for the softmax sampling in decoding."
        ),
    )
    batch_size: int = field(
        default=8,
        metadata=dict(description="Batch size for the conditional generation"),
    )
    tolerance: float = field(
        default=20.0,
        metadata=dict(
            description="Precision tolerance for the conditional generation task. Given in percent"
        ),
    )
    sampling_wrapper: Dict = field(
        default_factory=dict,
        metadata=dict(
            description="""High-level entry point for SMILES-level access. Provide a
            dictionary that is used to build a custom sampling wrapper.
            NOTE: If this is used, the `target` needs to be a single SMILES string.
            Example: {
                'fraction_to_mask': 0.5,
                'atoms_to_mask': [],
                'property_goal': {'<qed>': 0.85}
            }
            - 'fraction_to_mask' specifies the ratio of tokens that can be changed by
                the model.
            - 'atoms_to_mask' specifies which atoms can be masked. This defaults
                to an empty list, meaning that all tokens can be masked.
            - 'property_goal' specifies the target conditions for the generation. The
                properties need to be specified as a dictionary. The keys need to be
                properties supported by the algorithm version.
            """
        ),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.

        Returns:
            target description.
        """
        return {
            "title": "Masked input sequence",
            "description": (
                "A sequence with a property value and a SELFIES string. Masking can either occur on the property or on the SELFIES, but not both."
                "For the scale of the property values, please see the task/dataset."
            ),
            "type": "string",
        }

    def get_conditional_generator(
        self, resources_path: str, context: str
    ) -> ChemicalLanguageRT:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.regression_transformer.implementation.ChemicalLanguageRT.generate_batch>` method for targeted generation.
        """
        self.generator = ChemicalLanguageRT(
            resources_path=resources_path,
            context=context,
            search=self.search,
            temperature=self.temperature,
            batch_size=self.batch_size,
            tolerance=self.tolerance,
            sampling_wrapper=self.sampling_wrapper,
        )
        return self.generator

    def validate_item(self, item: str) -> Union[Molecule, Sequence]:  # type: ignore
        """Check that item is a valid sequence.

        Args:
            item: a generated item that is possibly not valid.

        Raises:
            InvalidItem: in case the item can not be validated.

        Returns:
            the validated item.
        """
        if item is None:
            raise InvalidItem(title="InvalidSequence", detail="Sequence is None")
        (
            items,
            _,
        ) = self.generator.validate_output([item])
        if items[0] is None:
            if self.generator.task == "generation":
                title = "InvalidSMILES"
                detail = f'rdkit.Chem.MolFromSmiles returned None for "{item}"'
            else:
                title = "InvalidNumerical"
                detail = f'"{item}" is not a valid floating point number'
            raise InvalidItem(title=title, detail=detail)
        return item


@ApplicationsRegistry.register_algorithm_application(RegressionTransformer)
class RegressionTransformerProteins(AlgorithmConfiguration[Sequence, Sequence]):
    """
    Configuration to generate protein given a continuous property target and a partial AAs.

    Implementation from the paper: https://arxiv.org/abs/2202.01338. It can also predict the property given a full sequence.

    Examples:
        An example for generating a peptide around a desired property value::

            config = RegressionTransformerProteins(
                search='sample', temperature=2, tolerance=5
            )
            target = "<stab>1.1234|TTIKNG[MASK][MASK][MASK]YTVPLSPEQAAK[MASK][MASK][MASK]KKRWPDYEVQIHGNTVKVT"
            stability_generator = RegressionTransformer(
                configuration=config, target=target
            )
            list(stability_generator.sample(5))

        An example for predicting the stability of a peptide::

            config = RegressionTransformerProteins(search='greedy')
            target = "<stab>[MASK][MASK][MASK][MASK][MASK]|GSQEVNSNASPEEAEIARKAGATTWTEKGNKWEIRI"
            stability_generator = RegressionTransformer(
                configuration=config, target=target
            )
            list(stability_generator.sample(1))
    """

    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = field(
        default="stability",
        metadata=dict(
            description="The version of the algorithm to use.",
            options=["stability"],
        ),
    )

    search: str = field(
        default="sample",
        metadata=dict(
            description="Search algorithm to use for the generation: sample or greedy"
        ),
    )

    temperature: float = field(
        default=1.4,
        metadata=dict(
            description="Temperature parameter for the softmax sampling in decoding."
        ),
    )
    batch_size: int = field(
        default=32,
        metadata=dict(description="Batch size for the conditional generation"),
    )
    tolerance: float = field(
        default=20.0,
        metadata=dict(
            description="Precision tolerance for the conditional generation task. Given in percent"
        ),
    )
    sampling_wrapper: Dict = field(
        default_factory=dict,
        metadata=dict(
            description="""High-level entry point for SMILES-level access. Provide a
            dictionary that is used to build a custom sampling wrapper.
            NOTE: If this is used, the `target` needs to be a single SMILES string.
            Example: {
                'fraction_to_mask': 0.5,
                'atoms_to_mask': [],
                'property_goal': {'<qed>': 0.85}
            }
            - 'fraction_to_mask' specifies the ratio of tokens that can be changed by
                the model.
            - 'atoms_to_mask' specifies which atoms can be masked. This defaults
                to an empty list, meaning that all tokens can be masked.
            - 'property_goal' specifies the target conditions for the generation. The
                properties need to be specified as a dictionary. The keys need to be
                properties supported by the algorithm version.
            """
        ),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.

        Returns:
            target description.
        """
        return {
            "title": "Masked input sequence",
            "description": "A sequence with a property value and an AAS. Masking can either occur on the property or on the AAS, but not both.",
            "type": "string",
        }

    def get_conditional_generator(
        self, resources_path: str, context: str
    ) -> ProteinLanguageRT:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.
            context: input sequence to be used for the generation.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.regression_transformer.implementation.ProteinLanguageRT.generate_batch>` method for targeted generation.
        """

        self.generator = ProteinLanguageRT(
            resources_path=resources_path,
            search=self.search,
            temperature=self.temperature,
            context=context,
            batch_size=self.batch_size,
            tolerance=self.tolerance,
            sampling_wrapper=self.sampling_wrapper,
        )
        return self.generator

    def validate_item(self, item: str) -> Union[Molecule, Sequence]:  # type: ignore
        """Check that item is a valid sequence.

        Args:
            item: a generated item that is possibly not valid.

        Raises:
            InvalidItem: in case the item can not be validated.

        Returns:
            the validated item.
        """
        if item is None:
            raise InvalidItem(title="InvalidSequence", detail="Sequence is None")
        (
            items,
            _,
        ) = self.generator.validate_output([item])
        if items[0] is None:
            if self.generator.task == "generation":
                title = "InvalidSequence"
                detail = f'"{item}" does not adhere to IUPAC convention for AAS'
            else:
                title = "InvalidNumerical"
                detail = (
                    f'"{item}" is not a valid Sequence with a floating point number'
                )
            raise InvalidItem(title=title, detail=detail)
        return item
