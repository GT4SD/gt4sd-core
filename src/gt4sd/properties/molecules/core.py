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
from paccmann_generator.drug_evaluators import SCScore
from pydantic import Field

from ..core import CallablePropertyPredictor, PropertyPredictorParameters
from ..utils import get_activity_fn, get_similarity_fn
from .functions import (
    bertz,
    esol,
    is_scaffold,
    lipinski,
    logp,
    molecular_weight,
    number_of_aromatic_rings,
    number_of_atoms,
    number_of_h_acceptors,
    number_of_h_donors,
    number_of_heterocycles,
    number_of_large_rings,
    number_of_rings,
    number_of_rotatable_bonds,
    number_of_stereocenters,
    plogp,
    qed,
    sas,
    tpsa,
)


# NOTE: property prediction parameters
class ScscoreConfiguration(PropertyPredictorParameters):
    score_scale: int = 5
    fp_len: int = 1024
    fp_rad: int = 2


class SimilaritySeedParameters(PropertyPredictorParameters):
    smiles: str = Field(..., example="c1ccccc1")
    fp_key: str = "ECFP4"


class ActivityAgainstTargetParameters(PropertyPredictorParameters):
    target: str = Field(..., example="drd2", description="name of the target.")


# NOTE: property prediction classes
class Plogp(CallablePropertyPredictor):
    """Calculate the penalized logP of a molecule. This is the logP minus the number of
    rings with > 6 atoms minus the SAS.
    """

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=plogp, parameters=parameters)


class Lipinski(CallablePropertyPredictor):
    """Calculate whether a molecule adheres to the Lipinski-rule-of-5.
    A crude approximation of druglikeness.
    """

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=lipinski, parameters=parameters)


class Esol(CallablePropertyPredictor):
    """Estimate the water solubility of a molecule."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=esol, parameters=parameters)


class Scscore(CallablePropertyPredictor):
    """Calculate the synthetic complexity score (SCScore) of a molecule."""

    def __init__(
        self, parameters: ScscoreConfiguration = ScscoreConfiguration()
    ) -> None:
        super().__init__(
            callable_fn=SCScore(**parameters.dict()), parameters=parameters
        )


class Sas(CallablePropertyPredictor):
    """Calculate the synthetic accessibility score (SAS) for a molecule."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=sas, parameters=parameters)


class Bertz(CallablePropertyPredictor):
    """Calculate Bertz index of a molecule."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=bertz, parameters=parameters)


class Tpsa(CallablePropertyPredictor):
    """Calculate the total polar surface area of a molecule."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=tpsa, parameters=parameters)


class Logp(CallablePropertyPredictor):
    """Calculates the partition coefficient of a molecule."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=logp, parameters=parameters)


class Qed(CallablePropertyPredictor):
    """Calculate the quantitative estimate of drug-likeness (QED) of a molecule."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=qed, parameters=parameters)


class NumberHAcceptors(CallablePropertyPredictor):
    """Calculate number of H acceptors of a molecule."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=number_of_h_acceptors, parameters=parameters)


class NumberAtoms(CallablePropertyPredictor):
    """Calculate number of atoms of a molecule."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=number_of_atoms, parameters=parameters)


class NumberHDonors(CallablePropertyPredictor):
    """Calculate number of H donors of a molecule."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=number_of_h_donors, parameters=parameters)


class NumberAromaticRings(CallablePropertyPredictor):
    """Calculate number of aromatic rings of a molecule."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=number_of_aromatic_rings, parameters=parameters)


class NumberRings(CallablePropertyPredictor):
    """Calculate number of rings of a molecule."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=number_of_rings, parameters=parameters)


class NumberRotatableBonds(CallablePropertyPredictor):
    """Calculate number of rotatable bonds of a molecule."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=number_of_rotatable_bonds, parameters=parameters)


class NumberLargeRings(CallablePropertyPredictor):
    """Calculate the amount of large rings (> 6 atoms) of a molecule."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=number_of_large_rings, parameters=parameters)


class MolecularWeight(CallablePropertyPredictor):
    """Calculate molecular weight of a molecule."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=molecular_weight, parameters=parameters)


class IsScaffold(CallablePropertyPredictor):
    """Whether a molecule is identical to its Murcko Scaffold."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=is_scaffold, parameters=parameters)


class NumberHeterocycles(CallablePropertyPredictor):
    """The amount of heterocycles of a molecule."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=number_of_heterocycles, parameters=parameters)


class NumberStereocenters(CallablePropertyPredictor):
    """The amount of stereo centers of a molecule."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        super().__init__(callable_fn=number_of_stereocenters, parameters=parameters)


class SimilaritySeed(CallablePropertyPredictor):
    """Calculate the similarity of a molecule to a seed molecule."""

    def __init__(self, parameters: SimilaritySeedParameters) -> None:
        super().__init__(
            callable_fn=get_similarity_fn(
                target_mol=parameters.smiles, fp_key=parameters.fp_key
            ),
            parameters=parameters,
        )


class ActivityAgainstTarget(CallablePropertyPredictor):
    """Calculate the activity of a molecule against a target molecule."""

    def __init__(self, parameters: ActivityAgainstTargetParameters) -> None:
        super().__init__(
            callable_fn=get_activity_fn(target=parameters.target), parameters=parameters
        )
