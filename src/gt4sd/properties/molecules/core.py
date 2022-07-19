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

from ..core import (
    CallableConfigurableProperty,
    CallableProperty,
    PropertyConfiguration,
    SmallMolecule,
)
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


# Parameter classes
class ScscoreConfiguration(PropertyConfiguration):
    score_scale: int = 5
    fp_len: int = 1024
    fp_rad: int = 2


class SimilaritySeedParameters(PropertyConfiguration):
    seed: SmallMolecule
    fp_key: str = "ECFP4"

    class Config:
        arbitrary_types_allowed = True


class ActivityAgainstTargetParameters(PropertyConfiguration):
    target: str


# Property classes
class Plogp(CallableConfigurableProperty):
    """Calculate the penalized logP of a molecule. This is the logP minus the number of
    rings with > 6 atoms minus the SAS.
    """

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=plogp, parameters=parameters)


class Lipinski(CallableConfigurableProperty):
    """Calculate whether a molecule adheres to the Lipinski-rule-of-5.
    A crude approximation of druglikeness.
    """

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=lipinski, parameters=parameters)


class Esol(CallableConfigurableProperty):
    """Estimate the water solubility of a molecule."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=esol, parameters=parameters)


class Scscore(CallableProperty):
    """Calculate the synthetic complexity score (SCScore) of a molecule."""

    def __init__(
        self, parameters: ScscoreConfiguration = ScscoreConfiguration()
    ) -> None:
        super().__init__(
            callable_fn=SCScore(**parameters.dict()), parameters=parameters
        )


class Sas(CallableConfigurableProperty):
    """Calculate the synthetic accessibility score (SAS) for a molecule."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=sas, parameters=parameters)


class Bertz(CallableConfigurableProperty):
    """Calculate Bertz index of a molecule."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=bertz, parameters=parameters)


class Tpsa(CallableConfigurableProperty):
    """Calculate the total polar surface area of a molecule."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=tpsa, parameters=parameters)


class Logp(CallableConfigurableProperty):
    """Calculates the partition coefficient of a molecule."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=logp, parameters=parameters)


class Qed(CallableConfigurableProperty):
    """Calculate the quantitative estimate of drug-likeness (QED) of a molecule."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=qed, parameters=parameters)


class NumberHAcceptors(CallableConfigurableProperty):
    """Calculate number of H acceptors of a molecule."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=number_of_h_acceptors, parameters=parameters)


class NumberAtoms(CallableConfigurableProperty):
    """Calculate number of atoms of a molecule."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=number_of_atoms, parameters=parameters)


class NumberHDonors(CallableConfigurableProperty):
    """Calculate number of H donors of a molecule."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=number_of_h_donors, parameters=parameters)


class NumberAromaticRings(CallableConfigurableProperty):
    """Calculate number of aromatic rings of a molecule."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=number_of_aromatic_rings, parameters=parameters)


class NumberRings(CallableConfigurableProperty):
    """Calculate number of rings of a molecule."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=number_of_rings, parameters=parameters)


class NumberRotatableBonds(CallableConfigurableProperty):
    """Calculate number of rotatable bonds of a molecule."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=number_of_rotatable_bonds, parameters=parameters)


class NumberLargeRings(CallableConfigurableProperty):
    """Calculate the amount of large rings (> 6 atoms) of a molecule."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=number_of_large_rings, parameters=parameters)


class MolecularWeight(CallableConfigurableProperty):
    """Calculate molecular weight of a molecule."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=molecular_weight, parameters=parameters)


class IsScaffold(CallableConfigurableProperty):
    """Whether a molecule is identical to its Murcko Scaffold."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=is_scaffold, parameters=parameters)


class NumberHeterocycles(CallableConfigurableProperty):
    """The amount of heterocycles of a molecule."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=number_of_heterocycles, parameters=parameters)


class NumberStereocenters(CallableConfigurableProperty):
    """The amount of stereo centers of a molecule."""

    def __init__(
        self, parameters: PropertyConfiguration = PropertyConfiguration()
    ) -> None:
        super().__init__(callable_fn=number_of_stereocenters, parameters=parameters)


class SimilaritySeed(CallableProperty):
    """Calculate the similarity of a molecule to a seed molecule."""

    def __init__(self, parameters: SimilaritySeedParameters) -> None:
        super().__init__(
            callable_fn=get_similarity_fn(
                target_mol=parameters.seed, fp_key=parameters.fp_key
            ),
            parameters=parameters,
        )


class ActivityAgainstTarget(CallableProperty):
    """Calculate the activity of a molecule against a target molecule."""

    def __init__(self, parameters: ActivityAgainstTargetParameters) -> None:
        super().__init__(
            callable_fn=get_activity_fn(target=parameters.target), parameters=parameters
        )
