from dataclasses import field
from typing import Any, Callable


from ...domains.materials import Property, SmallMolecule, MacroMolecule, Protein
from .utils import to_mol, to_smiles
from typing import Dict, Tuple, Type, Union

from .functions import (
    activity_against_target,
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
    scscore,
    similarity_to_seed,
    tpsa,
)

from dataclasses import field

# CALLABLE_FUNCTIONS_FACTORY = {}

from ..core import PropertyPredictor, PropertyPredictorConfiguration, CallablePropertyPredictor

# class CallableFactory:

#     @staticmethod
#     def get(callable: Callable) -> PropertyPredictor:
#         return Callable

class PlogpParameters(PropertyPredictorConfiguration):
    a_parameter: int = field(
        default=0,
        metadata=dict(description=""),
    )

class LipinskiParameters(PropertyPredictorConfiguration):
    a_parameter: int = field(
        default=0,
        metadata=dict(description=""),
    )

class EsolParameters(PropertyPredictorConfiguration):
    a_parameter: int = field(
        default=0,
        metadata=dict(description=""),
    )

class ScscoreParameters(PropertyPredictorConfiguration):
    score_scale: int = field(
        default=5,
        metadata=dict(description=""),
    )

    fp_len: int = field(
        default=1024,
        metadata=dict(description=""),
    )

    fp_rad: int = field(
        default=2,
        metadata=dict(description=""),
    )

class SasParameters(PropertyPredictorConfiguration):
    a_parameter: int = field(
        default=0,
        metadata=dict(description=""),
    )

class BertzParameters(PropertyPredictorConfiguration):
    a_parameter: int = field(
        default=0,
        metadata=dict(description=""),
    )

class TpsaParameters(PropertyPredictorConfiguration):
    a_parameter: int = field(
        default=0,
        metadata=dict(description=""),
    )

class LogpParameters(PropertyPredictorConfiguration):
    a_parameter: int = field(
        default=0,
        metadata=dict(description=""),
    )

class QedParameters(PropertyPredictorConfiguration):
    a_parameter: int = field(
        default=0,
        metadata=dict(description=""),
    )

class NumberHAcceptorsParameters(PropertyPredictorConfiguration):
    pass

class NumberAtomsParameters(PropertyPredictorConfiguration):
    pass

class NumberHDonorsParameters(PropertyPredictorConfiguration):
    pass

class NumberAromaticRingsParameters(PropertyPredictorConfiguration):
    pass

class NumberRingsParameters(PropertyPredictorConfiguration):
    pass

class NumberRotatableBondsParameters(PropertyPredictorConfiguration):
    pass

class NumberLargeRingsParameters(PropertyPredictorConfiguration):
    pass

class MolecularWeightParameters(PropertyPredictorConfiguration):
    pass

class IsScaffoldParameters(PropertyPredictorConfiguration):
    pass

class NumberHeterocyclesParameters(PropertyPredictorConfiguration):
    pass

class NumberStereocentersParameters(PropertyPredictorConfiguration):
    pass

class SimilaritySeedParameters(PropertyPredictorConfiguration):
    pass

class ActivityAgainstTargetParameters(PropertyPredictorConfiguration):
    pass


#__________________________________________________________________________________________________
class Plogp(CallablePropertyPredictor):
    """Calculate the penalized logP of a molecule. This is the logP minus the number of
    rings with > 6 atoms minus the SAS.
    """

    def __init__(self, parameters: PlogpParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=plogp)


class Lipinski(CallablePropertyPredictor):
    """Calculate whether a molecule adheres to the Lipinski-rule-of-5.
    A crude approximation of druglikeness.
    """

    def __init__(self, parameters: LipinskiParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=lipinski)

class Esol(CallablePropertyPredictor):
    """Estimate the water solubility of a molecule.
    """

    def __init__(self, parameters: EsolParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=esol)


class Scscore(CallablePropertyPredictor):
    """Calculate the synthetic complexity score (SCScore) of a molecule.
    """

    def __init__(self, parameters: ScscoreParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=scscore)


class Sas(CallablePropertyPredictor):
    """Calculate the synthetic accessibility score (SAS) for a molecule.
    """

    def __init__(self, parameters: SasParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=sas)

class Bertz(CallablePropertyPredictor):
    """Calculate Bertz index of a molecule.
    """

    def __init__(self, parameters: BertzParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=bertz)


class Tpsa(CallablePropertyPredictor):
    """Calculate the total polar surface area of a molecule.
    """

    def __init__(self, parameters: TpsaParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=tpsa)


class Logp(CallablePropertyPredictor):
    """Calculates the partition coefficient of a molecule.
    """

    def __init__(self, parameters: LogpParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=logp)

class Qed(CallablePropertyPredictor):
    """Calculate the quantitative estimate of drug-likeness (QED) of a molecule.
    """

    def __init__(self, parameters: QedParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=qed)

class NumberHAcceptors(CallablePropertyPredictor):
    """Calculate number of H acceptors of a molecule.
    """

    def __init__(self, parameters: NumberHAcceptorsParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=number_of_h_acceptors)

class NumberAtoms(CallablePropertyPredictor):
    """Calculate number of atoms of a molecule.
    """

    def __init__(self, parameters: NumberAtomsParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=number_of_atoms)

class NumberHDonors(CallablePropertyPredictor):
    """Calculate number of H donors of a molecule.
    """

    def __init__(self, parameters: NumberHDonorsParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=number_of_h_donors)

class NumberAromaticRings(CallablePropertyPredictor):
    """Calculate number of aromatic rings of a molecule.
    """

    def __init__(self, parameters: NumberAromaticRingsParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=number_of_aromatic_rings)


class NumberRings(CallablePropertyPredictor):
    """Calculate number of rings of a molecule.
    """

    def __init__(self, parameters: NumberRingsParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=number_of_rings)


class NumberRotatableBonds(CallablePropertyPredictor):
    """Calculate number of rotatable bonds of a molecule.
    """

    def __init__(self, parameters: NumberRotatableBondsParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=number_of_rotatable_bonds)


class NumberLargeRings(CallablePropertyPredictor):
    """Calculate the amount of large rings (> 6 atoms) of a molecule.
    """

    def __init__(self, parameters: NumberLargeRingsParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=number_of_large_rings)


class MolecularWeight(CallablePropertyPredictor):
    """Calculate molecular weight of a molecule.
    """

    def __init__(self, parameters: MolecularWeightParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=molecular_weight)


class IsScaffold(CallablePropertyPredictor):
    """Whether a molecule is identical to its Murcko Scaffold.
    """

    def __init__(self, parameters: IsScaffoldParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=is_scaffold)


class NumberHeterocycles(CallablePropertyPredictor):
    """The amount of heterocycles of a molecule.
    """

    def __init__(self, parameters: NumberHeterocyclesParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=number_of_heterocycles)


class NumberStereocenters(CallablePropertyPredictor):
    """The amount of stereo centers of a molecule.
    """

    def __init__(self, parameters: NumberStereocentersParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=number_of_stereocenters)


class SimilaritySeed(CallablePropertyPredictor):
    """Calculate the similarity of a molecule to a seed molecule.
    """

    def __init__(self, parameters: SimilaritySeedParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=similarity_to_seed)


class ActivityAgainstTarget(CallablePropertyPredictor):
    """Calculate the activity of a molecule against a target molecule.
    """

    def __init__(self, parameters: ActivityAgainstTargetParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=activity_against_target)


