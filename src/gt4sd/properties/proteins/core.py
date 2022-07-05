from dataclasses import field

from ..core import CallablePropertyPredictor, PropertyPredictorConfiguration
from .functions import (
    aliphatic_index,
    aromaticity,
    boman_index,
    charge,
    charge_density,
    hydrophobic_ratio,
    instability,
    isoelectric_point,
    length,
)

# class CallableFactory:

#     @staticmethod
#     def get(callable: Callable) -> PropertyPredictor:
#         return Callable


class LengthParameters(PropertyPredictorConfiguration):
    pass


class BomanIndexParameters(PropertyPredictorConfiguration):
    pass


class AliphaticIndexParameters(PropertyPredictorConfiguration):
    pass


class HydrophobicRatioParameters(PropertyPredictorConfiguration):
    pass


class ChargeParameters(PropertyPredictorConfiguration):
    ph: float = field(
        default=7.4,
        metadata=dict(description=""),
    )

    amide: bool = field(
        default=True,
        metadata=dict(description=""),
    )


class ChargeDensityParameters(PropertyPredictorConfiguration):
    ph: float = field(
        default=7.4,
        metadata=dict(description=""),
    )

    amide: bool = field(
        default=True,
        metadata=dict(description=""),
    )


class IsoelectricPointParameters(PropertyPredictorConfiguration):
    amide: bool = field(
        default=True,
        metadata=dict(description=""),
    )


class AromaticityParameters(PropertyPredictorConfiguration):
    pass


class InstabilityParameters(PropertyPredictorConfiguration):
    pass


class Length(CallablePropertyPredictor):
    """Retrieves the number of residues of a protein."""

    def __init__(self, parameters: LengthParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=length)


class BomanIndex(CallablePropertyPredictor):
    """Computes the Boman index of a protein (sum of solubility values of all residues)."""

    def __init__(self, parameters: BomanIndexParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=boman_index)


class AliphaticIndex(CallablePropertyPredictor):
    """Computes the aliphatic index of a protein. Measure of thermal stability."""

    def __init__(self, parameters: AliphaticIndexParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=aliphatic_index)


class HydrophobicRatio(CallablePropertyPredictor):
    """Computes the hydrophobicity of a protein, relative freq. of **A,C,F,I,L,M & V**."""

    def __init__(self, parameters: HydrophobicRatioParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=hydrophobic_ratio)


class Charge(CallablePropertyPredictor):
    """Computes the charge of a protein."""

    def __init__(self, parameters: ChargeParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=charge)


class ChargeDensity(CallablePropertyPredictor):
    """Computes the charge density of a protein."""

    def __init__(self, parameters: ChargeDensityParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=charge_density)


class IsoelectricPoint(CallablePropertyPredictor):
    """Computes the isoelectric point of every residue and aggregates."""

    def __init__(self, parameters: IsoelectricPointParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=isoelectric_point)


class Aromaticity(CallablePropertyPredictor):
    """Computes aromaticity of the protein (relative frequency of Phe+Trp+Tyr)."""

    def __init__(self, parameters: AromaticityParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=aromaticity)


class Instability(CallablePropertyPredictor):
    """Calculates the protein instability."""

    def __init__(self, parameters: InstabilityParameters) -> None:
        super().__init__(parameters=parameters, callable_fn=instability)
