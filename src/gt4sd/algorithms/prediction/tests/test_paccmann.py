"""TopicsZeroShot tests."""

from typing import ClassVar, Type

import pytest
import numpy as np
from gt4sd.algorithms.core import AlgorithmConfiguration
from gt4sd.algorithms.prediction.paccmann.core import (
    BimodalMCAAffinityPredictorConfiguation, Paccmann,
)
from gt4sd.algorithms.registry import ApplicationsRegistry


def get_classvar_type(class_var):
    """Extract type from ClassVar type annotation: `ClassVar[T]] -> T`."""
    return class_var.__args__[0]


# @pytest.mark.parametrize(
#     "config_class, algorithm_type, domain, algorithm_name",
#     [
#         (
#             BimodalMCAAffinityPredictorConfiguation,
#             "prediction",
#             "material",
#             Paccmann.__name__,
#         )
#     ],
# )
# def test_config_class(
#     config_class: Type[AlgorithmConfiguration],
#     algorithm_type: str,
#     domain: str,
#     algorithm_name: str,
# ):
#     assert config_class.algorithm_type == algorithm_type
#     assert config_class.domain == domain
#     assert config_class.algorithm_name == algorithm_name

#     for keyword, type_annotation in config_class.__annotations__.items():
#         if keyword in ("algorithm_type", "domain", "algorithm_name"):
#             assert type_annotation.__origin__ is ClassVar  # type: ignore
#             assert str == get_classvar_type(type_annotation)


# @pytest.mark.parametrize(
#     "config_class",
#     [(BimodalMCAAffinityPredictorConfiguation)],
# )
# def test_config_instance(config_class: Type[AlgorithmConfiguration]):
#     config = config_class()  # type:ignore
#     assert config.algorithm_application == config_class.__name__





@pytest.mark.parametrize(
    "config, target, ligand_smiles, confidence, algorithm",
    [
        (
            BimodalMCAAffinityPredictorConfiguation,
            [
                'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT',
                'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT',
            ],
            [
                'CONN=COc1cc2ccccc2c11Occncc(Cl)c1N(O)O',
                'ClCCC(O1)C(C(N=C1C(=O)Nc1cccc(F)c1F)SO)C',
            ],
            True,
            Paccmann,
        )
    ],
)
def test_generation_via_import(config, target, ligand_smiles, confidence, algorithm):
    algorithm = algorithm(configuration=config())
    preds, preds_extra = algorithm.generator(
        target=target,
        ligand_smiles=ligand_smiles,
        confidence=confidence,
    )
    assert np.isclose(preds[0,0].item(), 0.5492, atol=1e-4)
    assert np.isclose(preds[1,0].item(), 0.4799, atol=1e-4)
    


# @pytest.mark.parametrize(
#     "algorithm_application, target, algorithm_type, domain, algorithm_name",
#     [
#         (
#             BimodalMCAAffinityPredictorConfiguation.__name__,
#             "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration.",
#             "prediction",
#             "nlp",
#             Paccmann.__name__,
#         ),
#     ],
# )

# def test_generation_via_registry(
#     algorithm_type, target, domain, algorithm_name, algorithm_application
# ):
#     algorithm = ApplicationsRegistry.get_application_instance(
#         target=target,
#         algorithm_type=algorithm_type,
#         domain=domain,
#         algorithm_name=algorithm_name,
#         algorithm_application=algorithm_application,
#     )
#     items = list(algorithm.sample(5))
#     assert len(items) == 5
