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
from enum import Enum
from typing import List

from paccmann_generator.drug_evaluators import SIDER
from paccmann_generator.drug_evaluators import ClinTox as _ClinTox
from paccmann_generator.drug_evaluators import OrganDB as _OrganTox
from paccmann_generator.drug_evaluators import SCScore
from paccmann_generator.drug_evaluators import Tox21 as _Tox21
from pydantic import Field
from tdc import Oracle
from tdc.metadata import download_receptor_oracle_name

from ...algorithms.core import (
    ConfigurablePropertyAlgorithmConfiguration,
    Predictor,
    PredictorAlgorithm,
)
from ...domains.materials import SmallMolecule
from ..core import (
    ApiTokenParameters,
    CallablePropertyPredictor,
    ConfigurableCallablePropertyPredictor,
    DomainSubmodule,
    IpAdressParameters,
    PropertyPredictorParameters,
    PropertyValue,
    S3Parameters,
)
from ..utils import (
    docking_import_check,
    get_activity_fn,
    get_similarity_fn,
    to_smiles,
    validate_api_token,
    validate_ip,
)
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


class AskcosParameters(IpAdressParameters):
    class Output(str, Enum):
        plausability: str = "plausibility"
        num_step: str = "num_step"
        synthesizability: str = "synthesizability"
        price: str = "price"

    output: Output = Field(
        default=Output.plausability,
        example=Output.synthesizability,
        description="Main output return type from ASKCOS",
        options=["plausibility", "num_step", "synthesizability", "price"],
    )
    save_json: bool = Field(default=False)
    file_name: str = Field(default="tree_builder_result.json")
    num_trials: int = Field(default=5)
    max_depth: int = Field(default=9)
    max_branching: int = Field(default=25)
    expansion_time: int = Field(default=60)
    max_ppg: int = Field(default=100)
    template_count: int = Field(default=1000)
    max_cum_prob: float = Field(default=0.999)
    chemical_property_logic: str = Field(default="none")
    max_chemprop_c: int = Field(default=0)
    max_chemprop_n: int = Field(default=0)
    max_chemprop_o: int = Field(default=0)
    max_chemprop_h: int = Field(default=0)
    chemical_popularity_logic: str = Field(default="none")
    min_chempop_reactants: int = Field(default=5)
    min_chempop_products: int = Field(default=5)
    filter_threshold: float = Field(default=0.1)
    return_first: str = Field(default="true")

    # Convert enum items back to strings
    class Config:
        use_enum_values = True


class MoleculeOneParameters(ApiTokenParameters):

    oracle_name: str = "Molecule One Synthesis"


class DockingTdcParameters(PropertyPredictorParameters):
    # To dock against a receptor defined via TDC
    target: str = Field(
        ...,
        example="1iep_docking",
        description="Target for docking, provided via TDC",
        options=download_receptor_oracle_name,
    )


class DockingParameters(PropertyPredictorParameters):
    # To dock against a user-provided receptor
    name: str = Field(default="pyscreener")
    receptor_pdb_file: str = Field(
        example="/tmp/2hbs.pdb", description="Path to receptor PDB file"
    )
    box_center: List[int] = Field(
        example=[15.190, 53.903, 16.917], description="Docking box center"
    )
    box_size: List[float] = Field(example=[20, 20, 20], description="Docking box size")


class S3ParametersMolecules(S3Parameters):
    domain: DomainSubmodule = DomainSubmodule("molecules")


class MCAParameters(S3ParametersMolecules):
    algorithm_name: str = "MCA"


class Tox21Parameters(MCAParameters):
    algorithm_application: str = "Tox21"


class ClinToxParameters(MCAParameters):
    algorithm_application: str = "ClinTox"


class SiderParameters(MCAParameters):
    algorithm_application: str = "SIDER"


class OrganToxParameters(MCAParameters):
    class Organs(str, Enum):
        adrenal_gland: str = "Adrenal Gland"
        bone_marrow: str = "Bone Marrow"
        brain: str = "Brain"
        eye: str = "Eye"
        heart: str = "Heart"
        kidney: str = "Kidney"
        liver: str = "Liver"
        lung: str = "Lung"
        lymph_node: str = "Lymph Node"
        mammary_gland: str = "Mammary Gland"
        ovary: str = "Ovary"
        pancreas: str = "Pancreas"
        pituitary_gland: str = "Pituitary Gland"
        spleen: str = "Spleen"
        stomach: str = "Stomach"
        testes: str = "Testes"
        thymus: str = "Thymus"
        thyroid_gland: str = "Thyroid Gland"
        urinary_bladder: str = "Urinary Bladder"
        uterus: str = "Uterus"

    class ToxType(str, Enum):
        chronic: str = "chronic"
        subchronic: str = "subchronic"
        multigenerational: str = "multigenerational"
        all: str = "all"

    algorithm_application: str = "OrganTox"
    site: Organs = Field(
        ...,
        example=Organs.kidney,
        description="name of the target site of interest.",
    )
    toxicity_type: ToxType = Field(
        default=ToxType.all,
        example=ToxType.chronic,
        description="type of toxicity for which predictions are made.",
        options=["chronic", "subchronic", "multigenerational", "all"],
    )


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


class Askcos(ConfigurableCallablePropertyPredictor):
    """
    A property predictor that uses the ASKCOs API to calculate the synthesizability
    of a molecule.
    """

    def __init__(self, parameters: AskcosParameters):

        # Raises if IP is not valid
        msg = (
            "You have to point to an IP address of a running ASKCOS instance. "
            "For details on setting this up, see: https://tdcommons.ai/functions/oracles/#askcos"
        )
        if not isinstance(parameters.host_ip, str):
            raise TypeError(f"IP adress must be a string, not {parameters.host_ip}")

        if not hasattr(parameters, "host_ip"):
            raise AttributeError(f"IP adress missing in {parameters}")

        if "http" not in parameters.host_ip:
            raise ValueError(
                f"ASKCOS requires an IP prepended with a http, e.g., "
                f"'http://xx.xx.xxx.xxx' and not {parameters.host_ip}."
            )
        ip = parameters.host_ip.split("//")[1]

        validate_ip(ip, message=msg)
        super().__init__(callable_fn=Oracle(name="ASKCOS"), parameters=parameters)


class MoleculeOne(CallablePropertyPredictor):
    """
    A property predictor that uses the MoleculeOne API to calculate the synthesizability
    of a molecule.
    """

    def __init__(self, parameters: MoleculeOneParameters):

        msg = (
            "You have to provide a valid API key, for details on setting this up, see: "
            "https://tdcommons.ai/functions/oracles/#moleculeone"
        )

        # Only performs type checking on API key
        validate_api_token(parameters, message=msg)

        super().__init__(
            callable_fn=Oracle(
                name=parameters.oracle_name, api_token=parameters.api_token
            ),
            parameters=parameters,
        )


class DockingTdc(ConfigurableCallablePropertyPredictor):
    """
    A property predictor that computes the docking score against a target
    provided via the TDC package (see: https://tdcommons.ai/functions/oracles/#docking-scores)
    """

    def __init__(self, parameters: DockingTdcParameters):

        docking_import_check()
        callable = Oracle(name=parameters.target)
        super().__init__(callable_fn=callable, parameters=parameters)


class Docking(ConfigurableCallablePropertyPredictor):
    """
    A property predictor that computes the docking score against a user-defined target.
    Relies on TDC backend, see https://tdcommons.ai/functions/oracles/#docking-scores for setup.
    """

    def __init__(self, parameters: DockingParameters):

        docking_import_check()
        callable = Oracle(
            name=parameters.name,
            receptor_pdb_file=parameters.receptor_pdb_file,
            box_center=parameters.box_center,
            box_size=parameters.box_size,
        )
        super().__init__(callable_fn=callable, parameters=parameters)


class _MCA(PredictorAlgorithm):
    """Base class for all MCA-based predictive algorithms."""

    def __init__(self, parameters: MCAParameters):

        # Set up the configuration from the parameters
        configuration = ConfigurablePropertyAlgorithmConfiguration(
            algorithm_type=parameters.algorithm_type,
            domain=parameters.domain,
            algorithm_name=parameters.algorithm_name,
            algorithm_application=parameters.algorithm_application,
            algorithm_version=parameters.algorithm_version,
        )

        # The parent constructor calls `self.get_model`.
        super().__init__(configuration=configuration)


class Tox21(_MCA):
    """Model to predict environmental toxicity for the 12 endpoints in Tox21."""

    def get_model(self, resources_path: str) -> Predictor:
        """Instantiate the actual model.

        Args:
            resources_path: local path to model files.

        Returns:
            Predictor: the model.
        """
        # This model returns a singular reward and not a prediction for all 12 classes.
        model = _Tox21(model_path=resources_path)

        # Wrapper to get toxicity-endpoint-level predictions
        def informative_model(x: SmallMolecule) -> List[PropertyValue]:
            x = to_smiles(x)
            _ = model(x)
            return model.predictions.detach().tolist()

        return informative_model

    @classmethod
    def get_description(cls) -> str:
        text = """
        This model predicts the 12 endpoints from the Tox21 challenge.
        The endpoints are: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
        For details on the data see: https://tripod.nih.gov/tox21/challenge/.
        """
        return text


class ClinTox(_MCA):
    """Model to predict environmental toxicity for the 12 endpoints in Tox21."""

    def get_model(self, resources_path: str) -> Predictor:
        """Instantiate the actual model.

        Args:
            resources_path: local path to model files.

        Returns:
            Predictor: the model.
        """
        # This model returns a singular reward and not a prediction for both classes.
        model = _ClinTox(model_path=resources_path)

        # Wrapper to get toxicity-endpoint-level predictions
        def informative_model(x: SmallMolecule) -> List[PropertyValue]:
            x = to_smiles(x)
            _ = model(x)
            return model.predictions.detach().tolist()

        return informative_model

    @classmethod
    def get_description(cls) -> str:
        text = """
        This model is a multitask classifier for two classes:
            1. Predicted probability to receive FDA approval.
            2. Predicted probability of failure in clinical trials.
        For details on the data see: https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a.
        """
        return text


class Sider(_MCA):
    def get_model(self, resources_path: str) -> Predictor:
        """Instantiate the actual model.

        Args:
            resources_path: local path to model files.

        Returns:
            Predictor: the model.
        """
        # This model returns a singular reward and not a prediction for both classes.
        model = SIDER(model_path=resources_path)

        # Wrapper to get toxicity-endpoint-level predictions
        def informative_model(x: SmallMolecule) -> List[PropertyValue]:
            x = to_smiles(x)
            _ = model(x)
            return model.predictions.detach().tolist()

        return informative_model

    @classmethod
    def get_description(cls) -> str:
        text = """
        This model is a multitask classifier to predict side effects of drugs across
        27 classes. For details on the data see:
        https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a.
        """
        return text


class OrganTox(_MCA):
    """Model to predict toxicity for different organs."""

    def __init__(self, parameters: OrganToxParameters) -> None:

        # Extract model-specific parameters
        self.site = parameters.site
        self.toxicity_type = parameters.toxicity_type

        super().__init__(parameters=parameters)

    def get_model(self, resources_path: str) -> Predictor:
        """Instantiate the actual model.

        Args:
            resources_path: local path to model files.

        Returns:
            Predictor: the model.
        """
        # This model returns a singular reward and not a prediction for both classes.
        model = _OrganTox(
            model_path=resources_path, site=self.site, toxicity_type=self.toxicity_type
        )

        # Wrapper to get toxicity-endpoint-level predictions
        def informative_model(x: SmallMolecule) -> List[PropertyValue]:
            x = to_smiles(x)
            _ = model(x)
            all_preds = model.predictions.detach()

            return all_preds[model.class_indices].tolist()

        return informative_model

    @classmethod
    def get_description(cls) -> str:
        text = """
        This model is a multitask classifier to toxicity across different organs and
        toxicity types (`chronic`, `subchronic`, `multigenerational` or `all`).
        The organ has to specified in the constructor. The toxicity type defaults to
        `all` in which case three values are returned in the order `chronic`,
        `multigenerational` and `subchronic`.

        For details on the data see:
        https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a.
        """
        return text
