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
"""Implementation of PaccMann^GP conditional generator."""

import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch
from paccmann_chemistry.models.vae import StackGRUDecoder, StackGRUEncoder, TeacherVAE
from paccmann_chemistry.utils.search import SamplingSearch
from paccmann_gp.affinity_minimization import AffinityMinimization
from paccmann_gp.combined_minimization import CombinedMinimization
from paccmann_gp.gp_optimizer import GPOptimizer
from paccmann_gp.mw_minimization import MWMinimization
from paccmann_gp.qed_minimization import QEDMinimization
from paccmann_gp.sa_minimization import SAMinimization
from paccmann_gp.smiles_generator import SmilesGenerator
from paccmann_predictor.models import MODEL_FACTORY
from pytoda.proteins.protein_language import ProteinLanguage
from pytoda.smiles.smiles_language import SMILESLanguage

from ....frameworks.torch import device_claim

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MINIMIZATION_FUNCTIONS = {
    "qed": QEDMinimization,
    "sa": SAMinimization,
    "molwt": MWMinimization,
    "affinity": AffinityMinimization,
}


class GPConditionalGenerator:
    """Conditional generator as implemented in https://doi.org/10.1021/acs.jcim.1c00889."""

    def __init__(
        self,
        resources_path: str,
        temperature: float = 1.4,
        generated_length: int = 100,
        batch_size: int = 32,
        limit: float = 5.0,
        acquisition_function: str = "EI",
        number_of_steps: int = 32,
        number_of_initial_points: int = 16,
        initial_point_generator: str = "random",
        seed: int = 42,
        number_of_optimization_rounds: int = 1,
        sampling_variance: float = 0.1,
        samples_for_evaluation: int = 4,
        maximum_number_of_sampling_steps: int = 32,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Initialize the conditional generator.

        Args:
            resources_path: directory where to find models and parameters.
            temperature: temperature parameter for the softmax sampling in decoding. Defaults to 1.4.
            generated_length: maximum length in tokens of the generated molcules (relates to the SMILES length). Defaults to 100.
            batch_size: batch size used for the generative model sampling. Defaults to 16.
            limit: hypercube limits in the latent space. Defaults to 5.0.
            acquisition_function: acquisition function used in the Gaussian process. Defaults to "EI". More details in https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html.
            number_of_steps: number of steps for an optmization round. Defaults to 32.
            number_of_initial_points: number of initial points evaluated. Defaults to 16.
            initial_point_generator: scheme to generate initial points. Defaults to "random". More details in https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html.
            seed: seed used for random number generation in the optimizer. Defaults to 42.
            number_of_optimization_rounds: maximum number of optimization rounds. Defaults to 1.
            sampling_variance: variance of the Gaussian noise applied during sampling from the optimal point. Defaults to 0.1.
            samples_for_evaluation: number of samples averaged for each minimization function evaluation. Defaults to 4.
            maximum_number_of_sampling_steps: maximum number of sampling steps in an optmization round. Defaults to 32.
            device: . Defaults to None, a.k.a, picking a default one ("gpu" if present, "cpu" otherwise).
        """
        # device
        self.device = device_claim(device)
        # setting sampling parameters
        self.temperature = temperature
        self.generated_length = generated_length
        self.batch_size = batch_size
        # setting VAE parameters
        self.svae_params = dict()
        with open(os.path.join(resources_path, "vae_model_params.json"), "r") as f:
            self.svae_params.update(json.load(f))
        smiles_language = SMILESLanguage.load(
            os.path.join(resources_path, "selfies_language.pkl")
        )
        # initialize encoder, decoder, testVAE, and GP_generator_MW
        self.gru_encoder = StackGRUEncoder(self.svae_params)
        self.gru_decoder = StackGRUDecoder(self.svae_params)
        self.gru_vae = TeacherVAE(self.gru_encoder, self.gru_decoder)
        self.gru_vae.load_state_dict(
            torch.load(
                os.path.join(resources_path, "vae_weights.pt"),
                map_location=self.device,
            )
        )
        self.gru_vae._associate_language(smiles_language)
        self.gru_vae.eval()
        self.smiles_generator = SmilesGenerator(
            self.gru_vae,
            search=SamplingSearch(temperature=self.temperature),
            generated_length=self.generated_length,
        )
        self.latent_dim = self.gru_decoder.latent_dim
        # setting affinity predictor parameters
        with open(os.path.join(resources_path, "mca_model_params.json")) as f:
            self.predictor_params = json.load(f)
        self.affinity_predictor = MODEL_FACTORY["bimodal_mca"](self.predictor_params)
        self.affinity_predictor.load(
            os.path.join(resources_path, "mca_weights.pt"),
            map_location=self.device,
        )
        affinity_protein_language = ProteinLanguage.load(
            os.path.join(resources_path, "protein_language.pkl")
        )
        affinity_smiles_language = SMILESLanguage.load(
            os.path.join(resources_path, "smiles_language.pkl")
        )
        self.affinity_predictor._associate_language(affinity_smiles_language)
        self.affinity_predictor._associate_language(affinity_protein_language)
        self.affinity_predictor.eval()
        # setting optimizer parameters
        self.limit = limit
        self.acquisition_function = acquisition_function
        self.number_of_initial_points = number_of_initial_points
        if number_of_steps < self.number_of_initial_points:
            logger.warning(
                "number of initial points is larger than number of steps "
                f"({self.number_of_initial_points}/{number_of_steps}). "
                f"Resetting number of steps to {self.number_of_initial_points}."
            )
            self.number_of_steps = self.number_of_initial_points
        else:
            self.number_of_steps = number_of_steps
        self.initial_point_generator = initial_point_generator
        self.seed = seed
        self.number_of_optimization_rounds = number_of_optimization_rounds
        self.sampling_variance = sampling_variance
        self.samples_for_evaluation = samples_for_evaluation
        self.maximum_number_of_sampling_steps = maximum_number_of_sampling_steps

    def target_to_minimization_function(
        self, target: Union[Dict[str, Dict[str, Any]], str]
    ) -> CombinedMinimization:
        """Use the target to configure a minimization function.

        Args:
            target: dictionary or JSON string describing the optimization target.

        Returns:
            a minimization function.
        """
        if isinstance(target, str):
            target_dictionary = json.loads(target)
        elif isinstance(target, dict):
            target_dictionary = deepcopy(target)
        else:
            raise ValueError(
                f"{target} of type {type(target)} is not supported: provide 'str' or 'Dict[str, Dict[str, Any]]'"
            )
        minimization_functions = []
        weights = []
        for minimization_function_name, parameters in target_dictionary.items():
            weight = 1.0
            if "weight" in parameters:
                weight = parameters.pop("weight")
            function_parameters = {
                **parameters,
                **{
                    "batch_size": self.samples_for_evaluation,
                    "smiles_decoder": self.smiles_generator,
                },
            }
            minimization_function = MINIMIZATION_FUNCTIONS[minimization_function_name]
            if minimization_function_name == "affinity":
                function_parameters["affinity_predictor"] = self.affinity_predictor
            minimization_functions.append(minimization_function(**function_parameters))
            weights.append(weight)
        return CombinedMinimization(
            minimization_functions=minimization_functions,
            batch_size=1,
            function_weights=weights,
        )

    def generate_batch(self, target: Any) -> List[str]:
        """Generate molecules given a target.

        Args:
            target: dictionary or JSON string describing the optimization target.

        Returns:
            a list of molecules as SMILES string.
        """
        # make sure the seed is transformed to avoid redundancy over multiple calls (using Knuth multiplicative hashing)
        self.seed = self.seed * 2654435761 % 2**32
        logger.info(f"configuring optimization for target: {target}")
        # target configuration
        self.target = target
        self.minimization_function = self.target_to_minimization_function(self.target)
        # optimizer configuration
        self.target_optimizer = GPOptimizer(self.minimization_function.evaluate)
        optimization_parameters = dict(
            dimensions=[(-self.limit, self.limit)] * self.latent_dim,
            acq_func=self.acquisition_function,
            n_calls=self.number_of_steps,
            n_initial_points=self.number_of_initial_points,
            initial_point_generator=self.initial_point_generator,
            random_state=self.seed,
        )
        logger.info(
            f"running optimization with the following parameters: {optimization_parameters}"
        )
        smiles_set = set()
        logger.info(
            f"running at most {self.number_of_optimization_rounds} optmization rounds"
        )
        for optimization_round in range(self.number_of_optimization_rounds):
            logger.info(f"starting round {optimization_round + 1}")
            optimization_parameters["random_state"] += optimization_round  # type:ignore
            res = self.target_optimizer.optimize(optimization_parameters)
            latent_point = torch.tensor([[res.x]])
            smiles_set_per_round = set()

            logger.info(f"starting sampling for {optimization_round + 1}")
            for _ in range(self.maximum_number_of_sampling_steps):
                generated_smiles = self.smiles_generator.generate_smiles(
                    latent_point.repeat(1, self.batch_size, 1)
                    + torch.cat(
                        (
                            torch.zeros(1, 1, self.latent_dim),
                            (self.sampling_variance**0.5)
                            * torch.randn(1, self.batch_size - 1, self.latent_dim),
                        ),
                        dim=1,
                    )
                )
                smiles_set_per_round.update(set(generated_smiles))
            smiles_set.update(smiles_set_per_round)
            logger.info(f"completing round {optimization_round + 1}")
        logger.info(f"generated {len(smiles_set)} molecules in the current run")
        return list(
            [molecule_smiles for molecule_smiles in smiles_set if molecule_smiles]
        )
