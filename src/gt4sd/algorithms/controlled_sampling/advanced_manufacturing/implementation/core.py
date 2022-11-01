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
"""Controlled sampling of concatenated encodings via Gaussian Process."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from skopt import gp_minimize
from skopt.space import Real

from .....frameworks.torch import device_claim

Point = Union[np.ndarray, torch.Tensor, pd.Series]
MINIMUM_REAL = np.finfo(dtype=np.float32).min
MAXIMUM_REAL = np.finfo(dtype=np.float32).max


def point_to_tensor(point: Point) -> torch.Tensor:
    """Convert point to tensor.

    Args:
        point: a point.

    Returns:
        tensor representing a point.
    """
    return (
        point.clone().detach().float()
        if isinstance(point, torch.Tensor)
        else torch.tensor(point).float()
    )


@dataclass
class Representation:
    """
    A generic representation for a composition problem.

    Attributes:
        model: a torch module for decoding.
        z_dimension: dimension of the latent space.
        fixed_representation: fixed representation in the latent space.
        z_index: slice for indexing a point to represent the latent space.
    """

    model: torch.nn.Module
    z_dimension: int
    fixed_representation: Optional[torch.Tensor] = None
    z_index: Optional[slice] = None

    def decode(self, z: Point) -> Any:
        """Decode the representation from the latent space.

        Args:
            z: a point in the latent space.

        Returns:
            the decoded representation.
        """
        z = torch.unsqueeze(point_to_tensor(z), dim=0)
        reconstructed = self.model.decode(z)  # type: ignore
        return reconstructed

    def deserialize(
        self, filepath: str, device: Optional[Union[torch.device, str]] = None
    ) -> None:
        """
        Deserialize a representation from file.

        Args:
            filepath: path to the serialized represantation.
            device: device where the inference
                is running either as a dedicated class or a string. If not provided is inferred.

        Returns:
            the representation object.
        """
        device = device_claim(device)
        weights = torch.load(filepath, map_location=device)
        self.model = self.model.to(device)  # type:ignore
        self.model.load_state_dict(weights)


class PropertyPredictor:
    """Property prediction class.

    Attributes:
        input_representations: order of the input representations.
    """

    input_representations: Optional[List[str]] = None

    def __call__(self, z: Point) -> float:
        """Call the property predictor on the point.

        Args:
            z: the point.

        Returns:
            the predicted property.
        """
        raise NotImplementedError("Propery prediction not implemented")


class Scaler:
    """Scaler class."""

    def __call__(self, example: Any) -> Any:
        """Scale the example appropriately.

        Args:
            example: an example prior to scaling.

        Returns:
            the scaled example.
        """
        raise NotImplementedError("Scaler not implemented not implemented")


RepresentationsDict = Dict[str, Representation]


class Objective:
    """
    Objective function for representations.
    """

    def __init__(
        self,
        targets: Dict[str, float],
        property_predictors: Dict[str, PropertyPredictor],
        representations: RepresentationsDict,
        representation_order: List[str] = None,
        scalers: Optional[Dict[str, Scaler]] = None,
        weights: Optional[Dict[str, float]] = None,
        custom_score_function: Optional[
            Callable[[Point, RepresentationsDict, Optional[Dict[str, Scaler]]], float]
        ] = None,
        custom_score_weight: float = 1.0,
        minimize: bool = True,
    ):
        """Constructs an objective function.

        Args:
            targets: a dictionary of target values.
            property_predictors: a dictionary of target property predictors.
            representations: a dictionary of decodeable representations.
            representation_order: order of the representations. Defaults to None, a.k.a., lexicographic order.
            scalers: scalers for represantation features. Defaults to None, a.k.a., no scaling.
            weights: weights for each the target. Defaults to None, a.k.a., targets evenly weigthed.
            custom_score_function: a custom score function to apply on decoded representations. Defaults to None, a.k.a., no custom score.
            custom_score_weight: weight for the custom score. Defaults to 1.0.
            minimize: whether the objective needs to be minimized. Defaults to True.
        """
        self.targets = targets
        self.property_predictors = property_predictors
        self.representations = representations
        if representation_order is None:
            self.representation_order = sorted(list(representations.keys()))
        else:
            self.representation_order = representation_order
        self.scalers = scalers
        self.weights = weights
        self.custom_score_function = custom_score_function
        self.custom_score_weight = custom_score_weight
        self.minimize = minimize

        if self.weights is None:
            weights_dictionary = dict()
            for target in self.targets.keys():
                weights_dictionary[target] = 1.0
            self.weights = weights_dictionary

    def construct_property_representation(
        self, z: torch.Tensor, property_name: str
    ) -> torch.Tensor:
        """Construct a representation for a specific property.

        The encoded point and fixed encodings (or slices thereof) if available
        are concatenated in the right order.

        Todo:
            Check explanation and improve it.

        Args:
            z: the point.
            property_name: name of the property for which to construct the representation.

        Returns:
            representation for which a specific property can be predicted.
        """
        # TODO make generic for self.representations: RepresentationsDict
        # defer this to the configuration, or some other place that defines how representations belong together
        propery_predictor = self.property_predictors[property_name]
        if propery_predictor.input_representations:
            representation_names = propery_predictor.input_representations
        else:
            representation_names = self.representation_order
        z_list = []
        for representation_name in representation_names:
            representation = self.representations[representation_name]
            if representation.fixed_representation:
                z_list.append(representation.fixed_representation)
            else:
                z_list.append(z[representation.z_index])
        z_latent = torch.cat(z_list)
        return z_latent.reshape(-1, z_latent.shape[0])

    def evaluate(self, z: Point) -> float:
        """Evaluate objective function for a point in latent space.

        Args:
            z: the point.

        Returns:
            the score of the point.
        """
        z = point_to_tensor(z)
        # predict all properties
        predicted_properties = dict()
        with torch.no_grad():
            for property_name in self.targets.keys():
                property_predictor = self.property_predictors[property_name]
                latent_z = self.construct_property_representation(
                    z=z, property_name=property_name
                )
                predicted_properties[property_name] = property_predictor(latent_z)

                # TODO aggregate the following scores over different properties,
                # or really only keep last one?
                if self.custom_score_function:
                    custom_score = self.custom_score_function(
                        latent_z,
                        self.representations,
                        self.scalers,
                    )
                else:
                    custom_score = 0

        scores = []
        for property_name, predicted_property in predicted_properties.items():
            scores.append(abs(self.targets[property_name] - predicted_property))
        score = sum(scores)

        # this is to penalize `custom_score` the non normalization
        total_score = score + self.custom_score_weight * custom_score

        if not self.minimize:
            score = -1 * total_score
        return total_score


class GaussianProcessRepresentationsSampler:
    def __init__(
        self,
        targets: Dict[str, float],
        property_predictors: Dict[str, PropertyPredictor],
        representations: RepresentationsDict,
        representation_order: Optional[List[str]] = None,
        bounds: Optional[
            Dict[str, Union[List[Tuple[float, float]], Tuple[float, float]]]
        ] = None,
        # TODO Any should be type of scaler; default to lambda returning 0?
        scalers: Optional[Dict[str, Scaler]] = None,
        weights: Optional[Dict[str, float]] = None,
        custom_score_function: Optional[
            Callable[[Point, RepresentationsDict, Optional[Dict[str, Scaler]]], float]
        ] = None,
        custom_score_weight: float = 1.0,
        minimize: bool = True,
        random_state: int = 42,
        random_starts: int = 10,
    ):
        """Constucts a GaussianProcessRepresentationsSampler.

        Args:
            targets: a dictionary of target values.
            property_predictors: a dictionary of target property predictors.
            representations: a dictionary of decodeable representations.
            representation_order: order of the representations. Defaults to None, a.k.a., lexicographic order.
            bounds: bounds for the optmization. Defaults to None, a.k.a., unbounded.
            scalers: scalers for represantation features. Defaults to None, a.k.a., no scaling.
            weights: weights for each the target. Defaults to None, a.k.a., targets evenly weigthed.
            custom_score_function: a custom score function to apply on decoded representations. Defaults to None, a.k.a., no custom score.
            custom_score_weight: weight for the custom score. Defaults to 1.0.
            minimize: whether the objective needs to be minimized. Defaults to True.
            random_state: random state. Defaults to 42.
            random_starts: number of random restarts. Defaults to 10.
        """
        self.targets = targets
        self.property_predictors = property_predictors
        self.representations = representations
        self.representation_order = representation_order
        if self.representation_order is None:
            self.representation_order = sorted(list(representations.keys()))
        self.scalers = scalers
        self.weigths = weights
        self.custom_score_function = custom_score_function
        self.custom_score_weight = custom_score_weight
        self.minimize = minimize
        self.random_state = random_state
        self.random_starts = random_starts
        self.set_bounds(bounds)
        self.dimensions = self.define_dimensions(self.representation_order)

    def _get_bounds(
        self,
        minimum_value: Union[float, np.float32],
        maximum_value: Union[float, np.float32],
        z_dimension: int,
    ) -> List[Tuple[Union[float, np.float32], Union[float, np.float32]]]:
        """
        Define a list of bounds for an hypercube.

        Args:
            minimum_value: minimum value.
            maximum_value: maximum value.
            z_dimension: dimension of the hypercube.

        Returns:
            the list of bounds.
        """
        return [(minimum_value, maximum_value) for _ in range(z_dimension)]

    def set_bounds(
        self,
        bounds: Optional[
            Dict[str, Union[List[Tuple[float, float]], Tuple[float, float]]]
        ] = None,
    ) -> None:
        """Set the bounds for the optimization.

        Args:
            bounds: bounds for the optmization. Defaults to None, a.k.a., unbounded.
        """
        self.bounds = bounds if bounds else dict()
        for representation_name, bounds in self.bounds.items():  # type:ignore
            z_dimension = self.representations[representation_name].z_dimension
            if isinstance(bounds, tuple):
                self.bounds[representation_name] = self._get_bounds(
                    bounds[0], bounds[1], z_dimension
                )
            else:
                self.bounds[representation_name] = bounds  # type:ignore
        for representation_name in self.representations.keys() - self.bounds.keys():
            z_dimension = self.representations[representation_name].z_dimension
            self.bounds[representation_name] = self._get_bounds(  # type: ignore
                MINIMUM_REAL, MAXIMUM_REAL, z_dimension
            )

    def define_dimensions(self, representation_order: List[str]) -> List[Real]:
        """Define the dimensions of the optimization space.

        Args:
            representation_order: order of the representations.

        Returns:
            a list of dimensions.
        """
        dimensions = []
        latent_index = 0
        for representation_name in representation_order:
            representation = self.representations[representation_name]
            representation_bounds = self.bounds[representation_name]
            representation.z_index = slice(
                latent_index, latent_index + representation.z_dimension
            )
            latent_index += representation.z_dimension
            dimensions.extend(  # type: ignore
                [  # type: ignore
                    Real(lower_bound, upper_bound)  # type: ignore
                    for lower_bound, upper_bound in representation_bounds
                ]
            )
        return dimensions

    def optimize(
        self,
        targets: Optional[Dict[str, float]] = None,
        relevant_representations: Optional[List[str]] = None,
        representation_order: Optional[List[str]] = None,
        z0: Optional[Point] = None,
        number_of_points: int = 1,
        number_of_steps: int = 50,
        random_state: int = 42,
        verbose: bool = False,
        weights: Optional[Dict[str, float]] = None,
        acquisition_method: str = "PI",
    ) -> List[Dict[str, Any]]:
        """Run the optimization.

        Args:
            targets: a dictionary of target values. Defaults to None, a.k.a., use the ones defined at construction time.
            relevant_representations: list of relevant representations to be optimized. Defaults to None, a.k.a., inferred from non fixed representations.
            representation_order: order of the representations. Defaults to None, a.k.a., use the one defined at construction time.
            z0: the starting point for the optimization. Defaults to None, a.k.a., perform random starts.
            number_of_points: number of optimal points to return. Defaults to 1.
            number_of_steps: number of optimization steps. Defaults to 50.
            random_state: random state. Defaults to 42.
            verbose: control verbosity. Defaults to False.
            weights: weights for each the target. Defaults to None, a.k.a., use the ones defined at construction time.
            acquisition_method: acquisition method to use in the Gaussian Process optimization. Defaults to "PI". More details at: https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html.

        Raises:
            NotImplementedError: invalid acquisition function.

        Returns:
            list of orderd optimal points with decoded relevant representations.
        """
        if representation_order is None:
            representation_order = self.representation_order
        dimensions = self.define_dimensions(cast(List[str], representation_order))

        np.random.seed(random_state)

        if acquisition_method not in ["PI", "LCB", "EI", "gp_hedge", "EIps", "PIps"]:
            raise NotImplementedError("Give valid acquisition function")

        if targets is None:
            targets = self.targets

        if weights is None:
            weights = self.weigths

        objective = Objective(
            targets=targets,
            property_predictors=self.property_predictors,
            representations=self.representations,
            representation_order=representation_order,
            scalers=self.scalers,
            weights=weights,
            custom_score_function=self.custom_score_function,
            custom_score_weight=self.custom_score_weight,
            minimize=self.minimize,
        )

        y0 = None
        random_starts: Optional[int]
        if z0 is None:
            random_starts = self.random_starts
        else:
            random_starts = None

        gaussian_process_results = gp_minimize(
            objective.evaluate,
            dimensions,
            n_calls=number_of_steps,
            n_random_starts=random_starts,
            x0=z0,
            y0=y0,
            acq_func=acquisition_method,
            random_state=np.random.RandomState(random_state),
            verbose=verbose,
        )

        objective_values, points = zip(
            *sorted(
                zip(
                    gaussian_process_results.func_vals, gaussian_process_results.x_iters
                )
            )
        )

        results_list = []

        if relevant_representations is None:
            relevant_representations = [
                representation_name
                for representation_name, representation in self.representations.items()
                if representation.fixed_representation is None
            ]

        for point, objective_value in zip(
            points[:number_of_points], objective_values[:number_of_points]
        ):
            optimization_result = {"objective": objective_value, "z": point}
            for representation_name in relevant_representations:
                representation = self.representations[representation_name]
                z = point[representation.z_index]
                optimization_result[representation_name] = representation.decode(z)

            results_list.append(optimization_result)

        return results_list


class Generator:
    def generate_samples(self, target: Any) -> List[Any]:
        """Generate samples.

        Args:
            target: target for generation.

        Returns:
            samples generated.
        """
        raise NotImplementedError("Generate samples not implemented.")
