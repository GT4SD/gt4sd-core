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
from typing import Any, Callable, Union

from pydantic import BaseModel, Field

PropertyValue = Union[float, int]


class DomainSubmodule(str, Enum):
    molecules: str = "molecules"
    properties: str = "properties"
    crystals: str = "crystals"


class PropertyPredictorParameters(BaseModel):
    """Abstract class for property computation."""

    pass


# Base class for property predictors that use S3 artifacts
class S3Parameters(PropertyPredictorParameters):

    algorithm_type: str = "prediction"

    domain: DomainSubmodule = Field(
        ..., example="molecules", description="Submodule of gt4sd.properties"
    )
    algorithm_name: str = Field(..., example="MCA", description="Name of the algorithm")
    algorithm_version: str = Field(
        ..., example="v0", description="Version of the algorithm"
    )
    algorithm_application: str = Field(..., example="Tox21")


class ApiTokenParameters(PropertyPredictorParameters):
    api_token: str = Field(
        ...,
        example="apk-c9db......",
        description="The API token/key to access the service",
    )


class IpAdressParameters(PropertyPredictorParameters):

    host_ip: str = Field(
        ...,
        example="xx.xx.xxx.xxx",
        description="The host IP address to access the service",
    )


class PropertyPredictor:
    """PropertyPredictor base class."""

    def __init__(
        self, parameters: PropertyPredictorParameters = PropertyPredictorParameters()
    ) -> None:
        """Construct a PropertyPredictor using the related parameters.
        Args:
            parameters: parameters to configure the predictor.
        """
        self.parameters = parameters

    def __call__(self, sample: Any) -> PropertyValue:
        """Call the PropertyPredictor.

        Args:
            sample: a sample to use for predicting the property of interest.

        Returns:
            Property:

        Example:
            An example for predicting properties::

                property_predictor = PropertyPredictor(parameters)
                value = property_predictor(sample)
        """
        raise NotImplementedError


class CallablePropertyPredictor(PropertyPredictor):
    """Property predictor based on a callable."""

    def __init__(
        self,
        callable_fn: Callable,
        parameters: PropertyPredictorParameters = PropertyPredictorParameters(),
    ) -> None:
        self.callable_fn = callable_fn
        super().__init__(parameters=parameters)

    def __call__(self, sample: Any) -> PropertyValue:
        """Call the PropertyPredictor.

        Args:
            sample: a sample to use for predicting the property of interest.

        Returns:
            Property: Property predicted by the predictor.

        Example:
            An example for predicting properties::

                property_predictor = CallablePropertyPredictor(callable_fn=lambda a: id(a), parameters)
                value = property_predictor(sample)
        """
        return self.callable_fn(sample)


class ConfigurableCallablePropertyPredictor(CallablePropertyPredictor):
    """Property predictor based on a callable that is configured using the provided parameters."""

    def __call__(self, sample: Any) -> PropertyValue:
        """Call the PropertyPredictor.

        Args:
            sample: a sample to use for predicting the property of interest.

        Returns:
            Property:

        Example:
            An example for predicting properties::

                property_predictor = CallablePropertyPredictor(callable_fn=lambda a, b: id(a), parameters)
                value = property_predictor(sample)
        """
        return self.callable_fn(sample, **self.parameters.dict())
