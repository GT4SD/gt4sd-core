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
"""Collection of available methods."""


import logging
from dataclasses import dataclass as vanilla_dataclass
from dataclasses import field, make_dataclass
from functools import WRAPPER_ASSIGNMENTS, update_wrapper
from typing import Any, Callable, ClassVar, Dict, List, NamedTuple, Optional, Type

import pydantic

# pyright (pylance in VSCode) does not support pydantic typechecking
# if typing.TYPE_CHECKING:
#     from dataclasses import dataclass
# else:
#     from pydantic.dataclasses import dataclass
from pydantic.dataclasses import dataclass

from ..exceptions import DuplicateApplicationRegistration
from .core import AlgorithmConfiguration, GeneratorAlgorithm

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ConfigurationTuple(NamedTuple):
    """Attributes to uniquely describe an AlgorithmConfiguration."""

    algorithm_type: str
    domain: str
    algorithm_name: str
    algorithm_application: str


class AnnotationTuple(NamedTuple):
    annotation: type
    default_value: Any  # TODO serializable type?


@vanilla_dataclass
class AlgorithmApplication:
    """Collect all needed to run an application."""

    algorithm_class: Type[GeneratorAlgorithm]
    configuration_class: Type[AlgorithmConfiguration]
    parameters_dict: Dict[str, AnnotationTuple] = field(default_factory=dict)
    # includes algorithm_version: str


class RegistryDict(Dict[ConfigurationTuple, AlgorithmApplication]):
    """Dict that raises when reassigning an existing key."""

    def __setitem__(self, key, value):
        if self.__contains__(key):
            raise DuplicateApplicationRegistration(
                title="Applications exists",
                detail=f"key {key} was already registered and would override another application.",
            )
            # if it's really needed for some reason, delete the item first, then add it.
        else:
            super().__setitem__(key, value)


class ApplicationsRegistry:
    """Registry to collect "applications" and make them accessible.

    An application denotes the combination of an
    :class:`AlgorithmConfiguration<gt4sd.algorithms.core.AlgorithmConfiguration>` and a
    :class:`GeneratorAlgorithm<gt4sd.algorithms.core.GeneratorAlgorithm>`.
    """

    # NOTE on import of registy also ensure import of modules to populate applications
    applications: RegistryDict = RegistryDict()

    @classmethod
    def _register_application(
        cls,
        algorithm_class: Type[GeneratorAlgorithm],
        algorithm_configuration_class: Type[AlgorithmConfiguration],
    ):
        # testing that configuration class is callable without arguments
        try:
            algorithm_configuration_class()
        except pydantic.ValidationError as e:
            logger.exception(e)
        config_tuple = cls.configuration_class_as_tuple(algorithm_configuration_class)
        cls.applications[config_tuple] = AlgorithmApplication(
            algorithm_class=algorithm_class,
            configuration_class=algorithm_configuration_class,
        )

    @classmethod
    def register_algorithm_application(
        cls,
        algorithm_class: Type[GeneratorAlgorithm],
        as_algorithm_application: Optional[str] = None,
    ) -> Callable[[Type[AlgorithmConfiguration]], Type[AlgorithmConfiguration]]:
        """Complete and register a configuration via decoration.

        Args:
            algorithm_class: The algorithm that uses the configuration.
            as_algorithm_application: Optional application name to use instead of
                the configurations class name.

        Returns:
            A function to complete the configuration class' attributes to reflect
            the matching GeneratorAlgorithm and application. The final class is
            registered and returned.

        Example:
            as decorator::

                from gt4sd.algorithms.registry import ApplicationsRegistry


                @ApplicationsRegistry.register_algorithm_application(SomeAlgorithm)
                class SomeApplication(AlgorithmConfiguration):
                    algorithm_type: ClassVar[str] = 'conditional_generation'
                    domain: ClassVar[str] = 'materials'
                    algorithm_version: str = 'v0'

                    some_more_serializable_arguments_with_defaults: int = 42

        Example:
            directly, here for an additional algorithm application with the same
            algorithm::

                AnotherApplication = ApplicationsRegistry.register_algorithm_application(
                    SomeAlgorithm, 'AnotherApplication'
                )(SomeApplication)
        """

        def decorator(
            configuration_class: Type[AlgorithmConfiguration],
        ) -> Type[AlgorithmConfiguration]:
            """Complete the configuration class' attributes and register the class.

            Args:
                configuration_class: class to complete.

            Returns:
                a completed class.
            """
            VanillaConfiguration = make_dataclass(
                cls_name=configuration_class.__name__,
                # call `@dataclass` for users to avoid confusion
                bases=(vanilla_dataclass(configuration_class),),
                fields=[
                    (
                        "algorithm_name",  # type: ignore
                        ClassVar[str],
                        field(default=algorithm_class.__name__),  # type: ignore
                    ),
                    (
                        "algorithm_application",  # type: ignore
                        ClassVar[str],
                        field(
                            default=(
                                as_algorithm_application or configuration_class.__name__  # type: ignore
                            )
                        ),
                    ),
                ],  # type: ignore
            )
            # NOTE: Duplicate call necessary for pydantic >=1.10.* - see https://github.com/pydantic/pydantic/issues/4695
            PydanticConfiguration: Type[AlgorithmConfiguration] = dataclass(  # type: ignore
                VanillaConfiguration
            )
            PydanticConfiguration: Type[AlgorithmConfiguration] = dataclass(  # type: ignore
                VanillaConfiguration
            )
            # get missing entries
            missing_in__dict__ = [
                key
                for key in configuration_class.__dict__
                if key not in PydanticConfiguration.__dict__
            ]

            update_wrapper(
                wrapper=PydanticConfiguration,
                wrapped=configuration_class,
                assigned=missing_in__dict__ + list(WRAPPER_ASSIGNMENTS),
                updated=(),  # default of '__dict__' does not apply here, see missing_in__dict__
            )

            cls._register_application(algorithm_class, PydanticConfiguration)

            return PydanticConfiguration

        return decorator

    @staticmethod
    def configuration_class_as_tuple(
        algorithm_configuration_class: Type[AlgorithmConfiguration],
    ) -> "ConfigurationTuple":
        """Get a hashable identifier per application."""
        return ConfigurationTuple(
            algorithm_type=algorithm_configuration_class.algorithm_type,
            domain=algorithm_configuration_class.domain,
            algorithm_name=algorithm_configuration_class.algorithm_name,
            algorithm_application=algorithm_configuration_class.algorithm_application,
        )

    @classmethod
    def get_application(
        cls,
        algorithm_type: str,
        domain: str,
        algorithm_name: str,
        algorithm_application: str,
    ) -> AlgorithmApplication:
        return cls.applications[
            ConfigurationTuple(
                algorithm_type=algorithm_type,
                domain=domain,
                algorithm_name=algorithm_name,
                algorithm_application=algorithm_application,
            )
        ]

    @classmethod
    def get_matching_configuration_defaults(
        cls,
        algorithm_type: str,
        domain: str,
        algorithm_name: str,
        algorithm_application: str,
    ) -> Dict[str, AnnotationTuple]:
        Configuration = cls.get_application(
            algorithm_type=algorithm_type,
            domain=domain,
            algorithm_name=algorithm_name,
            algorithm_application=algorithm_application,
        ).configuration_class

        defaults_dict = {}
        for (
            argument,
            default_value,
        ) in Configuration().__dict__.items():
            defaults_dict[argument] = AnnotationTuple(
                annotation=Configuration.__annotations__[argument],
                default_value=default_value,
            )
        return defaults_dict

    @classmethod
    def get_matching_configuration_schema(
        cls,
        algorithm_type: str,
        domain: str,
        algorithm_name: str,
        algorithm_application: str,
    ) -> Dict[str, Any]:
        Configuration = cls.get_application(
            algorithm_type=algorithm_type,
            domain=domain,
            algorithm_name=algorithm_name,
            algorithm_application=algorithm_application,
        ).configuration_class
        return Configuration.__pydantic_model__.schema()  # type: ignore

    @classmethod
    def get_configuration_instance(
        cls,
        algorithm_type: str,
        domain: str,
        algorithm_name: str,
        algorithm_application: str,
        *args,
        **kwargs,
    ) -> AlgorithmConfiguration:
        """Create an instance of the matching AlgorithmConfiguration from the ApplicationsRegistry.

        Args:
            algorithm_type: general type of generative algorithm.
            domain:  general application domain. Hints at input/output types.
            algorithm_name: name of the algorithm to use with this configuration.
            algorithm_application: unique name for the application that is the use of this
                configuration together with a specific algorithm.
            algorithm_version: to differentiate between different versions of an application.
            *args: additional positional arguments passed to the configuration.
            **kwargs: additional keyword arguments passed to the configuration.

        Returns:
            an instance of the configuration.
        """
        Configuration = cls.get_application(
            algorithm_type=algorithm_type,
            domain=domain,
            algorithm_name=algorithm_name,
            algorithm_application=algorithm_application,
        ).configuration_class
        return Configuration(*args, **kwargs)

    @classmethod
    def get_application_instance(
        cls,
        algorithm_type: str,
        domain: str,
        algorithm_name: str,
        algorithm_application: str,
        target: Any = None,
        **kwargs,
    ) -> GeneratorAlgorithm:
        """Instantiate an algorithm via a matching application from the ApplicationsRegistry.

        Additional arguments are passed to the configuration and override any arguments
        in the ApplicationsRegistry.

        Args:
            algorithm_type: general type of generative algorithm.
            domain:  general application domain. Hints at input/output types.
            algorithm_name: name of the algorithm to use with this configuration.
            algorithm_application: unique name for the application that is the use of this
                configuration together with a specific algorithm.
            algorithm_version: to differentiate between different versions of an application.
            target: optional context or condition for the generation.
            **kwargs: additional keyword arguments passed to the configuration.

        Returns:
            an instance of a generative algorithm ready to sample from.
        """
        application_tuple = cls.get_application(
            algorithm_type=algorithm_type,
            domain=domain,
            algorithm_name=algorithm_name,
            algorithm_application=algorithm_application,
        )
        parameters = {
            key: annotation_tuple.default_value
            for key, annotation_tuple in application_tuple.parameters_dict.items()
        }
        parameters.update(kwargs)

        return application_tuple.algorithm_class(
            configuration=application_tuple.configuration_class(**parameters),
            target=target,
        )

    @classmethod
    def list_available(cls) -> List[Dict[str, str]]:
        available = []
        for config_tuple, application in cls.applications.items():
            available.extend(
                [
                    dict(**config_tuple._asdict(), algorithm_version=version)
                    for version in application.configuration_class.list_versions()
                ]
            )
        return available
