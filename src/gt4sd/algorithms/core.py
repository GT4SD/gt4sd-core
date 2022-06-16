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
"""Bases classes and core code used across multiple algorithms."""

from __future__ import annotations

import logging
import os
import shutil
import signal
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

from ..configuration import (
    GT4SDConfiguration,
    get_algorithm_subdirectories_in_cache,
    get_algorithm_subdirectories_with_s3,
    get_cached_algorithm_path,
    sync_algorithm_with_s3,
    upload_to_s3,
)
from ..exceptions import InvalidItem, S3SyncError, SamplingError
from ..training_pipelines.core import TrainingPipelineArguments

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

gt4sd_configuration_instance = GT4SDConfiguration.get_instance()

# leave typing generic for algorithm implementation
S = TypeVar("S")  # used for generated items
T = TypeVar("T")  # used for target of generation
U = TypeVar("U")  # used for additional context (e.g. part of target definition)

# callable taking a target
Targeted = Callable[[T], Iterable[Any]]
# callable not taking any target
Untargeted = Callable[[], Iterable[Any]]


class GeneratorAlgorithm(ABC, Generic[S, T]):
    """Interface for automated generation via an :class:`AlgorithmConfiguration`."""

    generator: Union[Untargeted, Targeted[T]]
    target: Optional[T]

    #: The maximum amount of time we should let the algorithm run
    max_runtime: int = gt4sd_configuration_instance.gt4sd_max_runtime
    #: The maximum number of samples a user can try to run in one go
    max_samples: int = gt4sd_configuration_instance.gt4sd_max_number_of_samples

    generate: Untargeted

    def __init__(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T] = None,
    ):
        """Targeted or untargeted generation.

        Args:
            configuration: application specific helper that allows to setup the
                generator.
            target: context or condition for the generation. Defaults to None.
        """
        logger.info(
            f"runnning {self.__class__.__name__} with configuration={configuration}"
        )
        generator = self.get_generator(configuration, target)
        setattr(
            self,
            "generate",
            self._setup_untargeted_generator(
                configuration=configuration, generator=generator, target=target
            ),
        )

    @abstractmethod
    def get_generator(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ) -> Union[Untargeted, Targeted[T]]:
        """Set up the detail implementation using the configuration.

        Note:
            This is the major method to implement in child classes, it is called
            at instantiation of the GeneratorAlgorithm and must return a callable:

            - Either :obj:`Untargeted`: the callable is taking no arguements,
              and target has to be :obj:`None`.
            - Or :obj:`Targeted`: the callable with the target (but not :obj:`None`).

        Args:
            configuration: application specific helper that allows to setup the
                generator.
            target: context or condition for the generation. Defaults to None.

        Returns:
            generator, the detail implementation used for generation.
            If the target is None, the generator is assumed to be untargeted.
        """

    def _setup_untargeted_generator(
        self,
        configuration: AlgorithmConfiguration[S, T],
        generator: Union[Untargeted, Targeted[T]],
        target: Optional[T] = None,
    ) -> Untargeted:
        """Targeted or untargeted generation.

        Args:
            configuration: application specific helper that allows to setup the
                generator.
            generator: the detail implementation used for generation.
                If the target is None, the generator is assumed to be untargeted.
            target: context or condition for the generation. Defaults to None.
        """
        self.configuration = configuration
        self.target = target
        self.generator = generator

        if target is None:
            return self.generator  # type: ignore
        else:
            return partial(self.generator, self.target)  # type: ignore

    def timeout(self, signum, frame):
        raise TimeoutError(
            "Alarm signal received, probably because a signal.alarm timed out.",
        )

    def sample(self, number_of_items: int = 100) -> Iterator[S]:
        """Generate a number of unique and valid items.

        Filters duplicate items and iterates batches of generated items to reach
        the desired number of samples, but the number of yielded items is not
        guaranteed:
        In case the generate method does not create new samples for
        GT4SD_MAX_NUMBER_OF_STUCK_CALLS times, it will terminate the
        sampling process.

        Args:
            number_of_items: number of items to generate.
                Defaults to 100.

        Raises:
            SamplingError: when requesting too many items or when no items were yielded.
                The later happens in case of not generating samples in a number of calls
                and when taking longer than the allowed time limit.

        Yields:
            the items.
        """

        if number_of_items > self.max_samples:
            detail = (
                f"{number_of_items} is too many items to generate, "
                f"must be under {self.max_samples+1} samples."
            )
            logger.warning(detail)
            raise SamplingError(title="Exceeding max_samples", detail=detail)

        def raise_if_none_sampled(items: set, detail: str):
            """If exiting early there should be at least one generated item.

            Args:
                items: to check if it's empty.
                detail: error message in case the exception is raised.

            Raises:
                SamplingError: using the given detail.
            """
            if len(items) == 0:
                raise SamplingError(
                    title="No samples generated",
                    detail="No samples generated. " + detail,
                )

        item_set = set()
        stuck_counter = 0
        item_set_length = 0
        signal.signal(signal.SIGALRM, self.timeout)
        signal.alarm(self.max_runtime)
        try:
            while True:
                generated_items = self.generate()  # type:ignore
                for item in generated_items:
                    if item in item_set:
                        continue
                    else:
                        try:
                            valid_item = self.configuration.validate_item(item)
                            yield valid_item
                            item_set.add(item)
                            if len(item_set) == number_of_items:
                                signal.alarm(0)
                                return
                        except InvalidItem as error:
                            logger.debug(
                                f"item {item} could not be validated, "
                                f"raising {error.title}: {error.detail}"
                            )
                            continue
                # make sure we don't keep sampling more than a given number of times,
                # in case no new items are generated.
                if len(item_set) == item_set_length:
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                if (
                    stuck_counter
                    >= gt4sd_configuration_instance.gt4sd_max_number_of_stuck_calls
                ):
                    detail = f"no novel samples generated for more than {gt4sd_configuration_instance.gt4sd_max_number_of_stuck_calls} cycles"
                    logger.warning(detail + ", exiting")
                    signal.alarm(0)
                    raise_if_none_sampled(items=item_set, detail=detail)
                    return
                item_set_length = len(item_set)
        except TimeoutError:
            detail = f"Samples took longer than {self.max_runtime} seconds to generate."
            logger.warning(detail + ", exiting")
            raise_if_none_sampled(items=item_set, detail=detail)
        signal.alarm(0)

    def validate_configuration(
        self, configuration: AlgorithmConfiguration
    ) -> AlgorithmConfiguration:
        """Overload to validate the a configuration for the algorithm.

        Args:
            configuration: the algorithm configuration.

        Raises:
            InvalidAlgorithmConfiguration: in case the configuration for the algorithm is invalid.

        Returns:
            the validated configuration.
        """
        logger.info("no parameters validation")
        return configuration


@dataclass
class AlgorithmConfiguration(Generic[S, T]):
    """Algorithm parameter definitions and implementation setup.

    The signature of this class constructor (given by the instance attributes) is used
    for the REST API and needs to be serializable.

    Child classes will add additional instance attributes to configure their respective
    algorithms. This will require setting default values for all of the attributes defined
    here.
    However, the values for :attr:`algorithm_name` and :attr:`algorithm_application`
    are set the registering decorator.

    This strict setup has the following desired effects:

    - Ease child implementation. For example::

        from typing import ClassVar

        from gt4sd.algorithms.registry import ApplicationsRegistry
        from gt4sd.algorithms.core import AlgorithmConfiguration

        @ApplicationsRegistry.register_algorithm_application(ChildOfGeneratorAlgorithm)
        class ConfigurationForChildOfGeneratorAlgorithm(AlgorithmConfiguration):
            algorithm_type: ClassVar[str] = 'generation'
            domain: ClassVar[str] = 'materials'
            algorithm_version: str = 'version3.14'
            actual_parameter: float = 1.61

            # no __init__ definition required


    2. Retrieve the algorithm and configuration easily (via the four class attributes)
    from the :class:`ApplicationsRegistry<gt4sd.algorithms.registry.ApplicationsRegistry>`.
    For example::

       from gt4sd.algorithms.registry import ApplicationsRegistry

       application = ApplicationsRegistry.get_application(
            algorithm_type='generation',
            domain='materials',
            algorithm_name='ChildOfGeneratorAlgorithm',
            algorithm_application='ConfigurationForChildOfGeneratorAlgorithm',
        )
        Algorithm = application.algorithm_class
        Configuration = application.configuration_class

    3. An effortless validation at instantiation via :mod:`pydantic`.

    4. An effortless mapping to artifacts on s3, see :meth:`ensure_artifacts`.

    Todo:
        show how to register a configuration manually (in case it applies to multiple
        algorithms and/or applications)

    """

    #: General type of generative algorithm.
    algorithm_type: ClassVar[str]
    #: General application domain. Hints at input/output types.
    domain: ClassVar[str]
    #: Name of the algorithm to use with this configuration.
    #:
    #: Will be set when registering to :class:`ApplicationsRegistry<gt4sd.algorithms.registry.ApplicationsRegistry>`
    algorithm_name: ClassVar[str]
    #: Unique name for the application that is the use of this
    #: configuration together with a specific algorithm.
    #:
    #: Will be set when registering to :class:`ApplicationsRegistry<gt4sd.algorithms.registry.ApplicationsRegistry>`,
    #: but can be given by direct registration (See :meth:`register_algorithm_application<gt4sd.algorithms.registry.ApplicationsRegistry.register_algorithm_application>`)
    algorithm_application: ClassVar[str]

    #: To differentiate between different versions of an application.
    #:
    #: There is no imposed naming convention.
    algorithm_version: str = ""

    def get_target_description(self) -> Optional[Dict[str, str]]:
        """Get description of the target for generation.

        Returns:
            target description, returns None in case no target is used.
        """
        return {
            "title": "Target for generation",
            "type": "object",
            "description": "Optional target for generation.",
        }

    def to_dict(self) -> Dict[str, Any]:
        """Represent the configuration as a dictionary.

        Returns:
            description of the configuration with parameters description.
        """
        base_configuration_fields_set = set(
            AlgorithmConfiguration.__dataclass_fields__.keys()  # type:ignore
        )
        application_configuration_dict = dict(description=self.__doc__)
        for name, base_description in self.__pydantic_model__.schema()[  # type:ignore
            "properties"
        ].items():
            if name not in base_configuration_fields_set:
                description = dict(
                    getattr(
                        self.__dataclass_fields__[name], "metadata", {}  # type:ignore
                    )
                )
                description.update(base_description)
                if "default" in description:
                    description["optional"] = True
                else:
                    description["optional"] = False
                application_configuration_dict[name] = description  # type:ignore
        return application_configuration_dict

    def validate_item(self, item: Any) -> S:
        """Overload to validate an item.

        Args:
            item: validate an item.

        Raises:
            InvalidItem: in case the item can not be validated.

        Returns:
            S: the validated item.
        """
        # no item validation
        return item

    @classmethod
    def get_application_prefix(cls) -> str:
        """Get prefix up to the specific application.

        Returns:
            the application prefix.
        """
        return os.path.join(
            cls.algorithm_type, cls.algorithm_name, cls.algorithm_application
        )

    @classmethod
    def list_versions(cls) -> Set[str]:
        """Get possible algorithm versions.

        S3 is searched as well as the local cache is searched for matching versions.

        Returns:
            viable values as :attr:`algorithm_version` for the environment.
        """

        prefix = cls.get_application_prefix()
        try:
            versions = get_algorithm_subdirectories_with_s3(prefix)
        except (KeyError, S3SyncError) as error:
            logger.info(
                f"searching S3 raised {error.__class__.__name__}, using local cache only."
            )
            logger.debug(error)
            versions = set()
        versions = versions.union(get_algorithm_subdirectories_in_cache(prefix))
        return versions

    @classmethod
    def list_remote_versions(cls, prefix) -> Set[str]:
        """Get possible algorithm versions on s3.
           Before uploading an artifact on S3, we need to check that
           a particular version is not already present and overwrite by mistake.
           If the final set is empty we can then upload the folder artifact.
           If the final set is not empty, we need to check that the specific version
           of interest is not present.

        only S3 is searched (not the local cache) for matching versions.

        Returns:
            viable values as :attr:`algorithm_version` for the environment.
        """
        # all name without version
        if not prefix:
            prefix = cls.get_application_prefix()
        try:
            versions = get_algorithm_subdirectories_with_s3(prefix)
        except (KeyError, S3SyncError) as error:
            logger.info(
                f"searching S3 raised {error.__class__.__name__}. This means that no versions are available on S3."
            )
            logger.debug(error)
            versions = set()
        return versions

    @classmethod
    def get_filepath_mappings_for_training_pipeline_arguments(
        cls, training_pipeline_arguments: TrainingPipelineArguments
    ) -> Dict[str, str]:
        """Ger filepath mappings for the given training pipeline arguments.

        Args:
            training_pipeline_arguments: training pipeline arguments.

        Raises:
            ValueError: in case no mapping is available.

        Returns:
            a mapping between artifacts' files and training pipeline's output files.
        """
        raise ValueError(
            f"{cls.__name__} artifacts not mapped for {training_pipeline_arguments}"
        )

    @classmethod
    def save_version_from_training_pipeline_arguments_postprocess(
        cls,
        training_pipeline_arguments: TrainingPipelineArguments,
    ):
        """Postprocess after saving.

        Args:
            training_pipeline_arguments: training pipeline arguments.
        """
        pass

    @classmethod
    def save_version_from_training_pipeline_arguments(
        cls,
        training_pipeline_arguments: TrainingPipelineArguments,
        target_version: str,
        source_version: Optional[str] = None,
    ) -> None:
        """Save a version using training pipeline arguments.

        Args:
            training_pipeline_arguments: training pipeline arguments.
            target_version: target version used to save the model in the cache.
            source_version: source version to use for missing artifacts.
                Defaults to None, a.k.a., use the default version.
        """
        filepaths_mapping: Dict[str, str] = {}
        try:
            filepaths_mapping = (
                cls.get_filepath_mappings_for_training_pipeline_arguments(
                    training_pipeline_arguments=training_pipeline_arguments
                )
            )
        except ValueError:
            logger.info(
                f"{cls.__name__} can not save a version based on {training_pipeline_arguments}"
            )
        if len(filepaths_mapping) > 0:
            if source_version is None:
                source_version = cls.algorithm_version
            source_missing_path = cls.ensure_artifacts_for_version(source_version)
            target_path = os.path.join(
                get_cached_algorithm_path(),
                cls.get_application_prefix(),
                target_version,
            )
            filepaths_mapping = {
                filename: source_filepath
                if os.path.exists(source_filepath)
                else os.path.join(source_missing_path, filename)
                for filename, source_filepath in filepaths_mapping.items()
            }
            logger.info(f"Saving artifacts into {target_path}...")
            try:
                os.makedirs(target_path)
            except OSError:
                logger.warning(
                    f"Artifacts already existing in {target_path}, overwriting them..."
                )
                os.makedirs(target_path, exist_ok=True)
            for target_filename, source_filepath in filepaths_mapping.items():
                target_filepath = os.path.join(target_path, target_filename)
                logger.info(
                    f"Saving artifact {source_filepath} into {target_filepath}..."
                )
                shutil.copyfile(source_filepath, target_filepath)

            cls.save_version_from_training_pipeline_arguments_postprocess(
                training_pipeline_arguments
            )

            logger.info(f"Artifacts saving completed into {target_path}")

    @classmethod
    def upload_version_from_training_pipeline_arguments_postprocess(
        cls,
        training_pipeline_arguments: TrainingPipelineArguments,
    ):
        """Postprocess after uploading. Not implemented yet.

        Args:
            training_pipeline_arguments: training pipeline arguments.
        """
        pass

    @classmethod
    def upload_version_from_training_pipeline_arguments(
        cls,
        training_pipeline_arguments: TrainingPipelineArguments,
        target_version: str,
        source_version: Optional[str] = None,
    ) -> None:
        """Upload a version using training pipeline arguments.

        Args:
            training_pipeline_arguments: training pipeline arguments.
            target_version: target version used to save the model in s3.
            source_version: source version to use for missing artifacts.
                Defaults to None, a.k.a., use the default version.
        """
        filepaths_mapping: Dict[str, str] = {}

        try:
            filepaths_mapping = (
                cls.get_filepath_mappings_for_training_pipeline_arguments(
                    training_pipeline_arguments=training_pipeline_arguments
                )
            )
        except ValueError:
            logger.info(
                f"{cls.__name__} can not save a version based on {training_pipeline_arguments}"
            )

        if len(filepaths_mapping) > 0:
            # probably redundant
            if source_version is None:
                source_version = cls.algorithm_version
            source_missing_path = cls.ensure_artifacts_for_version(source_version)

            # prefix for a run
            prefix = cls.get_application_prefix()
            # versions in s3 with that prefix
            versions = cls.list_remote_versions(prefix)

            # check if the target version is already in s3. If yes, don't upload.
            if target_version not in versions:
                logger.info(
                    f"There is no version {target_version} in S3, starting upload..."
                )
            else:
                logger.info(
                    f"Version {target_version} already exists in S3, skipping upload..."
                )
                return

            # mapping between filenames and paths for a version.
            filepaths_mapping = {
                filename: source_filepath
                if os.path.exists(source_filepath)
                else os.path.join(source_missing_path, filename)
                for filename, source_filepath in filepaths_mapping.items()
            }

            logger.info(
                f"Uploading artifacts into {os.path.join(prefix, target_version)}..."
            )
            try:
                for target_filename, source_filepath in filepaths_mapping.items():
                    # algorithm_type/algorithm_name/algorithm_application/version/filename
                    # for the moment we assume that the prefix exists in s3.
                    target_filepath = os.path.join(
                        prefix, target_version, target_filename
                    )
                    upload_to_s3(target_filepath, source_filepath)
                    logger.info(
                        f"Upload artifact {source_filepath} into {target_filepath}..."
                    )

            except S3SyncError:
                logger.warning("Problem with upload...")
                return

            logger.info(
                f"Artifacts uploading completed into {os.path.join(prefix, target_version)}"
            )

    @classmethod
    def ensure_artifacts_for_version(cls, algorithm_version: str) -> str:
        """The artifacts matching the path defined by class attributes and the given version are downloaded.

        That is all objects under ``algorithm_type/algorithm_name/algorithm_application/algorithm_version``
        in the bucket are downloaded.

        Args:
            algorithm_version: version of the algorithm to ensure artifacts for.

        Returns:
            the common local path of the matching artifacts.
        """
        prefix = os.path.join(
            cls.get_application_prefix(),
            algorithm_version,
        )
        try:
            local_path = sync_algorithm_with_s3(prefix)
        except (KeyError, S3SyncError) as error:
            logger.info(
                f"searching S3 raised {error.__class__.__name__}, using local cache only."
            )
            logger.debug(error)
            local_path = get_cached_algorithm_path(prefix)
            if not os.path.isdir(local_path):
                raise OSError(
                    f"artifacts directory {local_path} does not exist locally, and syncing with s3 failed: {error}"
                )

        return local_path

    def ensure_artifacts(self) -> str:
        """The artifacts matching the path defined by class attributes are downloaded.

        That is all objects under ``algorithm_type/algorithm_name/algorithm_application/algorithm_version``
        in the bucket are downloaded.

        Returns:
            the common local path of the matching artifacts.
        """
        return self.ensure_artifacts_for_version(self.algorithm_version)


def get_configuration_class_with_attributes(
    klass: Type[AlgorithmConfiguration],
) -> Type[AlgorithmConfiguration]:
    """Get AlgorithmConfiguration with set attributes.

    Args:
        klass: a class to be used to extract attributes from.

    Returns:
        a class with the attributes set.
    """
    configuration_class = deepcopy(AlgorithmConfiguration)
    setattr(configuration_class, "algorithm_type", klass.algorithm_type)
    setattr(configuration_class, "algorithm_name", klass.algorithm_name)
    setattr(configuration_class, "algorithm_application", klass.__name__)
    setattr(configuration_class, "algorithm_version", klass.algorithm_version)
    return configuration_class


class PropertyPredictor(ABC, Generic[S, U]):
    """WIP"""

    def __init__(self, context: U) -> None:
        """Property predictor to investigate items.

        Args:
            context: the context in which a property of an item can be
                computed or checked is very application specific.
        """
        self.context = context
        # or pass these with methods?

    @abstractmethod
    def satisfies(self, item: S) -> bool:
        """Check whether an item satisfies given requirements.

        Args:
            item: the item to check.

        Returns:
            bool:
        """

    def compute(self, item: S) -> Any:
        """Compute some metric/property on an item.

        Args:
            item: the item to compute a metric on.

        Returns:
            Any: the computed metric/property.
        """
