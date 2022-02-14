
# Adding a new algorithm

## Getting started

The general structure of a single conditional generation algorithm in `gt4sd-core` is shown here

```{code-block} sh
gt4sd-core
    |gt4sd
    |   |algorithms
    |   |   |conditional_generation
    |   |   |   |__init__.py
    |   |   |   |[My_Algorithm]
    |   |   |   |   |__init__.py
    |   |   |   |   |core.py
    |   |   |   |   |implementation
```

At the time of writing these are the only files you will need to be aware of to add your own custom algorithm to `gt4sd-core`. Here we will talk through the implementation of a template algorithm we have called {py:class}`Template<gt4sd.algorithms.conditional_generation.template.core.Template>`, this algorithm will take a string input and return a list with the single item `Hello` + input, i.e. input=`World` outputs the list `[Hello World]`.

Since `Template` is a conditional generation algorithm, I have created the `My_Algorithm` folder (`template`) in the `conditional_generation` folder, and inside added the 3 files `__init__.py`, `core.py`, and `implementation.py`.

## Implementation

Starting with the file `implementation.py` we have the following code

```{code-block} python
class Generator:
    """Basic Generator for the template algorithm"""

    def __init__(
        self,
        resources_path: str,
        temperature: int
    ):
        """Initialize the Generator.

        Args:
            resources_path: directory where to find models and parameters.

        """

        self.resources_path = resources_path
        self.temperature = temperature

    def hello_name(
        self,
        name: str,
    ) -> List[str]:
        """Validate a list of strings.

        Args:
            name: a string.

        Returns:
            a list containing salutation and temperature converted to fahrenheit.
        """
        return [
            f"Hello {str(name)} {random.randint(1, int(1e6))} times and, fun fact, {str(self.temperature)} celsius equals to {(self.temperature * (9/5) + 32)} fahrenheit."
        ]
```

Here we have created a class called {py:class}`Generator<gt4sd.algorithms.conditional_generation.template.implementation.Generator>` with 2 functions:

```{code-block} python
___init__(self, resources_path: str, temperature: int)
```

which is used to initialise the generator, set addional parameters ( in this case `temperature` is the addional parameter ) and the directory from where the model is located, and

```{code-block} python
hello_name(self, name: str) -> List[str]
```

which is the actual implementation of the algorithm. For this guide our algorithm takes in a string `name` and `temperature` and outputs a single string `Hello name a random number of times and temperature in fahrenheit` in a list.

For your specific algorithm this second function will be your own code.

## Core

Now we will look into the file `core.py`

```{code-block} python
import logging
from typing import ClassVar, Optional, TypeVar, Callable, Iterable, Any, Dict

from ...core import AlgorithmConfiguration, GeneratorAlgorithm  # type: ignore
from ...registry import ApplicationsRegistry  # type: ignore
from .implementation import Generator  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = TypeVar("T")
S = TypeVar("S")
Targeted = Callable[[T], Iterable[Any]]


class Template(GeneratorAlgorithm[S, T]):
    """Template Algorithm."""

    def __init__(
        self, configuration: AlgorithmConfiguration[S, T], target: Optional[T] = None
    ):
        """Template Generation

        Args:
            configuration: domain and application
                specification, defining types and validations.
            target: Optional, in this inistance we will convert to a string.

        Example:
            An example for using this temmplate::

            target = 'World'
            configuration = TemplateGenerator()
            algorithm = Template(configuration=configuration, target=target)
            items = list(algorithm.sample(1))
            print(items)
        """

        configuration = self.validate_configuration(configuration)
        # TODO there might also be a validation/check on the target input

        super().__init__(
            configuration=configuration,
            target=target,  # type:ignore
        )

    def get_generator(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ) -> Targeted[T]:
        """Get the function to hello_name from generator.

        Args:
            configuration: helps to set up the application.
            target: context or condition for the generation. Just an optional string here.

        Returns:
            callable generating a list of 1 item containing salutation and temperature converted to fahrenheit.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: Generator = configuration.get_conditional_generator(  # type: ignore
            self.local_artifacts
        )
        return implementation.hello_name  # type:ignore

    def validate_configuration(
        self, configuration: AlgorithmConfiguration
    ) -> AlgorithmConfiguration:
        # TODO raise InvalidAlgorithmConfiguration
        assert isinstance(configuration, AlgorithmConfiguration)
        return configuration


@ApplicationsRegistry.register_algorithm_application(Template)
class TemplateGenerator(AlgorithmConfiguration[str, str]):
    """Configuration for specific generator."""

    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"
    g
    temperature: int = field(
        default=36,
        metadata=dict(
            description="Temperature parameter ( in celsius )"
        ),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.
        Returns:
            target description.
        """
        return {
            "title": "Target name",
            "description": "A simple string to define the name in the output [Hello name].",
            "type": "string",
        }

    def get_conditional_generator(self, resources_path: str) -> Generator:
        return Generator(
            resources_path=resources_path,
            temperature=self.temperature
        )

```

`core.py` will contain at least two classes. The first is named after your algorithm, in our example this class is called {py:class}`Template<gt4sd.algorithms.conditional_generation.template.core.Template>`, which is initialised with a `GeneratorAlgorithm` object. The second is an `AlgorithmConfiguration`, in this case called {py:class}`TemplateGenerator<gt4sd.algorithms.conditional_generation.template.core.TemplateGenerator>`, which is used to configure your algorithm.

### Template

Your algorithm, {py:class}`Template<gt4sd.algorithms.conditional_generation.template.core.Template>` for us, needs to contain at least two functions.

```{code-block} python
__init__(self, configuration, target)
```

This is used to initialise the algorithm by passing in the algorithm configuration and an optional parameter. The configuration parameter is the object created from the `TemplateGenerator` class and the `target` parameter in this case will be string we are passing through to our algorithm.

```{code-block} python
get_generator(self, configuration, target)
```

This function is required get the implementation from the generator configuration. It then returns the function in the implementation with corresponds with your algorithm. In our case this is {py:class}`implementation.hello_name<gt4sd.algorithms.conditional_generation.template.implementation.Generator>`.

```{code-block} python
validate_configuration(self, configuration)
```

This is a optional helper function to validate that a valid configuration is provided. A similar validation method could be created to check that a user has added a valid input or `target` in our case.

### TemplateGenerator

Finally you will need to create a specific configuration for your algorithm, In our case called {py:class}`TemplateGenerator<gt4sd.algorithms.conditional_generation.template.core.TemplateGenerator>`, note that in our implementation we have tagged this class with `@ApplicationsRegistry.register_algorithm_application(Template)`. This decorator is needed to add the algorithm to the `ApplicationRegistry,` you should add a similar decorator to your implementation of `AlgorithmConfiguration` replacing the `Template` name in the decorator with the name of your algorithm.

In this class there are three required strings `algorithm_type`, `domain`, and `algorithm_version` which are all self explanatory:

- `algorithm_type` is the type of algorithm you are implementing, i.e., `generation`.
- `domain` is the domain your algorithm is applied to, i.e., `materials`.
- `algorithm_version` is the version of algorithm you are on, i.e., `v0`.

These strings will set the location for resource cache of the model.
Make sure you create the appropriate path in the S3 storage used (default bucket name `algorithms`, `algorithms/{algorithm_type}/{algorithm_name}/{algorithm_application}/{algorithm_version}`) where your artifacts will be uploaded: `algorithms/conditional_generation/Template/TemplateGenerator/v0`.

There are two required functions for our configuration:

The first function needed is

```{code-block} python
get_target_description(self) -> Dict[str, str]
```

which returns a dictionary defining the type of `target`, for our algorithm this is a string, and both a title and description of what that `target` represents. This method is needed to populate documentation for the end user.

The final function needed is

```{code-block} python
get_conditional_generator(self, resources_path: str) -> Generator
```

which is used to return the Generator from the resource path.

Note that if we wish to implement specific configurations for this algorithm this can also be set by creating additional `AlgorithmGenerator`s in `core.py` and adding each parameter via a `field` object i.e.

```{code-block} python
    algorithm_type: ClassVar[str] = 'conditional_generation'
    domain: ClassVar[str] = 'materials'
    algorithm_version: str = 'v0'

    batch_size: int = field(
        default=32,
        metadata=dict(description="Batch size used for the generative model sampling."),
    )
    temperature: float = field(
        default=1.4,
        metadata=dict(
            description="Temperature parameter for the softmax sampling in decoding."
        ),
    )
    generated_length: int = field(
        default=100,
        metadata=dict(
            description="Maximum length in tokens of the generated molcules (relates to the SMILES length)."
        ),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.
        Returns:
            target description.
        """
        return {
            "title": "Gene expression profile",
            "description": "A gene expression profile to generate effective molecules against.",
            "type": "list",
        }

    def get_conditional_generator(
        self, resources_path: str
    ) -> ProteinSequenceConditionalGenerator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate_batch<gt4sd.algorithms.conditional_generation.paccmann_rl.implementation.ConditionalGenerator.generate_batch>` method for targeted generation.
        """
        return ProteinSequenceConditionalGenerator(
            resources_path=resources_path,
            temperature=self.temperature,
            generated_length=self.generated_length,
            samples_per_protein=self.batch_size,
        )
```

`field` is used to set a default configuration and a description of the parameter which is used to populate the documentation returned to the end user similar to `get_target_description`. Algorithm configuration parameters can be validated by adding the implementation of a `__post_init__` method as described [here](https://docs.python.org/3/library/dataclasses.html#post-init-processing).

## Final steps

Finally to complete our implementation we need to import all the algorithms and configurations in our created `__init__.py` folder like so

```{code-block} python
from .core import (
    Template,
    TemplateGenerator,
)

__all__ = [
    'Template',
    'TemplateGenerator',
]
```

and to automatically add the algorithm to the registry without any manual imports, we have to import the generator class which in our case is `TemplateGenerator` to the outermost `__init__.py` of the subdirectory `algorithms`.

```{code-block} python
from .template.core import TemplateGenerator
```

## Using a custom algorithm

Now that the new algorithm is implemented we can use it the same was as shown before

### Explicitly

```{code-block} python
from gt4sd.algorithms.conditional_generation.template import (
    TemplateGenerator, Template
)
target = 'World'
configuration = TemplateGenerator()
algorithm = Template(configuration=configuration, target=target)
items = list(algorithm.sample(1))
print(items)
```

### Registry

```{code-block} python
from gt4sd.algorithms.registry import ApplicationsRegistry
target = 'World'
algorithm = ApplicationsRegistry.get_application_instance(
    target=target,
    algorithm_type='conditional_generation',
    domain='materials',
    algorithm_name='Template',
    algorithm_application='TemplateGenerator',
)
items = list(algorithm.sample(1))
print(items)
```
