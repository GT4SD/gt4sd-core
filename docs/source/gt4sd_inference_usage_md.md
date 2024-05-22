# GT4SD inference examples

```{note}
You can {download}`Download the source file for this page <./gt4sd_inference_usage_md.md>`
```

```{contents}
:depth: 2
```

## Overview

This notebook show the basic usage of the GT4SD algorithms.

- running an  algorithm explicitly calling the implementation
- use the {py:class}`ApplicationsRegistry<gt4sd.algorithms.registry.ApplicationsRegistry>` to instantiate and call the algorithms

### A note on the setup

In the following we assume that the toolkit has been setup using your preferred environment file (`conda*.yml`):

```{code-block} sh
# create and activate environment
conda env create -f conda_cpu_mac.yml # or conda_cpu_linux.yml or conda_gpu.yml
conda activate gt4sd
# install the toolkit
pip install .
```

## Running algorithms explicitly

To run algorithms explicitly we only need to instantiate a {py:class}`GeneratorAlgorithm<gt4sd.algorithms.core.GeneratorAlgorithm>`
and the companion {py:class}`AlgorithmConfiguration<gt4sd.algorithms.core.AlgorithmConfiguration>`.
Then based on the actual algorithm type we might need to pass a `target` for generation.

Next we see an example of {py:class}`PaccMannRL<gt4sd.algorithms.conditional_generation.paccmann_rl.core.PaccMannRL>`,
a `conditional_generation` algorithm:

```{code-block} python
from gt4sd.algorithms.conditional_generation.paccmann_rl.core import (
    PaccMannRLProteinBasedGenerator, PaccMannRL
)
target = 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT'
configuration = PaccMannRLProteinBasedGenerator()
algorithm = PaccMannRL(configuration=configuration, target=target)
items = list(algorithm.sample(10))
print(items)
```

For vanilla generation algorithms (`generation`), like {py:class}`PolymerBlocks<gt4sd.algorithms.generation.polymer_blocks.core.PolymerBlocks>`,
the usage is analogous, but no `target` is required:

```{code-block} python
from gt4sd.algorithms.generation.polymer_blocks.core import (
    PolymerBlocksGenerator, PolymerBlocksGenerator
)
configuration = PolymerBlocksGenerator()
algorithm = PolymerBlocksGenerator(configuration=configuration)
items = list(algorithm.sample(10))
print(items)
```

## Running algorithms via the registry

Here we show how the toolkit algorithms can be instantiated and run using the {py:class}`ApplicationsRegistry<gt4sd.algorithms.registry.ApplicationsRegistry>`.

Next we see an example of {py:class}`PaccMannRL<gt4sd.algorithms.conditional_generation.paccmann_rl.core.PaccMannRL>`, a `conditional_generation` algorithm:

```{code-block} python
from gt4sd.algorithms.registry import ApplicationsRegistry
target = 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT'
algorithm = ApplicationsRegistry.get_application_instance(
    target=target,
    algorithm_type='conditional_generation',
    domain='materials',
    algorithm_name='PaccMannRL',
    algorithm_application='PaccMannRLProteinBasedGenerator',
    generated_length=5,
)
items = list(algorithm.sample(10))
print(items)
```

Similarly we can use the registry to run {py:class}`PolymerBlocks<gt4sd.algorithms.generation.polymer_blocks.core.PolymerBlocks>`:

```{code-block} python
from gt4sd.algorithms.registry import ApplicationsRegistry
algorithm = ApplicationsRegistry.get_application_instance(
    algorithm_type='generation',
    domain='materials',
    algorithm_name='PolymerBlocks',
    algorithm_application='PolymerBlocksGenerator',
    generated_length=10,
)
items = list(algorithm.sample(10))
print(items)
```