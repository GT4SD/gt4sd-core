# GT4SD (Generative Toolkit for Scientific Discovery)

[![PyPI version](https://badge.fury.io/py/gt4sd.svg)](https://badge.fury.io/py/gt4sd)
[![Actions tests](https://github.com/gt4sd/gt4sd-core/actions/workflows/tests.yaml/badge.svg)](https://github.com/gt4sd/gt4sd-core/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions](https://img.shields.io/badge/contributions-welcome-blue)](https://github.com/GT4SD/gt4sd-core/blob/main/CONTRIBUTING.md)
[![Docs](https://img.shields.io/badge/website-live-brightgreen)](https://gt4sd.github.io/gt4sd-core/)
[![Total downloads](https://pepy.tech/badge/gt4sd)](https://pepy.tech/project/gt4sd)
[![Monthly downloads](https://pepy.tech/badge/gt4sd/month)](https://pepy.tech/project/gt4sd)

<img src="./docs/_static/gt4sd_logo.png" alt="logo" width="500"/>

The GT4SD (Generative Toolkit for Scientific Discovery) is an open-source platform to accelerate hypothesis generation in the scientific discovery process. It provides a library for making state-of-the-art generative AI models easier to use.

For full details on the library API and examples see the [docs](https://gt4sd.github.io/gt4sd-core/).

## Installation

### pip

If you simply want to use `gt4sd` in your projects, install it via `pip` from [PyPI](https://pypi.org/project/gt4sd/):

```sh
pip install gt4sd
```

You can install `gt4sd` directly from GitHub:

```sh
pip install git+https://github.com/GT4SD/gt4sd-core
```

### Development setup & installation

If you would like to contribute to the package, we recommend the following development setup:

```sh
git clone git@github.com:GT4SD/gt4sd-core.git
cd gt4ds-core
conda env create -f conda.yml
conda activate gt4sd
pip install -e .
```

Learn more in [CONTRIBUTING.md](./CONTRIBUTING.md)

## Supported packages

Beyond implementing various generative modeling inference and training pipelines GT4SD is designed to provide a high-level API that implement an harmonized interface for several existing packages:

- [GuacaMol](https://github.com/BenevolentAI/guacamol): inference pipelines for the baselines models.
- [MOSES](https://github.com/molecularsets/moses): inference pipelines for the baselines models.
- [TAPE](https://github.com/songlab-cal/tape): encoder modules compatible with the protein language models.
- [PaccMann](https://github.com/PaccMann/): inference pipelines for all algorithms of the PaccMann family as well as traiing pipelines for the generative VAEs.
- [transformers](https://huggingface.co/transformers): training and inference pipelines for generative models from the [HuggingFace Models](https://huggingface.co/models)

## Using GT4SD

### Running inference pipelines

Running an algorithm is as easy as typing:

```python
from gt4sd.algorithms.conditional_generation.paccmann_rl.core import (
    PaccMannRLProteinBasedGenerator, PaccMannRL
)
target = 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT'
# algorithm configuration with default parameters
configuration = PaccMannRLProteinBasedGenerator()
# instantiate the algorithm for sampling
algorithm = PaccMannRL(configuration=configuration, target=target)
items = list(algorithm.sample(10))
print(items)
```

Or you can use the `ApplicationRegistry` to run an algorithm instance using a
serialized representation of the algorithm:

```python
from gt4sd.algorithms.registry import ApplicationsRegistry
target = 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT'
algorithm = ApplicationsRegistry.get_application_instance(
    target=target,
    algorithm_type='conditional_generation',
    domain='materials',
    algorithm_name='PaccMannRL',
    algorithm_application='PaccMannRLProteinBasedGenerator',
    generated_length=32,
    # include additional configuration parameters as **kwargs
)
items = list(algorithm.sample(10))
print(items)
```

### Running training pipelines via the CLI command

GT4SD provides a trainer client based on the `gt4sd-trainer` CLI command. The trainer currently supports training pipelines for language modeling (`language-modeling-trainer`), PaccMann (`paccmann-vae-trainer`) and Granular (`granular-trainer`, multimodal compositional autoencoders).

```console
$ gt4sd-trainer --help
usage: gt4sd-trainer [-h] --training_pipeline_name TRAINING_PIPELINE_NAME
                     [--configuration_file CONFIGURATION_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --training_pipeline_name TRAINING_PIPELINE_NAME
                        Training type of the converted model, supported types:
                        granular-trainer, language-modeling-trainer, paccmann-
                        vae-trainer. (default: None)
  --configuration_file CONFIGURATION_FILE
                        Configuration file for the trainining. It can be used
                        to completely by-pass pipeline specific arguments.
                        (default: None)
```

To launch a training you have two options.

You can either specify the training pipeline and the path of a configuration file that contains the needed training parameters:

```sh
gt4sd-trainer  --training_pipeline_name ${TRAINING_PIPELINE_NAME} --configuration_file ${CONFIGURATION_FILE}
```

Or you can provide directly the needed parameters as argumentsL

```sh
gt4sd-trainer  --training_pipeline_name language-modeling-trainer --type mlm --model_name_or_path mlm --training_file /pah/to/train_file.jsonl --validation_file /path/to/valid_file.jsonl 
```

To get more info on a specific training pipeleins argument simply type:

```sh
gt4sd-trainer --training_pipeline_name ${TRAINING_PIPELINE_NAME} --help
```

<!-- Adding examples and notebooks is a must here -->

<!-- Having a list of all supported algorithms wouldn be nice! -->

## References

If you use `gt4sd` in your projects, please consider citing the following:

```bib
@software{GT4SD,
author = {GT4SD Team},
month = {2},
title = {{GT4SD (Generative Toolkit for Scientific Discovery)}},
url = {https://github.com/GT4SD/gt4sd-core},
version = {main},
year = {2022}
}
```

## License

The `gt4sd` codebase is under MIT license.
For individual model usage, please refer to the model licenses found in the original packages.
