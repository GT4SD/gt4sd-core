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

### requirements

Currently `gt4sd` relies on:

- python>=3.7,<3.8
- pip>=19.1,<20.3

We are actively working on relaxing these, so stay tuned or help us with this by [contributing](./CONTRIBUTING.md) to the project.

### conda

The recommended way to install the `gt4sd` is to create a dedicated conda environment, this will ensure all requirements are satisfied:

```sh
conda env create -f conda.yml
conda activate gt4sd
```

And install the package via `pip` from [PyPI](https://pypi.org/project/gt4sd/):

```sh
pip install gt4sd
```

**NOTE:** In case you want to reuse an existing compatible environment (see [requirements](#requirements)), you can use `pip`, but as of now (:eyes: on [issue](https://github.com/GT4SD/gt4sd-core/issues/31) for changes), some dependencies require installation from GitHub, so for a complete setup install them with:

```sh
pip install -r vcs_requirements.txt
```

### Development setup & installation

If you would like to contribute to the package, we recommend the following development setup:

```sh
conda env create -f conda.yml
conda activate gt4sd
# install gt4sd in editable mode
pip install --no-deps -e .
```

Learn more in [CONTRIBUTING.md](./CONTRIBUTING.md)

## Getting started

After install you can use `gt4sd` right away in your discovery workflows.

### Running inference pipelines in your python code

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

Or you can use the `ApplicationRegistry` to run an algorithm instance using a serialized representation of the algorithm:

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

### Running inference pipelines via the CLI command

GT4SD can run inference pipelines based on the `gt4sd-inference` CLI command.
It allows to run all inference algorithms directly from the command line.

You can run inference pipelines simply typing:

```console
gt4sd-inference --algorithm_name PaccMannRL --algorithm_application PaccMannRLProteinBasedGenerator --target MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT --number_of_samples 10
```

The command supports multiple parameters to select an algorithm and configure it for inference:

```console
$ gt4sd-inference --help
usage: gt4sd-inference [-h] [--algorithm_type ALGORITHM_TYPE]
                       [--domain DOMAIN] [--algorithm_name ALGORITHM_NAME]
                       [--algorithm_application ALGORITHM_APPLICATION]
                       [--algorithm_version ALGORITHM_VERSION]
                       [--target TARGET]
                       [--number_of_samples NUMBER_OF_SAMPLES]
                       [--configuration_file CONFIGURATION_FILE]
                       [--print_info [PRINT_INFO]]

optional arguments:
  -h, --help            show this help message and exit
  --algorithm_type ALGORITHM_TYPE
                        Inference algorithm type, supported types:
                        conditional_generation, controlled_sampling,
                        generation, prediction. (default: None)
  --domain DOMAIN       Domain of the inference algorithm, supported types:
                        materials, nlp. (default: None)
  --algorithm_name ALGORITHM_NAME
                        Inference algorithm name. (default: None)
  --algorithm_application ALGORITHM_APPLICATION
                        Inference algorithm application. (default: None)
  --algorithm_version ALGORITHM_VERSION
                        Inference algorithm version. (default: None)
  --target TARGET       Optional target for generation represented as a
                        string. Defaults to None, it can be also provided in
                        the configuration_file as an object, but the
                        commandline takes precendence. (default: None)
  --number_of_samples NUMBER_OF_SAMPLES
                        Number of generated samples, defaults to 5. (default:
                        5)
  --configuration_file CONFIGURATION_FILE
                        Configuration file for the inference pipeline in JSON
                        format. (default: None)
  --print_info [PRINT_INFO]
                        Print info for the selected algorithm, preventing
                        inference run. Defaults to False. (default: False)
```

You can use `gt4sd-inference` to directly get information on the configuration parameters for the selected algorithm:

```console
gt4sd-inference --algorithm_name PaccMannRL --algorithm_application PaccMannRLProteinBasedGenerator --print_info
INFO:gt4sd.cli.inference:Selected algorithm: {'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'PaccMannRL', 'algorithm_application': 'PaccMannRLProteinBasedGenerator', 'algorithm_version': 'v0'}
INFO:gt4sd.cli.inference:Selected algorithm support the following configuration parameters:
{
 "batch_size": {
  "description": "Batch size used for the generative model sampling.",
  "title": "Batch Size",
  "default": 32,
  "type": "integer",
  "optional": true
 },
 "temperature": {
  "description": "Temperature parameter for the softmax sampling in decoding.",
  "title": "Temperature",
  "default": 1.4,
  "type": "number",
  "optional": true
 },
 "generated_length": {
  "description": "Maximum length in tokens of the generated molcules (relates to the SMILES length).",
  "title": "Generated Length",
  "default": 100,
  "type": "integer",
  "optional": true
 }
}
Target information:
{
 "target": {
  "title": "Target protein sequence",
  "description": "AA sequence of the protein target to generate non-toxic ligands against.",
  "type": "string"
 }
}
```

### Running training pipelines via the CLI command

GT4SD provides a trainer client based on the `gt4sd-trainer` CLI command.

The trainer currently supports the following training pipelines:

- `language-modeling-trainer`: language modelling via HuggingFace transfomers and PyTorch Lightning.
- `paccmann-vae-trainer`: PaccMann VAE models.
- `granular-trainer`: multimodal compositional autoencoders supporting MLP, RNN and Transformer layers.
- `guacamol-lstm-trainer`: GuacaMol LSTM models.
- `moses-organ-trainer`: Moses Organ implementation.
- `moses-vae-trainer`: Moses VAE models.
- `torchdrug-gcpn-trainer`: TorchDrug Graph Convolutional Policy Network model.
- `torchdrug-graphaf-trainer`: TorchDrug autoregressive GraphAF model.

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

Or you can provide directly the needed parameters as arguments:

```sh
gt4sd-trainer  --training_pipeline_name language-modeling-trainer --type mlm --model_name_or_path mlm --training_file /path/to/train_file.jsonl --validation_file /path/to/valid_file.jsonl
```

To get more info on a specific training pipeleins argument simply type:

```sh
gt4sd-trainer --training_pipeline_name ${TRAINING_PIPELINE_NAME} --help
```

### Saving a trained algorithm for inference via the CLI command

Once a training pipeline has been run via the `gt4sd-trainer`, it's possible to save the trained algorithm via `gt4sd-saving` for usage in compatible inference pipelines.

Here a small example for `PaccmannGP` algorithm ([paper](https://doi.org/10.1021/acs.jcim.1c00889)).

You can train a model with `gt4sd-trainer` (quick training using few data, not really recommended for a realistic model :warning:):

```sh
gt4sd-trainer  --training_pipeline_name paccmann-vae-trainer --epochs 250 --batch_size 4 --n_layers 1 --rnn_cell_size 16 --latent_dim 16 --train_smiles_filepath src/gt4sd/training_pipelines/tests/molecules.smi --test_smiles_filepath src/gt4sd/training_pipelines/tests/molecules.smi --model_path /tmp/gt4sd-paccmann-gp/ --training_name fast-example --eval_interval 15 --save_interval 15 --selfies
```

Save the model with the compatible inference pipeline using `gt4sd-saving`:

```sh
gt4sd-saving --training_pipeline_name paccmann-vae-trainer --model_path /tmp/gt4sd-paccmann-gp/ --training_name fast-example --target_version fast-example-v0 --algorithm_application PaccMannGPGenerator
```

Run the algorithm via `gt4sd-inference` (again the model produced in the example is trained on dummy data and will give dummy outputs, do not use it as is :no_good:):

```sh
gt4sd-inference --algorithm_name PaccMannGP --algorithm_application PaccMannGPGenerator --algorithm_version fast-example-v0 --number_of_samples 5  --target '{"molwt": {"target": 60.0}}'
```

### Additional examples

Find more examples in [notebooks](./notebooks)

<!-- Having a list of all supported algorithms wouldn be nice! -->

## Supported packages

Beyond implementing various generative modeling inference and training pipelines GT4SD is designed to provide a high-level API that implement an harmonized interface for several existing packages:

- [GuacaMol](https://github.com/BenevolentAI/guacamol): inference pipelines for the baselines models and training pipelines for LSTM models.
- [Moses](https://github.com/molecularsets/moses): inference pipelines for the baselines models and training pipelines for VAEs and Organ.
- [TorchDrug](https://github.com/DeepGraphLearning/torchdrug): inference and training pipelines for GCPN and GraphAF models. Training pipelines support custom datasets as well as datasets native in TorchDrug.
- [MoLeR](https://github.com/microsoft/molecule-generation): inference pipelines for MoLeR (**MO**lecule-**LE**vel **R**epresentation) generative models for de-novo and scaffold-based generation.
- [TAPE](https://github.com/songlab-cal/tape): encoder modules compatible with the protein language models.
- [PaccMann](https://github.com/PaccMann/): inference pipelines for all algorithms of the PaccMann family as well as training pipelines for the generative VAEs.
- [transformers](https://huggingface.co/transformers): training and inference pipelines for generative models from [HuggingFace Models](https://huggingface.co/models)

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
