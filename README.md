# GT4SD (Generative Toolkit for Scientific Discovery)

[![PyPI version](https://badge.fury.io/py/gt4sd.svg)](https://badge.fury.io/py/gt4sd)
[![Actions tests](https://github.com/gt4sd/gt4sd-core/actions/workflows/tests.yaml/badge.svg)](https://github.com/gt4sd/gt4sd-core/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions](https://img.shields.io/badge/contributions-welcome-blue)](https://github.com/GT4SD/gt4sd-core/blob/main/CONTRIBUTING.md)
[![Docs](https://img.shields.io/badge/website-live-brightgreen)](https://gt4sd.github.io/gt4sd-core/)
[![Total downloads](https://pepy.tech/badge/gt4sd)](https://pepy.tech/project/gt4sd)
[![Monthly downloads](https://pepy.tech/badge/gt4sd/month)](https://pepy.tech/project/gt4sd)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GT4SD/gt4sd-core/main)
[![DOI](https://zenodo.org/badge/458309249.svg)](https://zenodo.org/badge/latestdoi/458309249)
[![2022 IEEE Open Software Services Award](https://img.shields.io/badge/Award-2022%20IEEE%20Open%20Software%20Services%20Award-yellow)](https://conferences.computer.org/services/2022/awards/oss_award.html)
[![Paper DOI: 10.1038/s41524-023-01028-1](https://zenodo.org/badge/DOI/10.1038/s41524-023-01028-1.svg)](https://www.nature.com/articles/s41524-023-01028-1)

<img src="./docs/_static/gt4sd_graphical_abstract.png" alt="logo" width="800">


The **GT4SD** (Generative Toolkit for Scientific Discovery) is an open-source platform to accelerate hypothesis generation in the scientific discovery process. It provides a library for making state-of-the-art generative AI models easier to use.

For full details on the library API and examples see the [docs](https://gt4sd.github.io/gt4sd-core/).
Almost all pretrained models are also available via `gradio`-powered [web apps](https://huggingface.co/GT4SD) on Hugging Face Spaces.

This branch contains a minimal version which supports only the [Regression Transformers](https://github.com/IBM/regression-transformer/): training and inference pipelines to generate small molecules, polymers or peptides based on numerical property constraints. For details [read the paper](https://www.nature.com/articles/s42256-023-00639-z).

## Installation

```sh
git clone https://github.com/GT4SD/gt4sd-core.git -b rt-minimal
cd gt4sd-core/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install .
# for development
# pip install -r dev_requirements.txt
# pip install -e .
```

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

@article{manica2022gt4sd,
  title={Accelerating material design with the generative toolkit for scientific discovery},
  author={Manica, Matteo and Born, Jannis and Cadow, Joris and Christofidellis, Dimitrios and Dave, Ashish and Clarke, Dean and Teukam, Yves Gaetan Nana and Giannone, Giorgio and Hoffman, Samuel C and Buchan, Matthew and others},
  journal={npj Computational Materials},
  volume={9},
  number={1},
  pages={69},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

## License

The `gt4sd` codebase is under MIT license.
For individual model usage, please refer to the model licenses found in the original packages.
