# IBM Molecule Generation Experience (Community Version)


IBM Molecule Generation Experience (MolGX) is a tool to accelerate an AI-driven design of new materials. 
This is the Community Version which implements a small yet essential subset of its capabilities selected 
from the Enterprise Version. 
With the Community Version, we intend to share our important technologies with a wide range of communities
as well as to further improve these technologies through a collaborative, open development. 

## Requirements

MolGX runs with the following versions of Python and pip:

1. Python >=3.7, <3.8

2. pip>=19.1, <20.3


This restriction intends to be consistent with [gt4sd](https://github.com/GT4SD/gt4sd-core). 

## Installation

We recommend to create a conda environment such as:

```CommandLine
conda create -n molgx_env python=3.7 anaconda
```

Then, for Windows tupe the following command:

```CommandLine
activate molgx_env # for windows
```

For the other environments such as Linux/MacOS:

```CommandLine
conda activate molgx_env # for the others
```

There are two ways to install MolGX:

Type the following command if you want to install MolGX from [PyPI](https://pypi.org/): 

```CommandLine
pip install molgx
```

Type the following commands if you want to clone the source code to install it: 

```CommandLine
git clone https://github.com/GT4SD/gt4sd-core.git
cd ./src/gt4sd/frameworks/molgx
pip install .
```

## Running MolGX

At present, there are two ways to run MolGX. One is to use it as a standalone application that allows to use its full capabilities. 
The other is to use a pretrained model under GT4SD, which plans to be extended to support more capabilities. 

##  Running an example on jupyter notebook as a standalone application

[Here](/example/jupyter_notebook/MolGX_tutorial.ipynb) is an example on giving an overview of the usage of MolGX. 
You will need to install the Jupyter Notebook to run the example.
One way is to install the Jupyter Notebook is: 

```CommandLine
conda install jupyter notebook
```

Then, you will be able to invoke it with ```jupyter-notebook```.

## Communicating with GT4SD

A pre-trained model for 10 QM9 samples with target propetries homo and lumo is along with GT4SD core algorithms. Running the algorithm is as easy as typing:

```CommandLine
from gt4sd.algorithms.conditional_generation.molgx.core import MolGX, MolGXQM9Generator

import logging
logging.disable(logging.INFO)

configuration = MolGXQM9Generator()
algorithm = MolGX(configuration=configuration)
items = list(algorithm.sample(3))
print(items)
```

See this [example](/example/jupyter_notebook/gt4sd_molgx_example.ipynb). 

## Building a documentation

You will need [Sphinx](https://www.sphinx-doc.org/en/master/index.html). You can install it with Anaconda as follows:

```CommandLine
conda install sphinx
```

Type the following command to generate a document: 

```CommandLine
cd ./docs
make html
```

You will then find the html files under `docs/_build/html` and open `index.html` with your web browsewr. 

## For developers

Type the following command after activating your conda environment:

```CommandLine
pip install -e .
```

## Miscellaneous

The web application of MolGX is available [here](https://molgx.draco.res.ibm.com/). 

Additionally, the following papers describe some of the essential algorithms implemented in the Community Version as well as the other techniques not implemented here: 

1. Seiji Takeda, Toshiyuki Hama, Hsiang-Han Hsu, Akihiro Kishimoto, Makoto Kogoh, Takumi Hongo, Kumiko Fujieda, Hideaki Nakashika, Dmitry Zubarev, Daniel P. Sanders, Jed W. Pitera, Junta Fuchiwaki, Daiju Nakano. 
[Molecule Generation Experience: An Open Platform of Material Design for Public Users](https://arxiv.org/abs/2108.03044). CoRR abs/2108.03044, 2021. 

2. Seiji Takeda, Toshiyuki Hama, Hsiang-Han Hsu, Victoria A. Piunova, Dmitry Zubarev, Daniel P. Sanders, Jed W. Pitera, Makoto Kogoh, Takumi Hongo, Yenwei Cheng, Wolf Bocanett, Hideaki Nakashika, Akihiro Fujita, Yuta Tsuchiya, Katsuhiko Hino, Kentaro Yano, Shuichi Hirose, Hiroki Toda, Yasumitsu Orii, Daiju Nakano. 
[Molecular Inverse-Design Platform for Material Industries](https://arxiv.org/abs/2004.11521). pages 2961-2969, KDD 2020. 

Finally, we use some of the data extracted from the [QM9 database](http://quantum-machine.org/) with the following references:

1. L. Ruddigkeit, R. van Deursen, L. C. Blum, J.-L. Reymond, Enumeration of 166 billion organic small molecules in the chemical universe database GDB-17, J. Chem. Inf. Model. 52, 2864–2875, 2012.
2. R. Ramakrishnan, P. O. Dral, M. Rupp, O. A. von Lilienfeld, Quantum chemistry structures and properties of 134 kilo molecules, Scientific Data 1, 140022, 2014.
