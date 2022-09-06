# GT4SD - training GFlowNets on generic task


```{contents}
:depth: 2
```

## Overview

This notebook shows the basic usage of the GFlowNetwork (GFN) framework on a generic task. 
We provide an example of how to setup GFN to train on QM9 in `examples/gflownet/main_qm9.py`.
The implementation is adapted from: https://github.com/recursionpharma/gflownet.

The user has to define (at least) 2 main components:

* a *dataset* compatible with `GFlowNetDataset` (see `gt4sd/frameworks/gflownet/tests/qm9.py`)
* a *task* compatible with `GFlowNetTask` where defining the reward function (see `gt4sd/frameworks/gflownet/tests/qm9.py`).

Here we are assuming that:
* an *environment* compatible with `GraphBuildingEnvironment` for graph-based problems is implemented in `envs/graph_building_env.py`;
* a *context* compatible with `GraphBuildingEnvContext` to specify how to use the basic building blocks in the environment is implemented in `envs/mol_building_env.py`;
* *action* in the environment is discrete and prescribed by `GraphActionCategorical` for graph-based problems in `envs/graph_building_env.py`.


## A note on the requirements

GFN relies on `pytorch_lightning` and `pytorch_geometric`. 
We recommend training GFN on GPU and checking the pytorch_geomtric requirements for your environment.

## Debugging

Training GFN can be a long process. To debug your training pipeline, set `development=True`. This will activate `fast_dev_run` functionality in the pytorch_lightning trainer.
If training gets stuck and the dataloader does not yield data, set `num_workers=0`.

## Minimal training example

Here we provide a minimal traninng script. We implemented a dataset and task in the `examples` folder and rely on environment, context and training routines in `frameworks`.

```python
from gt4sd.frameworks.gflownet.arg_parser.parser import parse_arguments_from_config
from gt4sd.frameworks.gflownet.envs.graph_building_env import GraphBuildingEnv
from gt4sd.frameworks.gflownet.envs.mol_building_env import MolBuildingEnvContext
from gt4sd.frameworks.gflownet.tests.qm9 import QM9Dataset, QM9GapTask
from gt4sd.frameworks.gflownet.train.core import train_gflownet


def main():
    """Run basic GFN training on QM9."""

    configuration = {"dataset": "qm9", "dataset_path": "/GFN/qm9.h5", "device": "cpu"}
    # add user configuration
    configuration.update(vars(parse_arguments_from_config()))

    # build the environment and context
    environment = GraphBuildingEnv()
    context = MolBuildingEnvContext()
    # build the dataset
    dataset = QM9Dataset(configuration["dataset_path"], target="gap")
    # build the task
    task = QM9GapTask(
        configuration=configuration,
        dataset=dataset,
    )
    # train gflownet
    train_gflownet(
        configuration=configuration,
        dataset=dataset,
        environment=environment,
        context=context,
        task=task,
    )


if __name__ == "__main__":
    main()
```