# GT4SD - training GFlowNets on generic task


```{contents}
:depth: 2
```

## Overview

This notebook shows the basic usage of the GFN framework in GT4SD on a generic task. We provide an example of how to setup GFN to train on QM9 in `examples/gflownet/main_qm9.py`
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
from gt4sd.frameworks.gflownet.envs.graph_building_env import GraphBuildingEnv
from gt4sd.frameworks.gflownet.envs.mol_building_env import MolBuildingEnvContext
from gt4sd.frameworks.gflownet.tests.qm9 import QM9Dataset, QM9GapTask
from gt4sd.frameworks.gflownet.train.core import train_gflownet_main


def main():
    """Run basic GFN training on QM9."""

    hps = {"dataset": "qm9", "dataset_path": "/Users/ggi/GFN/qm9.h5", "device": "cpu"}

    dataset = QM9Dataset(hps["dataset_path"], target="gap")
    environment = GraphBuildingEnv()
    context = MolBuildingEnvContext()

    train_gflownet_main(
        configuration=hps,
        dataset=dataset,
        environment=environment,
        context=context,
        _task=QM9GapTask,
    )


if __name__ == "__main__":
    main()
```