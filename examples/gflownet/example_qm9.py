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
