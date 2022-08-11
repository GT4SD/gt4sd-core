from examples.gflownet.dateset_qm9 import QM9Dataset
from examples.gflownet.task_qm9 import QM9GapTask
from gt4sd.frameworks.gflownet.train.core import train_gflownet_main
from gt4sd.frameworks.gflownet.envs.graph_building_env import GraphBuildingEnv
from gt4sd.frameworks.gflownet.envs.mol_building_env import MolBuildingEnvContext


def main():
    """Run basic GFN training on QM9."""

    hps = {"dataset": "qm9", "dataset_path": "/Users/ggi/GFN/qm9.h5"}
    # data
    dataset = QM9Dataset(hps["dataset_path"], train=True, target="gap")
    # graph
    environment = GraphBuildingEnv()
    # specify how to build the graph
    context = MolBuildingEnvContext(atoms=["H", "C", "N", "F", "O"], num_cond_dim=32)

    train_gflownet_main(
        configuration=hps,
        dataset=dataset,
        environment=environment,
        context=context,
        _task=QM9GapTask,
    )


if __name__ == "__main__":
    main()
