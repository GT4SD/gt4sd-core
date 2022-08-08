from examples.gflownet.dateset_qm9 import QM9Dataset
from examples.gflownet.task_qm9 import QM9GapTask
from gt4sd.frameworks.gflownet.train.core import train_gflownet
from gt4sd.frameworks.gflownet.envs.graph_building_env import GraphBuildingEnv
from gt4sd.frameworks.gflownet.envs.mol_building_env import MolBuildingEnvContext 

def main():
    """Run basic GFN training on QM9."""

    hps = {
        "dataset": "qm9",
        "qm9_h5_path": "data/qm9.h5",
        "log_dir": "log/",
        "num_training_steps": 10000,
        "validate_every": 1000,
    }

    dataset = QM9Dataset(hps["qm9_h5_path"], train=True, target="gap")
    environment = GraphBuildingEnv()
    context = MolBuildingEnvContext(["H", "C", "N", "F", "O"], num_cond_dim=32)

    train_gflownet(
    configuration=hps,
    dataset=dataset,
    environment=environment,
    context=context,
    task=QM9GapTask,
)


if __name__ == "__main__":
    main()
