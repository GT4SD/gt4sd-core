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
