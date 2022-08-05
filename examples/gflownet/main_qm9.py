from examples.gflownet.module_qm9 import QM9Module
from gt4sd.frameworks.gflownet.dataloader.data_module import GFlowNetDataModule
from gt4sd.frameworks.gflownet.train.core import train_gflownet


def main():
    """Run basic GFN training on QM9."""

    hps = {
        "dataset": "qm9",
        "qm9_h5_path": "data/qm9.h5",
        "log_dir": "log/",
        "num_training_steps": 10000,
        "validate_every": 1000,
    }

    train_gflownet(hps, QM9Module, GFlowNetDataModule)


if __name__ == "__main__":
    main()
