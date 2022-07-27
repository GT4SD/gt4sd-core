import torch

from gt4sd.frameworks.gflownet.train.trainer_qm9 import QM9GapTrainer


def test_qm9():
    """Example of how this model can be run outside of Determined"""
    config_file = {
        "lr_decay": 10000,
        "qm9_h5_path": "qm9.h5",
        "log_dir": "./",
        "num_training_steps": 100,
        "validate_every": -1,
    }
    trial = QM9GapTrainer(config_file, torch.device("cpu"))
    trial.run()


if __name__ == "__main__":
    test_qm9()
