import os

import torch
from ruamel.yaml import YAML

from gt4sd.frameworks.gflownet.train.trainer_qm9 import QM9GapTrainer


def main():
    """Example of how this model can be run outside of Determined"""
    yaml = YAML(typ="safe", pure=True)
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm9.yaml")
    with open(config_file, "r") as f:
        hps = yaml.load(f)
    trial = QM9GapTrainer(hps, torch.device("cpu"))
    trial.run()


if __name__ == "__main__":
    main()
