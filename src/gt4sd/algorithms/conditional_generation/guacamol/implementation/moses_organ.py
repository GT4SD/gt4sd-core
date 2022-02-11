"""Moses Organ implementation."""

import argparse

from guacamol_baselines.moses_baselines.organ_distribution_learning import (
    OrganGenerator,
)


class Organ:
    def __init__(
        self,
        model_path: str,
        model_config_path: str,
        vocab_path: str,
        n_samples: int,
        n_batch: int,
        max_len: int,
    ):
        """Initialize Organ.

        Args:
            model_path: path from where to load the model
            model_config_path: path from where to load the model config
            vocab_path: path from where to load the vocab
            n_samples: Number of samples to sample
            n_batch: Size of the batch
            max_len: Max length of SMILES
        """
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--model_load", default=model_path)
        self.parser.add_argument("--config_load", default=model_config_path)
        self.parser.add_argument("--vocab_load", default=vocab_path)
        self.parser.add_argument("--n_samples", default=n_samples)
        self.parser.add_argument("--n_batch", default=n_batch)
        self.parser.add_argument("--max_len", default=max_len)
        self.parser.add_argument("--device", default="cpu")
        self.config = self.parser.parse_known_args()[0]

    def get_generator(self) -> OrganGenerator:
        """
        used for creating an instance of the OrganGenerator

        Returns:
            An instance of OrganGenerator
        """
        optimiser = OrganGenerator(self.config)
        return optimiser
