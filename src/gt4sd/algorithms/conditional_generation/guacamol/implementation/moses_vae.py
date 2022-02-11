"""Moses VAE implementation."""

import argparse
import logging

from guacamol_baselines.moses_baselines.vae_distribution_learning import VaeGenerator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class VAE:
    def __init__(
        self,
        model_path: str,
        model_config_path: str,
        vocab_path: str,
        n_samples: int,
        n_batch: int,
        max_len: int,
    ):
        """Initialize VAE.

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

    def get_generator(self) -> VaeGenerator:
        """
        used for creating an instance of the VaeGenerator

        Returns:
            An instance of VaeGenerator
        """
        optimiser = VaeGenerator(self.config)
        logger.debug(self.config)
        return optimiser
