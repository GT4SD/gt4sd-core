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
        device: str = "cpu",
    ):
        """Initialize VAE.

        Args:
            model_path: path from where to load the model.
            model_config_path: path from where to load the model config.
            vocab_path: path from where to load the vocab.
            n_samples: number of samples to sample.
            n_batch: size of the batch.
            max_len: max length of SMILES.
            device: device used for computation. Defaults to cpu.
        """
        self.config = argparse.Namespace(
            model_load=model_path,
            config_load=model_config_path,
            vocab_load=vocab_path,
            n_samples=n_samples,
            n_batch=n_batch,
            max_len=max_len,
            device=device,
        )

    def get_generator(self) -> VaeGenerator:
        """Create an instance of the VaeGenerator.

        Returns:
            an instance of VaeGenerator.
        """
        optimiser = VaeGenerator(self.config)
        logger.debug(self.config)
        return optimiser
