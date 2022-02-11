"""Recurrent Neural Networks with Proximal Policy Optimization algorithm implementation."""

from guacamol_baselines.smiles_lstm_ppo.goal_directed_generation import (
    PPODirectedGenerator,
)


class SMILESLSTMPPO:
    def __init__(
        self,
        model_path: str,
        num_epochs: int,
        episode_size: int,
        optimize_batch_size: int,
        entropy_weight: int,
        kl_div_weight: int,
        clip_param: float,
    ):
        """Initialize SMILESLSTMPPO.

        Args:
            model_path: path to load the model,
            num_epochs: number of epochs to sample
            episode_size: number of molecules sampled by the policy at the start of a series of ppo updates
            optimize_batch_size: batch size for the optimization
            entropy_weight: used for calculating entropy loss
            kl_div_weight: used for calculating Kullback-Leibler divergence loss
            clip_param: used for determining how far the new policy is from the old one
        """
        self.model_path = model_path
        self.num_epochs = num_epochs
        self.episode_size = episode_size
        self.optimize_batch_size = optimize_batch_size
        self.entropy_weight = entropy_weight
        self.kl_div_weight = kl_div_weight
        self.clip_param = clip_param

    def get_generator(self) -> PPODirectedGenerator:
        """
        used for creating an instance of the PPODirectedGenerator

        Returns:
            An instance of PPODirectedGenerator
        """
        optimiser = PPODirectedGenerator(
            pretrained_model_path=None,
            num_epochs=self.num_epochs,
            episode_size=self.episode_size,
            batch_size=self.optimize_batch_size,
            entropy_weight=self.entropy_weight,
            kl_div_weight=self.kl_div_weight,
            clip_param=self.clip_param,
        )
        return optimiser
