#
# MIT License
#
# Copyright (c) 2022 GT4SD team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
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
            model_path: path to load the model.
            num_epochs: number of epochs to sample.
            episode_size: number of molecules sampled by the policy at the start of a series of ppo updates.
            optimize_batch_size: batch size for the optimization.
            entropy_weight: used for calculating entropy loss.
            kl_div_weight: used for calculating Kullback-Leibler divergence loss.
            clip_param: used for determining how far the new policy is from the old one.
        """
        self.model_path = model_path
        self.num_epochs = num_epochs
        self.episode_size = episode_size
        self.optimize_batch_size = optimize_batch_size
        self.entropy_weight = entropy_weight
        self.kl_div_weight = kl_div_weight
        self.clip_param = clip_param

    def get_generator(self) -> PPODirectedGenerator:
        """Create an instance of the PPODirectedGenerator.

        Returns:
            an instance of PPODirectedGenerator.
        """
        optimiser = PPODirectedGenerator(
            pretrained_model_path=self.model_path,
            num_epochs=self.num_epochs,
            episode_size=self.episode_size,
            batch_size=self.optimize_batch_size,
            entropy_weight=self.entropy_weight,
            kl_div_weight=self.kl_div_weight,
            clip_param=self.clip_param,
        )
        return optimiser
