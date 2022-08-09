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
import logging
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset

from ..envs.graph_building_env import GraphBuildingEnvContext
from ..loss.trajectory_balance import TrajectoryBalance

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SamplingIterator(IterableDataset):
    """This class allows us to parallelise and train faster.
    By separating sampling data/the model and building torch geometric
    graphs from training the model, we can do the former in different
    processes, which is much faster since much of graph construction
    is CPU-bound. This sampler can handle offline and online data.
    """

    def __init__(
        self,
        dataset: Dataset,
        model: nn.Module,
        batch_size: int,
        ctx: GraphBuildingEnvContext,
        algo: TrajectoryBalance,
        task: nn.Module,
        device: str = "cuda",
        ratio: float = 0.5,
        stream: bool = True,
    ) -> None:
        """
        Args:
            dataset: a dataset instance.
            model: the model we sample from (must be on CUDA already or share_memory() must be called so that
                parameters are synchronized between each worker).
            batch_size: the number of trajectories, each trajectory will be comprised of many graphs, so this is
                _not_ the batch size in terms of the number of graphs (that will depend on the task).
            ctx: the graph environment.
            algo: the training algorithm, e.g. a TrajectoryBalance instance.
            task: ConditionalTask
            ratio: the ratio of offline trajectories in the batch.
            stream: if true, data is sampled iid for every batch. Otherwise, this is a normal in-order
                dataset iterator.
        """
        self.data = dataset
        self.model = model
        self.batch_size = batch_size
        self.offline_batch_size = int(np.ceil(batch_size * ratio))
        self.online_batch_size = int(np.floor(batch_size * (1 - ratio)))
        self.ratio = ratio
        self.ctx = ctx
        self.algo = algo
        self.task = task
        self.device = device
        self.stream = stream

    def _idx_iterator(self) -> torch.Tensor:
        """Returns an iterator over the indices of the dataset. The batch can be offline or online.

        Yields:
            Batch of indexes.
        """
        bs = self.offline_batch_size
        if self.stream:
            # If we're streaming data, just sample `offline_batch_size` indices
            # CHECK: shouldn't this be online_batch_size?
            while True:
                yield self.rng.integers(0, len(self.data), bs)
        else:
            # Otherwise, figure out which indices correspond to this worker
            worker_info = torch.utils.data.get_worker_info()
            n = len(self.data)

            if worker_info is None:
                start = 0
                end = n
                wid = -1
            else:
                nw = worker_info.num_workers
                wid = worker_info.id
                start = int(np.floor(n / nw * wid))
                end = int(np.ceil(n / nw * (wid + 1)))

            if end - start < bs:
                yield np.arange(start, end)
                return
            for i in range(start, end - bs, bs):
                yield np.arange(i, i + bs)
            if i + bs < end:
                yield np.arange(i + bs, end)

    def __len__(self):
        # if online
        if self.stream:
            return int(1e6)
        # if offline
        return len(self.data)

    def sample_offline(self, idcs):
        """Samples offline data.

        Args:
            idcs: the indices of the data to sample.

        Returns:
            trajs: the trajectories.
            rewards: the rewards.
        """
        # Sample offline batch (mols, rewards)
        mols, flat_rewards = map(list, zip(*[self.data[i] for i in idcs]))
        # rewards
        flat_rewards = list(self.task.flat_reward_transform(flat_rewards))
        # build graphs
        graphs = [self.ctx.mol_to_graph(m) for m in mols]
        # use trajectory balance to sample trajectories
        trajs = self.algo.create_training_data_from_graphs(graphs)
        return trajs, flat_rewards

    def predict_reward_model(self, trajs, flat_rewards, num_offline):
        """Predict rewards using the model.

        Args:
            trajs: the trajectories.
            flat_rewards: the rewards.
            num_offline: the number of offline trajectories.

        Returns:
            flat_rewards: the updated rewards.
        """
        # The model can be trained to predict its own reward,
        # i.e. predict the output of cond_info_to_reward
        pred_reward = [i["reward_pred"].cpu().item() for i in trajs[num_offline:]]
        flat_rewards += list(pred_reward)
        raise ValueError("make this flat rewards")  # TODO
        return flat_rewards

    def predict_reward_task(self, trajs, flat_rewards, num_offline, is_valid):
        """Predict rewards using the task.

        Args:
            trajs: the trajectories.
            flat_rewards: the rewards.
            num_offline: the number of offline trajectories.
            is_valid: whether the trajectories are valid.

        Returns:
            flat_rewards: the updated rewards.
        """
        # Otherwise, query the task for flat rewards
        valid_idcs = torch.tensor(
            [
                i + num_offline
                for i in range(self.online_batch_size)
                if trajs[i + num_offline]["is_valid"]
            ]
        ).long()

        pred_reward = torch.zeros((self.online_batch_size))
        # fetch the valid trajectories endpoints
        mols = [self.ctx.graph_to_mol(trajs[i]["traj"][-1][0]) for i in valid_idcs]

        # ask the task to compute their reward
        preds, m_is_valid = self.task.compute_flat_rewards(mols)
        # The task may decide some of the mols are invalid, we have to again filter those
        valid_idcs = valid_idcs[m_is_valid]
        _preds = torch.tensor(preds, dtype=torch.float32)
        pred_reward[valid_idcs - num_offline] = _preds

        is_valid[num_offline:] = False
        is_valid[valid_idcs] = True
        flat_rewards += list(pred_reward)

        # Override the is_valid key in case the task made some mols invalid
        for i in range(self.online_batch_size):
            trajs[num_offline + i]["is_valid"] = is_valid[num_offline + i].item()

        return trajs, flat_rewards

    def sample_online(self, trajs, flat_rewards, cond_info, num_offline):
        """Sample on-policy data.

        Args:
            trajs: the trajectories.
            flat_rewards: the rewards.
            cond_info: the conditional information.
            num_offline: the number of offline trajectories.

        Returns:
            trajs: the updated trajectories.
            flat_rewards: the updated rewards.
        """
        is_valid = torch.ones(cond_info["beta"].shape[0]).bool()

        with torch.no_grad():
            trajs += self.algo.create_training_data_from_own_samples(
                self.model,
                self.online_batch_size,
                cond_info["encoding"][num_offline:],
            )

        # predict reward with model
        if self.algo.bootstrap_own_reward:
            # TODO: fix this
            flat_rewards = self.predict_reward_model(trajs, flat_rewards, num_offline)

        # predict reward with task
        else:
            trajs, flat_rewards = self.predict_reward_task(
                trajs, flat_rewards, num_offline, is_valid
            )

        return trajs, flat_rewards

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Build batch using online and offline data."""
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        # set seed
        seed = np.random.default_rng(142857 + wid)
        self.rng = seed
        self.algo.rng = seed
        self.task.rng = seed

        self.ctx.device = self.device

        # iterate over the indices in the batch
        for idcs in self._idx_iterator():

            num_offline = idcs.shape[0]  # This is in [1, self.offline_batch_size]
            # Sample conditional info such as temperature, trade-off weights, etc.
            cond_info = self.task.sample_conditional_information(
                num_offline + self.online_batch_size
            )
            is_valid = torch.ones(cond_info["beta"].shape[0]).bool()

            # sample offline data
            trajs, flat_rewards = self.sample_offline(idcs)

            # Sample some on-policy data (sample online the model or the task)
            if self.online_batch_size > 0:
                # update trajectories and rewards with on-policy data
                trajs, flat_rewards = self.sample_online(
                    idcs, trajs, flat_rewards, cond_info, num_offline
                )

            # compute scalar rewards from conditional information & flat rewards
            rewards = self.task.cond_info_to_reward(cond_info, flat_rewards)
            # account for illegal actions
            rewards[torch.logical_not(is_valid)] = np.exp(
                self.algo.illegal_action_logreward
            )

            # Construct batch using trajectories, rewards
            batch = self.algo.construct_batch(trajs, cond_info["encoding"], rewards)
            batch.num_offline = num_offline
            yield batch
