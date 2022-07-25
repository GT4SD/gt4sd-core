import numpy as np
import torch
import torch.nn as nn
from rdkit import RDLogger
from torch.utils.data import Dataset, IterableDataset


class SamplingIterator(IterableDataset):
    """This class allows us to parallelise and train faster.

    By separating sampling data/the model and building torch geometric
    graphs from training the model, we can do the former in different
    processes, which is much faster since much of graph construction
    is CPU-bound.

    """

    def __init__(
        self,
        dataset: Dataset,
        model: nn.Module,
        batch_size: int,
        ctx,
        algo,
        task,
        device,
        ratio=0.5,
        stream=True,
    ):
        """Parameters
        ----------
        dataset: Dataset
            A dataset instance
        model: nn.Module
            The model we sample from (must be on CUDA already or share_memory() must be called so that
            parameters are synchronized between each worker)
        batch_size: int
            The number of trajectories, each trajectory will be comprised of many graphs, so this is
            _not_ the batch size in terms of the number of graphs (that will depend on the task)
        algo:
            The training algorithm, e.g. a TrajectoryBalance instance
        task: ConditionalTask
        ratio: float
            The ratio of offline trajectories in the batch.
        stream: bool
            If True, data is sampled iid for every batch. Otherwise, this is a normal in-order
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

    def _idx_iterator(self):
        RDLogger.DisableLog("rdApp.*")
        if self.stream:
            # If we're streaming data, just sample `offline_batch_size` indices
            while True:
                yield self.rng.integers(0, len(self.data), self.offline_batch_size)
        else:
            # Otherwise, figure out which indices correspond to this worker
            worker_info = torch.utils.data.get_worker_info()
            n = len(self.data)
            if worker_info is None:
                start, end, wid = 0, n, -1
            else:
                nw = worker_info.num_workers
                wid = worker_info.id
                start, end = int(np.floor(n / nw * wid)), int(
                    np.ceil(n / nw * (wid + 1))
                )
            bs = self.offline_batch_size
            if end - start < bs:
                yield np.arange(start, end)
                return
            for i in range(start, end - bs, bs):
                yield np.arange(i, i + bs)
            if i + bs < end:
                yield np.arange(i + bs, end)

    def __len__(self):
        if self.stream:
            return int(1e6)
        return len(self.data)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        self.rng = self.algo.rng = self.task.rng = np.random.default_rng(142857 + wid)
        self.ctx.device = self.device
        for idcs in self._idx_iterator():
            num_offline = idcs.shape[0]  # This is in [1, self.offline_batch_size]
            # Sample conditional info such as temperature, trade-off weights, etc.
            cond_info = self.task.sample_conditional_information(
                num_offline + self.online_batch_size
            )
            is_valid = torch.ones(cond_info["beta"].shape[0]).bool()

            # Sample some dataset data
            mols, flat_rewards = map(list, zip(*[self.data[i] for i in idcs]))
            flat_rewards = list(self.task.flat_reward_transform(flat_rewards))
            graphs = [self.ctx.mol_to_graph(m) for m in mols]
            trajs = self.algo.create_training_data_from_graphs(graphs)
            # Sample some on-policy data
            if self.online_batch_size > 0:
                with torch.no_grad():
                    trajs += self.algo.create_training_data_from_own_samples(
                        self.model,
                        self.online_batch_size,
                        cond_info["encoding"][num_offline:],
                    )
                if self.algo.bootstrap_own_reward:
                    # The model can be trained to predict its own reward,
                    # i.e. predict the output of cond_info_to_reward
                    pred_reward = [
                        i["reward_pred"].cpu().item() for i in trajs[num_offline:]
                    ]
                    raise ValueError("make this flat rewards")  # TODO
                else:
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
                    mols = [
                        self.ctx.graph_to_mol(trajs[i]["traj"][-1][0])
                        for i in valid_idcs
                    ]
                    # ask the task to compute their reward
                    preds, m_is_valid = self.task.compute_flat_rewards(mols)
                    # The task may decide some of the mols are invalid, we have to again filter those
                    valid_idcs = valid_idcs[m_is_valid]
                    pred_reward[valid_idcs - num_offline] = preds
                    is_valid[num_offline:] = False
                    is_valid[valid_idcs] = True
                    flat_rewards += list(pred_reward)
                    # Override the is_valid key in case the task made some mols invalid
                    for i in range(self.online_batch_size):
                        trajs[num_offline + i]["is_valid"] = is_valid[
                            num_offline + i
                        ].item()
            # Compute scalar rewards from conditional information & flat rewards
            rewards = self.task.cond_info_to_reward(cond_info, flat_rewards)
            rewards[torch.logical_not(is_valid)] = np.exp(
                self.algo.illegal_action_logreward
            )
            # Construct batch
            batch = self.algo.construct_batch(trajs, cond_info["encoding"], rewards)
            batch.num_offline = num_offline
            # TODO: There is a smarter way to do this
            # batch.pin_memory()
            yield batch
