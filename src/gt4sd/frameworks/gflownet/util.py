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
import queue
import threading

import torch
import torch.multiprocessing as mp
import torch.nn as nn


class MPModelPlaceholder:
    """This class can be used as a Model in a worker process, and
    translates calls to queries to the main process.
    """

    def __init__(self, in_queues, out_queues):
        self.qs = in_queues, out_queues
        self.device = torch.device("cpu")
        self._is_init = False

    def _check_init(self):
        if self._is_init:
            return
        info = torch.utils.data.get_worker_info()
        self.in_queue = self.qs[0][info.id]
        self.out_queue = self.qs[1][info.id]
        self._is_init = True

    def log_z(self, *a):
        self._check_init()
        self.in_queue.put(("log_z", *a))
        return self.out_queue.get()

    def __call__(self, *a):
        self._check_init()
        self.in_queue.put(("__call__", *a))
        return self.out_queue.get()


class MPModelProxy:
    """This class maintains a reference to an in-cuda-memory model, and
    creates a `placeholder` attribute which can be safely passed to
    multiprocessing DataLoader workers.

    This placeholder model sends messages accross multiprocessing
    queues, which are received by this proxy instance, which calls the
    model and sends the return value back to the worker.

    Starts its own (daemon) thread. Always passes CPU tensors between
    processes.
    """

    def __init__(self, model: torch.nn.Module, num_workers: int, cast_types: tuple):
        """Construct a multiprocessing model proxy for torch DataLoaders.

        Args:
            model: a torch model which lives in the main process to which method calls are passed.
            num_workers: number of workers.
            cast_types: types that will be cast to cuda when received as arguments of method calls.
        """
        self.in_queues = [mp.Queue() for i in range(num_workers)]  # type: ignore
        self.out_queues = [mp.Queue() for i in range(num_workers)]  # type: ignore
        self.placeholder = MPModelPlaceholder(self.in_queues, self.out_queues)
        self.model = model
        self.device = next(model.parameters()).device
        self.cuda_types = (torch.Tensor,) + cast_types
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def __del__(self):
        self.stop.set()

    def run(self):
        while not self.stop.is_set():
            for qi, q in enumerate(self.in_queues):
                try:
                    r = q.get(True, 1e-5)
                except queue.Empty:
                    continue
                except ConnectionError:
                    break
                attr, *args = r
                f = getattr(self.model, attr)
                args = [
                    i.to(self.device) if isinstance(i, self.cuda_types) else i
                    for i in args
                ]
                result = f(*args)
                if isinstance(result, (list, tuple)):
                    msg = [
                        i.detach().to(torch.device("cpu"))
                        if isinstance(i, self.cuda_types)
                        else i
                        for i in result
                    ]
                    self.out_queues[qi].put(msg)
                else:
                    msg = (
                        result.detach().to(torch.device("cpu"))
                        if isinstance(result, self.cuda_types)
                        else result
                    )
                    self.out_queues[qi].put(msg)


def wrap_model_mp(
    model: nn.Module, num_workers: int, cast_types: tuple
) -> MPModelPlaceholder:
    """Construct a multiprocessing model proxy for torch DataLoaders so
    that only one process ends up making cuda calls and holding cuda
    tensors in memory.

    Args:
        model: a torch model which lives in the main process to which method calls are passed.
        num_workers: number of DataLoader workers.
        cast_types: types that will be cast to cuda when received as arguments of method calls.
            torch.Tensor is cast by default.

    Returns:
        placeholder: a placeholder model whose method calls route arguments to the main process.

    """
    return MPModelProxy(model, num_workers, cast_types).placeholder
