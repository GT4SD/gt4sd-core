import queue
import threading

import torch
import torch.multiprocessing as mp


class MPModelPlaceholder:
    """This class can be used as a Model in a worker process, and
    translates calls to queries to the main process"""

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

    # TODO: make a generic method for this based on __getattr__
    def logZ(self, *a):
        self._check_init()
        self.in_queue.put(("logZ", *a))
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

        Parameters
        ----------
        model: torch.nn.Module
            A torch model which lives in the main process to which method calls are passed
        num_workers: int
            Number of DataLoader workers
        cast_types: tuple
            Types that will be cast to cuda when received as arguments of method calls.
            torch.Tensor is cast by default.
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


def wrap_model_mp(model, num_workers, cast_types):
    """Construct a multiprocessing model proxy for torch DataLoaders so
    that only one process ends up making cuda calls and holding cuda
    tensors in memory.

    Parameters
    ----------
    model: torch.Module
        A torch model which lives in the main process to which method calls are passed
    num_workers: int
        Number of DataLoader workers
    cast_types: tuple
        Types that will be cast to cuda when received as arguments of method calls.
        torch.Tensor is cast by default.

    Returns
    -------
    placeholder: MPModelPlaceholder
        A placeholder model whose method calls route arguments to the main process

    """
    return MPModelProxy(model, num_workers, cast_types).placeholder
