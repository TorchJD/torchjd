import time

import torch
from torch import Tensor


def noop():
    pass


def time_call(fn, init_fn=noop, pre_fn=noop, post_fn=noop, n_runs: int = 10) -> Tensor:
    init_fn()

    times = []
    for _ in range(n_runs):
        pre_fn()
        start = time.perf_counter()
        fn()
        post_fn()
        elapsed_time = time.perf_counter() - start
        times.append(elapsed_time)

    return torch.tensor(times)


def print_times(name: str, times: Tensor) -> None:
    print(f"{name} times (avg = {times.mean():.5f}, std = {times.std():.5f}")
    print(times)
    print()
