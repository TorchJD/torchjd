import torch
from contexttimer import Timer
from torch import Tensor

from torchjd.aggregation.mgda import MGDAWeighting

_SHAPES = [
    (1, 1),
    (1, 10),
    (1, 100),
    (1, 1_000),
    (1, 10_000),
    (10, 1),
    (10, 10),
    (10, 100),
    (10, 1_000),
    (10, 10_000),
    (10, 100_000),
    (10, 1_000_000),
    (10, 10_000_000),
    (10, 100_000_000),
    (100, 1),
    (100, 10),
    (100, 100),
    (100, 1_000),
    (100, 10_000),
    (1_000, 1),
    (1_000, 10),
    (1_000, 100),
    (1_000, 1_000),
    (1_000, 10_000),
    (10_000, 1),
    (10_000, 10),
    (10_000, 100),
    (10_000, 1_000),
    (10_000, 10_000),
]


def get_computation_milliseconds(func: callable, matrix: Tensor) -> float:
    with Timer(factor=1000) as t:
        func(matrix)

    return t.elapsed


def main():
    mgda = MGDAWeighting()
    func = mgda.forward

    for n_rows, n_cols in _SHAPES:
        matrix = torch.randn(n_rows, n_cols)
        time = get_computation_milliseconds(func, matrix)
        print(f"{n_rows}x{n_cols} - {time:.3f}ms")


if __name__ == "__main__":
    main()
