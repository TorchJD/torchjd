from typing import Callable

from torch import Node, Tensor


def get_jacobian_and_next_nodes(
    node: Node,
) -> tuple[Callable[[Tensor], tuple[Tensor, ...]], list[Node]]:
    next_functions = [fn[0] for fn in node.next_functions]
    next_functions, indices = zip(
        *[(fn, i) for i, fn in enumerate(next_functions) if fn is not None]
    )

    def jacobian(grad: Tensor) -> tuple[Tensor, ...]:
        output = [t for i, t in enumerate(node(grad)) if i in indices]
        return tuple(output)

    return jacobian, next_functions
