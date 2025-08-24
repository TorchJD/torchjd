import torch
from torch import Tensor, nn, vmap
from torch.autograd.graph import get_gradient_edge
from torch.utils._pytree import PyTree, tree_map

from torchjd.autogram._edge_registry import EdgeRegistry


class backward_fn(torch.autograd.Function):
    @staticmethod
    def forward(t: Tensor) -> None:
        return None

    @staticmethod
    def setup_context(*_):
        pass

    @staticmethod
    def vmap(info, in_dims: tuple[int | None], t: Tensor) -> tuple[None, None]:
        return None, None


class test_fn(torch.autograd.Function):

    @staticmethod
    def forward(t: Tensor) -> Tensor:
        return t

    @staticmethod
    def setup_context(*_):
        pass

    @staticmethod
    def backward(ctx, t: Tensor) -> Tensor:
        backward_fn.apply(t)
        return t


def test_non_batched():
    # This is an adaptation of basic example using autogram.
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 2))
    edge_registry = EdgeRegistry()

    def module_hook(_: nn.Module, __, output: PyTree) -> PyTree:
        edge_registry.register(get_gradient_edge(output))
        return tree_map(lambda x: test_fn.apply(x), output)

    for module in model.modules():
        if any(True for _ in module.parameters(recurse=False)):
            module.register_forward_hook(module_hook)

    input = torch.randn(16, 10)  # Batch of 16 random input vectors of length 10
    target1 = torch.randn(16)  # First batch of 16 targets
    target2 = torch.randn(16)  # Second batch of 16 targets

    loss_fn = MSELoss()
    output = model(input)
    output = test_fn.apply(output)
    loss1 = loss_fn(output[:, 0], target1)
    loss2 = loss_fn(output[:, 1], target2)
    losses = torch.stack([loss1, loss2])

    leaves = list(edge_registry.get_leaf_edges({get_gradient_edge(losses)}, set()))

    def differentiation(grads):
        return torch.autograd.grad(losses, leaves, grad_outputs=grads)

    vmap(differentiation)(torch.diag(torch.ones_like(losses)))
