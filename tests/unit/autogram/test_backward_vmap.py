from torch import vmap
from torch.autograd.graph import get_gradient_edge

from torchjd.autogram._edge_registry import EdgeRegistry
from torchjd.autogram._gramian_accumulator import GramianAccumulator
from torchjd.autogram._module_hook_manager import ModuleHookManager


def test_non_batched():
    # This is an adaptation of basic example using autogram.
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 2))
    edge_registry = EdgeRegistry()
    gramian_accumulator = GramianAccumulator()

    module_hook_manager = ModuleHookManager(edge_registry, gramian_accumulator)

    for module in model.modules():
        if any(True for _ in module.parameters(recurse=False)):
            module_hook_manager.hook_module(module)

    input = torch.randn(16, 10)  # Batch of 16 random input vectors of length 10
    target1 = torch.randn(16)  # First batch of 16 targets
    target2 = torch.randn(16)  # Second batch of 16 targets

    loss_fn = MSELoss()
    output = model(input)
    loss1 = loss_fn(output[:, 0], target1)
    loss2 = loss_fn(output[:, 1], target2)
    losses = torch.stack([loss1, loss2])

    module_hook_manager.gramian_accumulation_phase = True

    leaves = list(edge_registry.get_leaf_edges({get_gradient_edge(losses)}, set()))

    def differentiation(grads):
        return torch.autograd.grad(losses, leaves, grad_outputs=grads)

    vmap(differentiation)(torch.diag(torch.ones_like(losses)))
