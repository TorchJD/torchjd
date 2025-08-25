from torch import Tensor, vmap
from torch.autograd.graph import get_gradient_edge

from torchjd.autogram import Engine


def test_non_batched():
    # This is an adaptation of basic example using autogram.
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 2))

    engine = Engine(model.modules(), False)

    input = torch.randn(16, 10)  # Batch of 16 random input vectors of length 10
    target1 = torch.randn(16)  # First batch of 16 targets
    target2 = torch.randn(16)  # Second batch of 16 targets

    loss_fn = MSELoss()
    output = model(input)
    loss1 = loss_fn(output[:, 0], target1)
    loss2 = loss_fn(output[:, 1], target2)
    losses = torch.stack([loss1, loss2])

    output = losses
    grad_outputs = None

    # code of `Engine.compute_gramian`, copy-pasted and we replace all self with engine
    if output.ndim != 1:
        raise ValueError("We currently support computing the Gramian with respect to vectors only.")

    if grad_outputs is None:
        grad_outputs = torch.ones_like(output)

    engine._module_hook_manager.gramian_accumulation_phase = True

    leaf_targets = list(engine._target_edges.get_leaf_edges({get_gradient_edge(output)}, set()))

    def differentiation(_grad_output: Tensor) -> tuple[Tensor, ...]:
        return torch.autograd.grad(
            outputs=output,
            inputs=leaf_targets,
            grad_outputs=grad_outputs,
            retain_graph=True,
        )

    _ = vmap(differentiation)(torch.diag(grad_outputs))
