import torch
from torch import Tensor, nn, vmap


def compute_gramian_with_autograd(
    output: Tensor, inputs: list[nn.Parameter], retain_graph: bool = False
) -> Tensor:
    """
    Computes the Gramian of the Jacobian of the outputs with respect to the inputs using vmapped
    calls to the autograd engine.
    """

    filtered_inputs = [input for input in inputs if input.requires_grad]

    def get_vjp(grad_outputs: Tensor) -> list[Tensor]:
        grads = torch.autograd.grad(
            output,
            filtered_inputs,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        return [grad for grad in grads if grad is not None]

    jacobians = vmap(get_vjp)(torch.diag(torch.ones_like(output)))
    jacobian_matrices = [jacobian.reshape([jacobian.shape[0], -1]) for jacobian in jacobians]
    gramian = torch.sum(torch.stack([jacobian @ jacobian.T for jacobian in jacobian_matrices]))

    return gramian
