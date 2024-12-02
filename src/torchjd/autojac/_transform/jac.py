from itertools import accumulate
from typing import Iterable, Sequence

import torch
from torch import Size, Tensor

from ._differentiate import _Differentiate
from ._utils import _materialize
from .tensor_dict import Jacobians


class Jac(_Differentiate[Jacobians]):
    def __init__(
        self,
        outputs: Iterable[Tensor],
        inputs: Iterable[Tensor],
        chunk_size: int | None,
        retain_graph: bool = False,
        create_graph: bool = False,
    ):
        super().__init__(outputs, inputs, retain_graph, create_graph)
        self.chunk_size = chunk_size

    def _differentiate(self, jac_outputs: Sequence[Tensor]) -> tuple[Tensor, ...]:
        """
        Computes the jacobian of each output with respect to each input, and applies the linear
        transformations represented by the jac_outputs to the results.

        Returns one jacobian per input. The ith jacobian will be of shape
        ``(m,) + inputs[i].shape``, where ``m`` is the first dimension of the jac_outputs.

        :param jac_outputs: The sequence of tensors indicating how to scale the obtained jacobians.
            Its length should be equal to the length of ``outputs``. ``jac_outputs[i]`` should be of
            shape ``(m,) + outputs[i].shape``, where ``m`` is any value constant over the
            jac_outputs.
        """

        outputs = list(self.outputs)
        inputs = list(self.inputs)

        if len(inputs) == 0:
            return tuple()

        n_outputs = len(outputs)
        if len(jac_outputs) != n_outputs:
            raise ValueError(
                "Parameters `outputs` and `jac_outputs` should be sequences of the same length."
                f"Found `len(outputs) = {n_outputs}` and `len(jac_outputs) = {len(jac_outputs)}`."
            )

        if n_outputs == 0:
            return tuple(
                [
                    torch.empty((0,) + input.shape, device=input.device, dtype=input.dtype)
                    for input in inputs
                ]
            )

        def get_vjp(grad_outputs: Sequence[Tensor]) -> Tensor:
            optional_grads = torch.autograd.grad(
                outputs,
                inputs,
                grad_outputs=grad_outputs,
                retain_graph=self.retain_graph,
                create_graph=self.create_graph,
                allow_unused=True,
            )
            grads = _materialize(optional_grads, inputs=inputs)
            return torch.concatenate([grad.reshape([-1]) for grad in grads])

        # Because of a limitation of vmap, this breaks when some tensors have `retains_grad=True`.
        # See https://pytorch.org/functorch/stable/ux_limitations.html for more information.
        # This also breaks when some tensors have been produced by compiled functions.
        grouped_jacobian_matrix = torch.vmap(get_vjp, chunk_size=self.chunk_size)(jac_outputs)

        lengths = [input.numel() for input in inputs]
        jacobian_matrices = _extract_sub_matrices(grouped_jacobian_matrix, lengths)

        shapes = [input.shape for input in inputs]
        jacobians = _reshape_matrices(jacobian_matrices, shapes)

        return tuple(jacobians)


def _extract_sub_matrices(matrix: Tensor, lengths: Sequence[int]) -> list[Tensor]:
    cumulative_lengths = [*accumulate(lengths)]

    if cumulative_lengths[-1] != matrix.shape[1]:
        raise ValueError(
            "The sum of the provided lengths should be equal to the number of columns in the "
            "provided matrix."
        )

    start_indices = [0] + cumulative_lengths[:-1]
    end_indices = cumulative_lengths
    return [matrix[:, start:end] for start, end in zip(start_indices, end_indices)]


def _reshape_matrices(matrices: Sequence[Tensor], shapes: Sequence[Size]) -> Sequence[Tensor]:
    if len(matrices) != len(shapes):
        raise ValueError(
            "Parameters `matrices` and `shapes` should contain the same number of elements."
        )

    return [matrix.view((matrix.shape[0],) + shape) for matrix, shape in zip(matrices, shapes)]
