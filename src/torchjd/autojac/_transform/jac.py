import math
from functools import partial
from itertools import accumulate
from typing import Callable, Sequence

import torch
from torch import Size, Tensor

from ._differentiate import Differentiate
from ._materialize import materialize
from .ordered_set import OrderedSet
from .tensor_dict import Jacobians


class Jac(Differentiate[Jacobians]):
    def __init__(
        self,
        outputs: OrderedSet[Tensor],
        inputs: OrderedSet[Tensor],
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

        if len(self.inputs) == 0:
            return tuple()

        if len(self.outputs) == 0:
            return tuple(
                [
                    torch.empty((0,) + input.shape, device=input.device, dtype=input.dtype)
                    for input in self.inputs
                ]
            )

        def _get_vjp(grad_outputs: Sequence[Tensor], retain_graph: bool) -> Tensor:
            optional_grads = torch.autograd.grad(
                self.outputs,
                self.inputs,
                grad_outputs=grad_outputs,
                retain_graph=retain_graph,
                create_graph=self.create_graph,
                allow_unused=True,
            )
            grads = materialize(optional_grads, inputs=self.inputs)
            return torch.concatenate([grad.reshape([-1]) for grad in grads])

        # By the Jacobians constraint, this value should be the same for all jac_outputs.
        m = jac_outputs[0].shape[0]
        max_chunk_size = self.chunk_size if self.chunk_size is not None else m
        n_chunks = math.ceil(m / max_chunk_size)

        # List of tensors of shape [k_i, n] where the k_i's sum to m
        jac_matrix_chunks = []

        # First differentiations: always retain graph
        get_vjp_retain = partial(_get_vjp, retain_graph=True)
        for i in range(n_chunks - 1):
            start = i * max_chunk_size
            end = (i + 1) * max_chunk_size
            jac_outputs_chunk = [jac_output[start:end] for jac_output in jac_outputs]
            jac_matrix_chunks.append(_get_jac_matrix_chunk(jac_outputs_chunk, get_vjp_retain))

        # Last differentiation: retain the graph only if self.retain_graph==True
        get_vjp_last = partial(_get_vjp, retain_graph=self.retain_graph)
        start = (n_chunks - 1) * max_chunk_size
        jac_outputs_chunk = [jac_output[start:] for jac_output in jac_outputs]
        jac_matrix_chunks.append(_get_jac_matrix_chunk(jac_outputs_chunk, get_vjp_last))

        jac_matrix = torch.vstack(jac_matrix_chunks)
        lengths = [input.numel() for input in self.inputs]
        jac_matrices = _extract_sub_matrices(jac_matrix, lengths)

        shapes = [input.shape for input in self.inputs]
        jacs = _reshape_matrices(jac_matrices, shapes)

        return tuple(jacs)


def _get_jac_matrix_chunk(
    jac_outputs_chunk: list[Tensor], get_vjp: Callable[[Sequence[Tensor]], Tensor]
) -> Tensor:
    """
    Computes the jacobian matrix chunk corresponding to the provided get_vjp function, either by
    calling get_vjp directly or by wrapping it into a call to ``torch.vmap``, depending on the shape
    of the provided ``jac_outputs_chunk``. Because of the numerous issues of vmap, we use it only if
    necessary (i.e. when the ``jac_outputs_chunk`` have more than 1 row).
    """

    chunk_size = jac_outputs_chunk[0].shape[0]
    if chunk_size == 1:
        grad_outputs = [tensor.squeeze(0) for tensor in jac_outputs_chunk]
        gradient_vector = get_vjp(grad_outputs)
        return gradient_vector.unsqueeze(0)
    else:
        return torch.vmap(get_vjp, chunk_size=chunk_size)(jac_outputs_chunk)


def _extract_sub_matrices(matrix: Tensor, lengths: Sequence[int]) -> list[Tensor]:
    cumulative_lengths = [*accumulate(lengths)]
    start_indices = [0] + cumulative_lengths[:-1]
    end_indices = cumulative_lengths
    return [matrix[:, start:end] for start, end in zip(start_indices, end_indices)]


def _reshape_matrices(matrices: Sequence[Tensor], shapes: Sequence[Size]) -> Sequence[Tensor]:
    return [matrix.view((matrix.shape[0],) + shape) for matrix, shape in zip(matrices, shapes)]
