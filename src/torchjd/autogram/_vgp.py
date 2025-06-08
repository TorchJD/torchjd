from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn.parameter import Parameter


def vgp(
    func: Callable, *primals, has_aux: bool = False
) -> tuple[Tensor, Callable[[Tensor], Tensor]] | tuple[Tensor, Callable[[Tensor], Tensor], Any]:
    if has_aux:
        output, vjp_fn, aux = torch.func.vjp(func, *primals, has_aux=True)
    else:
        output, vjp_fn = torch.func.vjp(func, *primals, has_aux=False)
        aux = None

    if output.ndim != 1:
        raise ValueError("The function should return a vector")

    def vgp_fn(v: Tensor) -> Tensor:
        return torch.func.jvp(func, primals, tangents=vjp_fn(v))[1]

    if has_aux:
        return output, vgp_fn, aux
    else:
        return output, vgp_fn


def get_gramian(vgp_fn: Callable[[Tensor], Tensor], m: int) -> Tensor:
    identity = torch.eye(m)
    gramian = torch.func.vmap(vgp_fn)(identity)

    return gramian


def get_output_and_gramian(func: Callable, *primals) -> tuple[Tensor, Tensor]:
    output, vgp_fn = vgp(func, *primals)
    gramian = get_gramian(vgp_fn, output.shape[0])

    return output, gramian


def vgp_from_module_1(module: nn.Module, *inputs) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
    def functional_model_call(*primals) -> Tensor:
        params_dict = {
            key: primal for key, primal in zip(dict(module.named_parameters()).keys(), primals)
        }
        all_state = {**params_dict, **dict(module.named_buffers())}
        return functional_call(module, all_state, *inputs)

    return vgp(functional_model_call, *module.parameters())


def vgp_from_module_2(module: nn.Module, *inputs) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
    def functional_model_call_v2(primals: dict[str, Parameter]) -> Tensor:
        all_state = {**primals, **dict(module.named_buffers())}
        return functional_call(module, all_state, *inputs)

    return vgp(functional_model_call_v2, dict(module.named_parameters()))
