from torch import nn


def get_used_params(module: nn.Module) -> tuple[dict[str, nn.Parameter], dict[str, nn.Parameter]]:
    """
    Gets all parameters that a module uses. In reality, we return all direct params (which may
    include some unused params) and all the indirectly used params that we know about (we may be
    missing some in weird modules).

    Returns the tuple containing the params that require grad and the params that don't require
    grad.
    """

    direct_rg_params, direct_frozen_params = _get_direct_params(module)
    indirect_rg_params, indirect_frozen_params = _get_indirectly_used_params(module)
    rg_params = direct_rg_params | indirect_rg_params
    frozen_params = direct_frozen_params | indirect_frozen_params

    return rg_params, frozen_params


def _get_direct_params(
    module: nn.Module, prefix: str = ""
) -> tuple[dict[str, nn.Parameter], dict[str, nn.Parameter]]:
    rg_params = dict[str, nn.Parameter]()
    frozen_params = dict[str, nn.Parameter]()

    for name, param in module.named_parameters(recurse=False):
        if param.requires_grad:
            rg_params[prefix + name] = param
        else:
            frozen_params[prefix + name] = param

    return rg_params, frozen_params


def _get_indirectly_used_params(
    module: nn.Module,
) -> tuple[dict[str, nn.Parameter], dict[str, nn.Parameter]]:
    # MHA uses its out_proj child params itself. Note that we also check that the MHA still has
    # an out_proj attribute because it might change in the future (which will remove the
    # necessity of custom code for MHA entirely). See the status of
    # https://github.com/pytorch/pytorch/pull/126568
    if isinstance(module, nn.MultiheadAttention) and hasattr(module, "out_proj"):
        return _get_direct_params(module.out_proj, prefix="out_proj.")

    return {}, {}
