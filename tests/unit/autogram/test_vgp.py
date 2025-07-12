import torch
from pytest import mark
from torch import Tensor, nn
from torch.nn import Linear, ReLU, Sequential
from torch.testing import assert_close
from torch.utils._ordered_set import OrderedSet

from torchjd._autogram._vgp import (
    get_gramian,
    get_output_and_gramian,
    vgp_from_module_1,
    vgp_from_module_2,
)
from torchjd._autojac._transform import Diagonalize, Init, Jac
from torchjd._autojac._transform._aggregate import _Matrixify
from torchjd.aggregation._weighting_bases import PSDMatrix


def test_vgp():
    x = torch.tensor([1.0, 2.0, 3.0])

    def f(x: Tensor) -> Tensor:
        return torch.concatenate([x.sum().unsqueeze(0), (x**2).sum().unsqueeze(0)])

    y, gramian = get_output_and_gramian(f, x)

    expected_jacobian = torch.tensor([[1.0, 1.0, 1.0], [2.0, 4.0, 6.0]])
    expected_gramian = expected_jacobian @ expected_jacobian.T

    assert_close(gramian, expected_gramian)


def test_vgp_2():
    x1 = torch.tensor(1.0)
    x2 = torch.tensor([2.0, 3.0])

    def f(x1: Tensor, x2: Tensor) -> Tensor:
        return torch.concatenate(
            [(x1 + x2.sum()).unsqueeze(0), ((x1**2) + (x2**2).sum()).unsqueeze(0)]
        )

    y, gramian = get_output_and_gramian(f, x1, x2)

    expected_jacobian = torch.tensor([[1.0, 1.0, 1.0], [2.0, 4.0, 6.0]])
    expected_gramian = expected_jacobian @ expected_jacobian.T

    assert_close(gramian, expected_gramian)


@mark.parametrize("vgp_from_module", [vgp_from_module_1, vgp_from_module_2])
def test_vgp_3(vgp_from_module):
    x = torch.randn(16, 10)

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1), nn.Flatten(start_dim=0))

    output, vgp_fn = vgp_from_module(model, x)
    gramian = get_gramian(vgp_fn, output)

    expected_gramian = get_model_gramian_via_autojac(model, x)

    assert_close(gramian, expected_gramian)


def get_model_gramian_via_autojac(model: nn.Module, input: Tensor) -> PSDMatrix:
    params = OrderedSet(model.parameters())
    outputs = OrderedSet([model(input)])
    init = Init(outputs)
    diag = Diagonalize(outputs)
    jac = Jac(outputs, params, chunk_size=None)
    mat = _Matrixify()

    jacobian_matrices = (mat << jac << diag << init)({})
    gramian = torch.sum(torch.stack([J @ J.T for J in jacobian_matrices.values()]), dim=0)

    return gramian
