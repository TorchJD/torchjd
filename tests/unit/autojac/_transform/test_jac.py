import torch
from pytest import mark, raises

from torchjd.autojac._transform import Jac, Jacobians

from ._dict_assertions import assert_tensor_dicts_are_close


@mark.parametrize("chunk_size", [1, 3, None])
def test_single_input(chunk_size: int | None):
    """
    Tests that the Jac transform works correctly for an example of multiple differentiation. Here,
    the function considered is: `y = [a1 * x, a2 * x]`. We want to compute the jacobians of `y` with
    respect to `a1` and `a2`.
    """

    x = torch.tensor(5.0)
    a1 = torch.tensor(2.0, requires_grad=True)
    a2 = torch.tensor(3.0, requires_grad=True)
    y = torch.stack([a1 * x, a2 * x])
    input = Jacobians({y: torch.eye(2)})

    jac = Jac(outputs=[y], inputs=[a1, a2], chunk_size=chunk_size)

    jacobians = jac(input)
    expected_jacobians = {
        a1: torch.stack([x, torch.zeros([])]),
        a2: torch.stack([torch.zeros([]), x]),
    }

    assert_tensor_dicts_are_close(jacobians, expected_jacobians)


@mark.parametrize("chunk_size", [1, 3, None])
def test_empty_inputs_1(chunk_size: int | None):
    """
    Tests that the Jac transform works correctly when the `inputs` parameter is an empty `Iterable`.
    """

    y1 = torch.tensor(1.0, requires_grad=True)
    y2 = torch.tensor(1.0, requires_grad=True)
    y = torch.stack([y1, y2])
    input = Jacobians({y: torch.eye(2)})

    jac = Jac(outputs=[y], inputs=[], chunk_size=chunk_size)

    jacobians = jac(input)
    expected_jacobians = {}

    assert_tensor_dicts_are_close(jacobians, expected_jacobians)


@mark.parametrize("chunk_size", [1, 3, None])
def test_empty_inputs_2(chunk_size: int | None):
    """
    Tests that the Jac transform works correctly when the `inputs` parameter is an empty `Iterable`.
    """

    x = torch.tensor(5.0)
    a1 = torch.tensor(1.0, requires_grad=True)
    a2 = torch.tensor(1.0, requires_grad=True)
    y1 = a1 * x
    y2 = a2 * x
    y = torch.stack([y1, y2])
    input = Jacobians({y: torch.eye(2)})

    jac = Jac(outputs=[y], inputs=[], chunk_size=chunk_size)

    jacobians = jac(input)
    expected_jacobians = {}

    assert_tensor_dicts_are_close(jacobians, expected_jacobians)


def test_retain_graph():
    """Tests that the `Jac` transform behaves as expected with the `retain_graph` flag."""

    x = torch.tensor(5.0)
    a1 = torch.tensor(2.0, requires_grad=True)
    a2 = torch.tensor(3.0, requires_grad=True)
    y1 = a1 * x
    y2 = a2 * x
    y = torch.stack([y1, y2])
    input = Jacobians({y: torch.eye(2)})

    jac_retain_graph = Jac(outputs=[y], inputs=[a1, a2], chunk_size=None, retain_graph=True)
    jac_discard_graph = Jac(outputs=[y], inputs=[a1, a2], chunk_size=None, retain_graph=False)

    jac_retain_graph(input)
    jac_retain_graph(input)
    jac_discard_graph(input)
    with raises(RuntimeError):
        jac_retain_graph(input)
    with raises(RuntimeError):
        jac_discard_graph(input)


def test_two_levels():
    """
    Tests that the Jac transform works correctly for an example of chained differentiation. Here,
    the function considered is: `z = a * x1 * x2`, which is computed in 2 parts: `y = a * x1` and
    `z = y * x2`. We want to compute the derivative of `z` with respect to the parameter `a`, by
    using chain rule. This derivative should be equal to `x1 * x2`.
    """

    x1 = torch.tensor(5.0)
    x2 = torch.tensor(6.0)
    a1 = torch.tensor(2.0, requires_grad=True)
    a2 = torch.tensor(3.0, requires_grad=True)
    y1 = a1 * x1
    y2 = a2 * x1
    y = torch.stack([y1, y2])
    z = y * x2
    input = Jacobians({z: torch.eye(2)})

    outer_jac = Jac(outputs=[y], inputs=[a1, a2], chunk_size=None, retain_graph=True)
    inner_jac = Jac(outputs=[z], inputs=[y], chunk_size=None, retain_graph=True)
    composed_jac = outer_jac << inner_jac
    jac = Jac(outputs=[z], inputs=[a1, a2], chunk_size=None)

    jacobians = composed_jac(input)
    expected_jacobians = jac(input)

    assert_tensor_dicts_are_close(jacobians, expected_jacobians)


@mark.parametrize("chunk_size", [1, 3, None])
def test_multiple_outputs_1(chunk_size: int | None):
    """
    Tests that the Jac transform works correctly when the `outputs` contains 3 vectors.
    The input (jac_outputs) is not the same for all outputs, so that this test also checks that the
    scaling is performed correctly.
    """

    x = torch.tensor(5.0)
    a1 = torch.tensor(2.0, requires_grad=True)
    a2 = torch.tensor(3.0, requires_grad=True)
    y1 = torch.stack([a1 * x, a2 * x])
    y2 = torch.stack([a2**2, a1**2])
    y3 = torch.stack([a2**3, a1**3])

    identity_2x2 = torch.eye(2)
    zeros_2x2 = torch.zeros(2, 2)
    jac_output1 = torch.cat([identity_2x2 * 7, zeros_2x2, zeros_2x2])
    jac_output2 = torch.cat([zeros_2x2, identity_2x2, zeros_2x2])
    jac_output3 = torch.cat([zeros_2x2, zeros_2x2, identity_2x2])
    input = Jacobians({y1: jac_output1, y2: jac_output2, y3: jac_output3})

    jac = Jac(outputs=[y1, y2, y3], inputs=[a1, a2], chunk_size=chunk_size)

    jacobians = jac(input)
    zero_scalar = torch.tensor(0.0)
    expected_jacobians = {
        a1: torch.stack([x * 7, zero_scalar, zero_scalar, 2 * a1, zero_scalar, 3 * a1**2]),
        a2: torch.stack([zero_scalar, x * 7, 2 * a2, zero_scalar, 3 * a2**2, zero_scalar]),
    }

    assert_tensor_dicts_are_close(jacobians, expected_jacobians)


@mark.parametrize("chunk_size", [1, 3, None])
def test_multiple_outputs_2(chunk_size: int | None):
    """
    Same as test_multiple_outputs_1 but with different jac_outputs, so the returned jacobians are of
    different shapes.
    """

    x = torch.tensor(5.0)
    a1 = torch.tensor(2.0, requires_grad=True)
    a2 = torch.tensor(3.0, requires_grad=True)
    y1 = torch.stack([a1 * x, a2 * x])
    y2 = torch.stack([a2**2, a1**2])
    y3 = torch.stack([a2**3, a1**3])

    ones_2 = torch.ones(2)
    zeros_2 = torch.zeros(2)
    jac_output1 = torch.stack([ones_2 * 7, zeros_2, zeros_2])
    jac_output2 = torch.stack([zeros_2, ones_2, zeros_2])
    jac_output3 = torch.stack([zeros_2, zeros_2, ones_2])
    input = Jacobians({y1: jac_output1, y2: jac_output2, y3: jac_output3})

    jac = Jac(outputs=[y1, y2, y3], inputs=[a1, a2], chunk_size=chunk_size)

    jacobians = jac(input)
    expected_jacobians = {
        a1: torch.stack([x * 7, 2 * a1, 3 * a1**2]),
        a2: torch.stack([x * 7, 2 * a2, 3 * a2**2]),
    }

    assert_tensor_dicts_are_close(jacobians, expected_jacobians)


def test_composition_of_jacs_is_jac():
    """
    Tests that the composition of 2 Jac transforms is equivalent to computing the Jac directly in
    a single transform.
    """

    x1 = torch.tensor(5.0)
    x2 = torch.tensor(6.0)
    a = torch.tensor(2.0, requires_grad=True)
    y1 = a * x1
    y2 = a * x2
    z1 = y1 + x2
    z2 = y2 + x1
    input = Jacobians({z1: torch.tensor([1.0, 0.0]), z2: torch.tensor([0.0, 1.0])})

    outer_jac = Jac(outputs=[y1, y2], inputs=[a], chunk_size=None, retain_graph=True)
    inner_jac = Jac(outputs=[z1, z2], inputs=[y1, y2], chunk_size=None, retain_graph=True)
    composed_jac = outer_jac << inner_jac
    jac = Jac(outputs=[z1, z2], inputs=[a], chunk_size=None)

    jacobians = composed_jac(input)
    expected_jacobians = jac(input)

    assert_tensor_dicts_are_close(jacobians, expected_jacobians)


def test_conjunction_of_jacs_is_jac():
    """
    Tests that the conjunction of 2 Jac transforms is equivalent to computing the Jac directly in
    a single transform.
    """

    x1 = torch.tensor(5.0)
    x2 = torch.tensor(6.0)
    a1 = torch.tensor(2.0, requires_grad=True)
    a2 = torch.tensor(3.0, requires_grad=True)
    y1 = a1 * x1
    y2 = a2 * x2
    y = torch.stack([y1, y2])
    input = Jacobians({y: torch.eye(len(y))})

    jac1 = Jac(outputs=[y], inputs=[a1], chunk_size=None, retain_graph=True)
    jac2 = Jac(outputs=[y], inputs=[a2], chunk_size=None, retain_graph=True)
    conjunction_of_jacs = jac1 | jac2
    jac = Jac(outputs=[y], inputs=[a1, a2], chunk_size=None)

    jacobians = conjunction_of_jacs(input)
    expected_jacobians = jac(input)

    assert_tensor_dicts_are_close(jacobians, expected_jacobians)


def test_create_graph():
    """Tests that the Jac transform behaves correctly when `create_graph` is set to `True`."""

    x = torch.tensor(5.0)
    a1 = torch.tensor(2.0, requires_grad=True)
    a2 = torch.tensor(3.0, requires_grad=True)
    y1 = a1 * a2
    y2 = a2 * x
    y = torch.stack([y1, y2])
    input = Jacobians({y: torch.eye(2)})

    jac = Jac(outputs=[y], inputs=[a1, a2], chunk_size=None, create_graph=True)

    jacobians = jac(input)

    assert jacobians[a1].requires_grad
    assert jacobians[a2].requires_grad
