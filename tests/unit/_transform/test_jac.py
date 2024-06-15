import torch
from unit._transform.utils import assert_tensor_dicts_are_close

from torchjd._transform import Jac, Jacobians


def test_single_input():
    """
    Tests that the Jac transform works correctly for an example of multiple differentiation. Here,
    the functions considered are: `y1 = a1 * x` and `y2 = a2 * x`. We want to compute the jacobians
    of `[y1, y2]` with respect to the parameters `a1` and `a2`. These jacobians should be equal to
    [x, 0] and [0, x], respectively.
    """

    x = torch.tensor(5.0)
    a1 = torch.tensor(2.0, requires_grad=True)
    a2 = torch.tensor(3.0, requires_grad=True)
    y1 = a1 * x
    y2 = a2 * x
    y = torch.stack([y1, y2])
    input = Jacobians({y: torch.eye(2)})

    jac = Jac(outputs=[y], inputs=[a1, a2], chunk_size=None)

    jacobians = jac(input)
    expected_jacobians = {
        a1: torch.stack([x, torch.zeros([])]),
        a2: torch.stack([torch.zeros([]), x]),
    }

    assert_tensor_dicts_are_close(jacobians, expected_jacobians)


def test_empty_inputs_1():
    """
    Tests that the Jac transform works correctly when the `inputs` parameter is an empty `Iterable`.
    """

    y1 = torch.tensor(1.0, requires_grad=True)
    y2 = torch.tensor(1.0, requires_grad=True)
    y = torch.stack([y1, y2])
    input = Jacobians({y: torch.eye(2)})

    jac = Jac(outputs=[y], inputs=[], chunk_size=None)

    jacobians = jac(input)
    expected_jacobians = {}

    assert_tensor_dicts_are_close(jacobians, expected_jacobians)


def test_empty_inputs_2():
    """
    Tests that the Jac transform works correctly when the `inputs` parameter is an empty `Iterable`.
    """

    x = torch.tensor(5.0)
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    y1 = a * x
    y2 = b * x
    y = torch.stack([y1, y2])
    input = Jacobians({y: torch.eye(2)})

    jac = Jac(outputs=[y], inputs=[], chunk_size=None)

    jacobians = jac(input)
    expected_jacobians = {}

    assert_tensor_dicts_are_close(jacobians, expected_jacobians)


def test_two_levels():
    """
    Tests that the Jac transform works correctly for an example of chained differentiation. Here,
    the function considered is: `z = a * x1 * x2`, which is computed in 2 parts: `y = a * x1` and
    `z = y * x2`. We want to compute the derivative of `z` with respect to the parameter `a`, by
    using chain rule. This derivative should be equal to `x1 * x2`.
    """

    x1 = torch.tensor(5.0)
    x2 = torch.tensor(6.0)
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    y1 = a * x1
    y2 = b * x1
    y = torch.stack([y1, y2])
    z = y * x2
    input = Jacobians({z: torch.eye(2)})

    outer_jac = Jac(outputs=[y], inputs=[a, b], chunk_size=None, retain_graph=True)
    inner_jac = Jac(outputs=[z], inputs=[y], chunk_size=None, retain_graph=True)
    composed_jac = outer_jac << inner_jac
    jac = Jac(outputs=[z], inputs=[a, b], chunk_size=None)

    jacobians = composed_jac(input)
    expected_jacobians = jac(input)

    assert_tensor_dicts_are_close(jacobians, expected_jacobians)


def test_composition_of_jacs_is_jac():
    """
    Tests that the composition of 2 Jac transforms is equivalent to computing the Jac directly in
    a single transform.
    """

    x1 = torch.tensor(5.0)
    x2 = torch.tensor(6.0)
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    y1 = a * x1
    y2 = a * x2
    z1 = y1 + x2
    z2 = y2 + x1
    input = Jacobians({z1: torch.tensor([1.0, 0.0]), z2: torch.tensor([0.0, 1.0])})

    outer_jac = Jac(outputs=[y1, y2], inputs=[a, b], chunk_size=None, retain_graph=True)
    inner_jac = Jac(outputs=[z1, z2], inputs=[y1, y2], chunk_size=None, retain_graph=True)
    composed_jac = outer_jac << inner_jac
    jac = Jac(outputs=[z1, z2], inputs=[a, b], chunk_size=None)

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
