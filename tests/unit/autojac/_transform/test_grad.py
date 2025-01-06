import torch
from pytest import raises

from torchjd.autojac._transform import Grad, Gradients

from ._dict_assertions import assert_tensor_dicts_are_close


def test_single_input():
    """
    Tests that the Grad transform works correctly for a very simple example of differentiation.
    Here, the function considered is: `y = a * x`. We want to compute the derivative of `y` with
    respect to the parameter `a`. This derivative should be equal to `x`.
    """

    x = torch.tensor(5.0)
    a = torch.tensor(2.0, requires_grad=True)
    y = a * x
    input = Gradients({y: torch.ones_like(y)})

    grad = Grad(outputs=[y], inputs=[a])

    gradients = grad(input)
    expected_gradients = {a: x}

    assert_tensor_dicts_are_close(gradients, expected_gradients)


def test_empty_inputs_1():
    """
    Tests that the Grad transform works correctly when the `inputs` parameter is an empty
    `Iterable`.
    """

    y = torch.tensor(1.0, requires_grad=True)
    input = Gradients({y: torch.ones_like(y)})

    grad = Grad(outputs=[y], inputs=[])

    gradients = grad(input)
    expected_gradients = {}

    assert_tensor_dicts_are_close(gradients, expected_gradients)


def test_empty_inputs_2():
    """
    Tests that the Grad transform works correctly when the `inputs` parameter is an empty
    `Iterable`.
    """

    x = torch.tensor(5.0)
    a = torch.tensor(1.0, requires_grad=True)
    y = a * x
    input = Gradients({y: torch.ones_like(y)})

    grad = Grad(outputs=[y], inputs=[])

    gradients = grad(input)
    expected_gradients = {}

    assert_tensor_dicts_are_close(gradients, expected_gradients)


def test_retain_graph():
    """Tests that the `Grad` transform behaves as expected with the `retain_graph` flag."""

    x = torch.tensor(5.0)
    a = torch.tensor(2.0, requires_grad=True)
    y = a * x
    input = Gradients({y: torch.ones_like(y)})

    grad_retain_graph = Grad(outputs=[y], inputs=[a], retain_graph=True)
    grad_discard_graph = Grad(outputs=[y], inputs=[a], retain_graph=False)

    grad_retain_graph(input)
    grad_retain_graph(input)
    grad_discard_graph(input)
    with raises(RuntimeError):
        grad_retain_graph(input)
    with raises(RuntimeError):
        grad_discard_graph(input)


def test_single_input_two_levels():
    """
    Tests that the Grad transform works correctly when composed with another Grad transform.
    Here, the function considered is: `z = a * x1 * x2`, which is computed in 2 parts: `y = a * x1`
    and `z = y * x2`. We want to compute the derivative of `z` with respect to the parameter `a`, by
    using chain rule. This derivative should be equal to `x1 * x2`.
    """

    x1 = torch.tensor(5.0)
    x2 = torch.tensor(6.0)
    a = torch.tensor(2.0, requires_grad=True)
    y = a * x1
    z = y * x2
    input = Gradients({z: torch.ones_like(z)})

    outer_grad = Grad(outputs=[y], inputs=[a])
    inner_grad = Grad(outputs=[z], inputs=[y])
    grad = outer_grad << inner_grad

    gradients = grad(input)
    expected_gradients = {a: x1 * x2}

    assert_tensor_dicts_are_close(gradients, expected_gradients)


def test_empty_inputs_two_levels():
    """
    Tests that the Grad transform works correctly when the `inputs` parameter is an empty
    `Iterable`, with 2 composed Grad transforms.
    """

    x1 = torch.tensor(5.0)
    x2 = torch.tensor(6.0)
    a = torch.tensor(2.0, requires_grad=True)
    y = a * x1
    z = y * x2
    input = Gradients({z: torch.ones_like(z)})

    outer_grad = Grad(outputs=[y], inputs=[])
    inner_grad = Grad(outputs=[z], inputs=[y])
    composed_grad = outer_grad << inner_grad

    gradients = composed_grad(input)
    expected_gradients = {}

    assert_tensor_dicts_are_close(gradients, expected_gradients)


def test_vector_output():
    """
    Tests that the Grad transform works correctly when the `outputs` contains a single vector.
    The input (grad_outputs) is not the same for both values of the output, so that this test also
    checks that the scaling is performed correctly.
    """

    x = torch.tensor(5.0)
    a = torch.tensor(2.0, requires_grad=True)
    y = torch.stack([a * x, a**2])
    input = Gradients({y: torch.tensor([3.0, 1.0])})

    grad = Grad(outputs=[y], inputs=[a])

    gradients = grad(input)
    expected_gradients = {a: x * 3 + 2 * a}

    assert_tensor_dicts_are_close(gradients, expected_gradients)


def test_multiple_outputs():
    """
    Tests that the Grad transform works correctly when the `outputs` contains 2 scalars.
    The input (grad_outputs) is not the same for both outputs, so that this test also checks that
    the scaling is performed correctly.
    """

    x = torch.tensor(5.0)
    a = torch.tensor(2.0, requires_grad=True)
    y1 = a * x
    y2 = a**2
    input = Gradients({y1: torch.ones_like(y1) * 3, y2: torch.ones_like(y2)})

    grad = Grad(outputs=[y1, y2], inputs=[a])

    gradients = grad(input)
    expected_gradients = {a: x * 3 + 2 * a}

    assert_tensor_dicts_are_close(gradients, expected_gradients)


def test_multiple_tensor_outputs():
    """
    Tests that the Grad transform works correctly when the `outputs` contains several tensors of
    different shapes. The input (grad_outputs) is not the same for all values of the outputs, so
    that this test also checks that the scaling is performed correctly.
    """

    x = torch.tensor(5.0)
    a = torch.tensor(2.0, requires_grad=True)
    y1 = a * x
    y2 = torch.stack([a**2, 2 * a**2])
    y3 = torch.stack([a**3, 2 * a**3]).unsqueeze(0)
    input = Gradients(
        {
            y1: torch.tensor(3.0),
            y2: torch.tensor([6.0, 7.0]),
            y3: torch.tensor([[9.0, 10.0]]),
        }
    )

    grad = Grad(outputs=[y1, y2, y3], inputs=[a])

    gradients = grad(input)
    g = x * 3 + 2 * a * 6 + 2 * a * 2 * 7 + 3 * a**2 * 9 + 3 * a**2 * 2 * 10.0
    expected_gradients = {a: g}

    assert_tensor_dicts_are_close(gradients, expected_gradients)


def test_composition_of_grads_is_grad():
    """
    Tests that the composition of 2 Grad transforms is equivalent to computing the Grad directly in
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
    input = Gradients({z1: torch.ones_like(z1), z2: torch.ones_like(z2)})

    outer_grad = Grad(outputs=[y1, y2], inputs=[a, b], retain_graph=True)
    inner_grad = Grad(outputs=[z1, z2], inputs=[y1, y2], retain_graph=True)
    composed_grad = outer_grad << inner_grad
    grad = Grad(outputs=[z1, z2], inputs=[a, b])

    gradients = composed_grad(input)
    expected_gradients = grad(input)

    assert_tensor_dicts_are_close(gradients, expected_gradients)


def test_conjunction_of_grads_is_grad():
    """
    Tests that the conjunction of 2 Grad transforms is equivalent to computing the Grad directly in
    a single transform.
    """

    x1 = torch.tensor(5.0)
    x2 = torch.tensor(6.0)
    a1 = torch.tensor(2.0, requires_grad=True)
    a2 = torch.tensor(3.0, requires_grad=True)
    y = torch.stack([a1 * x1, a2 * x2])
    input = Gradients({y: torch.ones_like(y)})

    grad1 = Grad(outputs=[y], inputs=[a1], retain_graph=True)
    grad2 = Grad(outputs=[y], inputs=[a2], retain_graph=True)
    conjunction = grad1 | grad2
    grad = Grad(outputs=[y], inputs=[a1, a2])

    gradients = conjunction(input)
    expected_gradients = grad(input)

    assert_tensor_dicts_are_close(gradients, expected_gradients)


def test_create_graph():
    """Tests that the Grad transform behaves correctly when `create_graph` is set to `True`."""

    a = torch.tensor(2.0, requires_grad=True)
    y = a * a
    input = Gradients({y: torch.ones_like(y)})

    grad = Grad(outputs=[y], inputs=[a], create_graph=True)

    gradients = grad(input)

    assert gradients[a].requires_grad
