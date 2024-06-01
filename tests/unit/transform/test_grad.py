import torch
from unit.transform.utils import assert_tensor_dicts_are_close

from torchjd.transform import Grad, Gradients


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


def test_single_input_two_levels():
    """
    Tests that the Grad transform works correctly for a very simple example of differentiation.
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

    outer_grad = Grad(outputs=[y1, y2], inputs=[a, b])
    inner_grad = Grad(outputs=[z1, z2], inputs=[y1, y2])
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
    y1 = a1 * x1
    y2 = a2 * x2
    y = torch.stack([y1, y2])
    input = Gradients({y: torch.ones_like(y)})

    grad1 = Grad(outputs=[y], inputs=[a1])
    grad2 = Grad(outputs=[y], inputs=[a2])
    conjunction = grad1 | grad2
    grad = Grad(outputs=[y], inputs=[a1, a2])

    gradients = conjunction(input)
    expected_gradients = grad(input)

    assert_tensor_dicts_are_close(gradients, expected_gradients)
