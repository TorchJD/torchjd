import torch
from torch.testing import assert_close

from torchjd.autojac._transform import (
    Accumulate,
    Conjunction,
    Diagonalize,
    EmptyTensorDict,
    Grad,
    Gradients,
    Init,
    Jac,
    Jacobians,
    Select,
    Stack,
    TensorDict,
)

from ._dict_assertions import assert_tensor_dicts_are_close


def test_jac_is_stack_of_grads():
    """
    Tests that the Jac transform (composed with a Diagonalize) is equivalent to a Stack of Grad and
    Select transforms.
    """

    x = torch.tensor(5.0)
    a1 = torch.tensor(2.0, requires_grad=True)
    a2 = torch.tensor(3.0, requires_grad=True)
    y1 = a1 * x
    y2 = a2 * x
    input = Gradients({y1: torch.ones_like(y1), y2: torch.ones_like(y2)})

    jac = Jac(outputs=[y1, y2], inputs=[a1, a2], chunk_size=None, retain_graph=True)
    diag = Diagonalize([y1, y2])
    jac_diag = jac << diag

    grad1 = Grad(outputs=[y1], inputs=[a1, a2])
    grad2 = Grad(outputs=[y2], inputs=[a1, a2])
    select1 = Select([y1], [y1, y2])
    select2 = Select([y2], [y1, y2])
    stack_of_grads = Stack([grad1 << select1, grad2 << select2])

    jacobians = jac_diag(input)
    expected_jacobians = stack_of_grads(input)

    assert_tensor_dicts_are_close(jacobians, expected_jacobians)


def test_single_differentiation():
    """
    Tests that we can perform a single scalar differentiation with the composition of a Grad and an
    Init transform.
    """

    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = a * 2.0
    input = EmptyTensorDict()

    init = Init([y])
    grad = Grad([y], [a])
    transform = grad << init

    output = transform(input)
    expected_output = {a: torch.tensor([2.0, 2.0, 2.0])}

    assert_tensor_dicts_are_close(output, expected_output)


def test_multiple_differentiations():
    """
    Tests that we can perform multiple scalar differentiations with the conjunction of multiple Grad
    transforms, composed with an Init transform.
    """

    a1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    a2 = torch.tensor([1.0, 3.0, 5.0], requires_grad=True)
    y1 = a1 * 2.0
    y2 = a2 * 3.0
    input = EmptyTensorDict()

    grad1 = Grad([y1], [a1])
    grad2 = Grad([y2], [a2])
    select1 = Select([y1], [y1, y2])
    select2 = Select([y2], [y1, y2])
    init = Init([y1, y2])
    transform = ((grad1 << select1) | (grad2 << select2)) << init

    output = transform(input)
    expected_output = {
        a1: torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
        a2: torch.tensor([3.0, 3.0, 3.0]),
    }

    assert_tensor_dicts_are_close(output, expected_output)


def test_str():
    """Tests that the __str__ method works correctly even for a complex transform."""
    init = Init([])
    diag = Diagonalize([])
    jac = Jac([], [], chunk_size=None)
    transform = jac << diag << init

    assert str(transform) == "Jac ∘ Diagonalize ∘ Init"


def test_simple_conjunction():
    """
    Tests that the Conjunction transform works correctly with a simple example involving several
    Select transforms, whose keys form a partition of the keys of the input tensor dict.
    Because of this, the output is expected to be the same as the input.
    """

    x1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x2 = torch.tensor([1.0, 3.0, 5.0])
    x3 = torch.tensor(4.0)
    input = TensorDict({x1: torch.ones_like(x1), x2: torch.ones_like(x2), x3: torch.ones_like(x3)})

    select1 = Select([x1], [x1, x2, x3])
    select2 = Select([x2], [x1, x2, x3])
    select3 = Select([x3], [x1, x2, x3])
    conjunction = Conjunction([select1, select2, select3])

    output = conjunction(input)
    expected_output = input

    assert_tensor_dicts_are_close(output, expected_output)


def test_conjunction_is_commutative():
    """
    Tests that the Conjunction transform gives the same result no matter the order in which its
    transforms are given.
    """

    x1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x2 = torch.tensor([1.0, 3.0, 5.0])
    input = TensorDict({x1: torch.ones_like(x1), x2: torch.ones_like(x2)})

    a = Select([x1], [x1, x2])
    b = Select([x2], [x1, x2])
    flipped_conjunction = Conjunction([b, a])
    conjunction = Conjunction([a, b])

    output = flipped_conjunction(input)
    expected_output = conjunction(input)

    assert_tensor_dicts_are_close(output, expected_output)


def test_conjunction_is_associative():
    """
    Tests that the Conjunction transform gives the same result no matter how it is parenthesized.
    """

    x1 = torch.tensor([[3.0, 11.0], [2.0, 7.0]])
    x2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x3 = torch.tensor([1.0, 3.0, 5.0])
    x4 = torch.tensor(4.0)
    input = TensorDict(
        {
            x1: torch.ones_like(x1),
            x2: torch.ones_like(x2),
            x3: torch.ones_like(x3),
            x4: torch.ones_like(x4),
        }
    )

    a = Select([x1], [x1, x2, x3, x4])
    b = Select([x2], [x1, x2, x3, x4])
    c = Select([x3], [x1, x2, x3, x4])
    d = Select([x4], [x1, x2, x3, x4])

    parenthesized_conjunction = Conjunction([a, Conjunction([Conjunction([b, c]), d])])
    conjunction = Conjunction([a, b, c, d])

    output = parenthesized_conjunction(input)
    expected_output = conjunction(input)

    assert_tensor_dicts_are_close(output, expected_output)


def test_conjunction_accumulate_select():
    """
    Tests that it is possible to conjunct an Accumulate and a Select in this order.
    It is not trivial since the type of the TensorDict returned by the first transform (Accumulate)
    is EmptyDict, which is not the type that the conjunction should return (Gradients), but a
    subclass of it.
    """

    key = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    value = torch.ones_like(key)
    input = Gradients({key: value})

    select = Select([], [key])
    accumulate = Accumulate([key])
    conjunction = accumulate | select

    output = conjunction(input)
    expected_output = {}

    assert_tensor_dicts_are_close(output, expected_output)


def test_equivalence_jac_grads():
    """
    Tests that differentiation in parallel using `_jac` is equivalent to sequential differentiation
    using several calls to `_grad` and stacking the resulting gradients.
    """

    A = torch.tensor([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]], requires_grad=True)
    b = torch.tensor([0.0, 2.0], requires_grad=True)
    c = torch.tensor(1.0, requires_grad=True)

    X1 = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    x2 = torch.tensor([5.0, 4.0, 3.0])

    y1 = X1 @ A @ b
    y2 = x2 @ A @ b + c

    inputs = [A, b, c]
    outputs = [y1, y2]
    grad_outputs = [torch.ones_like(output) for output in outputs]

    grad1 = Grad(outputs=[outputs[0]], inputs=inputs, retain_graph=True)
    grad_dict_1 = grad1(Gradients({outputs[0]: grad_outputs[0]}))
    grad_1_A, grad_1_b, grad_1_c = grad_dict_1[A], grad_dict_1[b], grad_dict_1[c]

    grad2 = Grad(outputs=[outputs[1]], inputs=inputs, retain_graph=True)
    grad_dict_2 = grad2(Gradients({outputs[1]: grad_outputs[1]}))
    grad_2_A, grad_2_b, grad_2_c = grad_dict_2[A], grad_dict_2[b], grad_dict_2[c]

    n_outputs = len(outputs)
    batched_grad_outputs = [
        torch.zeros((n_outputs,) + grad_output.shape) for grad_output in grad_outputs
    ]
    for i, grad_output in enumerate(grad_outputs):
        batched_grad_outputs[i][i] = grad_output

    jac = Jac(outputs=outputs, inputs=inputs, chunk_size=None)
    jac_dict = jac(
        Jacobians({outputs[0]: batched_grad_outputs[0], outputs[1]: batched_grad_outputs[1]})
    )
    jac_A, jac_b, jac_c = jac_dict[A], jac_dict[b], jac_dict[c]

    assert_close(jac_A, torch.stack([grad_1_A, grad_2_A]))
    assert_close(jac_b, torch.stack([grad_1_b, grad_2_b]))
    assert_close(jac_c, torch.stack([grad_1_c, grad_2_c]))
