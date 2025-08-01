import torch
from pytest import mark, raises
from torch.autograd import grad
from torch.testing import assert_close
from unit._utils import randn_, tensor_

from torchjd import backward
from torchjd._autojac._backward import _create_transform
from torchjd._autojac._transform import OrderedSet
from torchjd.aggregation import MGDA, Aggregator, Mean, Random, Sum, UPGrad


def test_check_create_transform():
    """Tests that _create_transform creates a valid Transform."""

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    transform = _create_transform(
        tensors=OrderedSet([y1, y2]),
        aggregator=Mean(),
        inputs=OrderedSet([a1, a2]),
        retain_graph=False,
        parallel_chunk_size=None,
    )

    output_keys = transform.check_keys(set())
    assert output_keys == set()


@mark.parametrize("aggregator", [Mean(), UPGrad(), MGDA(), Random()])
def test_various_aggregators(aggregator: Aggregator):
    """Tests that backward works for various aggregators."""

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    backward([y1, y2], aggregator)

    for a in [a1, a2]:
        assert (a.grad is not None) and (a.shape == a.grad.shape)


@mark.parametrize("aggregator", [Mean(), UPGrad()])
@mark.parametrize("shape", [(1, 3), (2, 3), (2, 6), (5, 8), (20, 55)])
@mark.parametrize("manually_specify_inputs", [True, False])
@mark.parametrize("chunk_size", [1, 2, None])
def test_value_is_correct(
    aggregator: Aggregator,
    shape: tuple[int, int],
    manually_specify_inputs: bool,
    chunk_size: int | None,
):
    """
    Tests that the .grad value filled by backward is correct in a simple example of matrix-vector
    product.
    """

    J = randn_(shape)
    input = randn_([shape[1]], requires_grad=True)
    output = J @ input  # Note that the Jacobian of output w.r.t. input is J.

    if manually_specify_inputs:
        inputs = [input]
    else:
        inputs = None

    backward(
        [output],
        aggregator,
        inputs=inputs,
        parallel_chunk_size=chunk_size,
    )

    assert_close(input.grad, aggregator(J))


def test_empty_inputs():
    """Tests that backward does not fill the .grad values if no input is specified."""

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    backward([y1, y2], Mean(), inputs=[])

    for a in [a1, a2]:
        assert a.grad is None


def test_partial_inputs():
    """
    Tests that backward fills the right .grad values when only a subset of the actual inputs are
    specified as inputs.
    """

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    backward([y1, y2], Mean(), inputs=[a1])

    assert (a1.grad is not None) and (a1.shape == a1.grad.shape)
    assert a2.grad is None


def test_empty_tensors_fails():
    """Tests that backward raises an error when called with an empty list of tensors."""

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    with raises(ValueError):
        backward([], UPGrad(), inputs=[a1, a2])


def test_multiple_tensors():
    """
    Tests that giving multiple tensors to backward is equivalent to giving a single tensor
    containing the all the values of the original tensors.
    """

    aggregator = UPGrad()

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)
    inputs = [a1, a2]

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    backward([y1, y2], aggregator, retain_graph=True)

    input_to_grad = {a: a.grad for a in inputs}
    for a in inputs:
        a.grad = None

    backward(torch.cat([y1.reshape(-1), y2.reshape(-1)]), aggregator)

    for a in inputs:
        assert (a.grad == input_to_grad[a]).all()


@mark.parametrize("chunk_size", [None, 1, 2, 4])
def test_various_valid_chunk_sizes(chunk_size):
    """Tests that backward works for various valid values of parallel_chunk_size."""

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    backward([y1, y2], UPGrad(), parallel_chunk_size=chunk_size)

    for a in [a1, a2]:
        assert (a.grad is not None) and (a.shape == a.grad.shape)


@mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_fails(chunk_size: int):
    """Tests that backward raises an error when using invalid chunk sizes."""

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    with raises(ValueError):
        backward([y1, y2], UPGrad(), parallel_chunk_size=chunk_size)


def test_input_retaining_grad_fails():
    """
    Tests that backward raises an error when some input in the computation graph of the ``tensors``
    parameter retains grad and vmap has to be used.
    """

    a = tensor_([1.0, 2.0], requires_grad=True)
    b = 2 * a
    b.retain_grad()
    y = 3 * b

    with raises(RuntimeError):
        backward(tensors=y, aggregator=UPGrad(), inputs=[b])


def test_non_input_retaining_grad_fails():
    """
    Tests that backward fails to fill a valid `.grad` when some tensor in the computation graph of
    the ``tensors`` parameter retains grad and vmap has to be used.
    """

    a = tensor_([1.0, 2.0], requires_grad=True)
    b = 2 * a
    b.retain_grad()
    y = 3 * b

    # backward itself doesn't raise the error, but it fills b.grad with a BatchedTensor
    backward(tensors=y, aggregator=UPGrad(), inputs=[a])

    with raises(RuntimeError):
        # Using such a BatchedTensor should result in an error
        _ = -b.grad


@mark.parametrize("chunk_size", [1, 3, None])
def test_tensor_used_multiple_times(chunk_size: int | None):
    """
    Tests that backward works correctly when one of the inputs is used multiple times. In this
    setup, the autograd graph is still acyclic, but the graph of tensors used becomes cyclic.
    """

    a = tensor_(3.0, requires_grad=True)
    b = 2.0 * a
    c = a * b
    d = a * c
    e = a * d
    aggregator = UPGrad()

    backward([d, e], aggregator=aggregator, parallel_chunk_size=chunk_size)

    expected_jacobian = tensor_(
        [
            [2.0 * 3.0 * a**2],
            [2.0 * 4.0 * a**3],
        ],
    )

    assert_close(a.grad, aggregator(expected_jacobian).squeeze())


def test_repeated_tensors():
    """
    Tests that backward does not allow repeating tensors.

    This behavior is different from torch.autograd.backward which would sum the gradients of the
    repeated tensors, but it simplifies a lot the implementation of autojac and there are
    alternative ways of producing Jacobians with repeated rows anyway.
    """

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + (a2**2).sum()

    with raises(ValueError):
        backward([y1, y1, y2], Sum())


def test_repeated_inputs():
    """
    Tests that backward correctly works when some inputs are repeated. In this case, since
    torch.autograd.backward ignores the repetition of the inputs, it is natural for autojac to
    ignore that as well.
    """

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + (a2**2).sum()

    expected_grad_wrt_a1 = grad([y1, y2], a1, retain_graph=True)[0]
    expected_grad_wrt_a2 = grad([y1, y2], a2, retain_graph=True)[0]

    backward([y1, y2], Sum(), inputs=[a1, a1, a2])

    assert_close(a1.grad, expected_grad_wrt_a1)
    assert_close(a2.grad, expected_grad_wrt_a2)
