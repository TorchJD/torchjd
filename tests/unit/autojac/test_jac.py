from utils.tensors import tensor_

from torchjd.autojac import jac
from torchjd.autojac._jac import _create_transform
from torchjd.autojac._transform import OrderedSet


def test_check_create_transform():
    """Tests that _create_transform creates a valid Transform."""

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    transform = _create_transform(
        outputs=OrderedSet([y1, y2]),
        inputs=OrderedSet([a1, a2]),
        retain_graph=False,
        parallel_chunk_size=None,
    )

    output_keys = transform.check_keys(set())
    assert output_keys == {a1, a2}


def test_jac():
    """Tests that jac works."""

    a1 = tensor_([1.0, 2.0], requires_grad=True)
    a2 = tensor_([3.0, 4.0], requires_grad=True)
    inputs = [a1, a2]

    y1 = tensor_([-1.0, 1.0]) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()
    outputs = [y1, y2]

    jacobians = jac(outputs, inputs)

    assert len(jacobians) == len([a1, a2])
    for jacobian, a in zip(jacobians, [a1, a2]):
        assert jacobian.shape[0] == len([y1, y2])
        assert jacobian.shape[1:] == a.shape
