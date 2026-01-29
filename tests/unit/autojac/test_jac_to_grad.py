from pytest import mark, raises
from utils.asserts import assert_grad_close, assert_has_jac, assert_has_no_jac
from utils.tensors import tensor_

from torchjd.aggregation import (
    IMTLG,
    MGDA,
    Aggregator,
    AlignedMTL,
    ConFIG,
    Constant,
    DualProj,
    GradDrop,
    Krum,
    Mean,
    PCGrad,
    Random,
    Sum,
    TrimmedMean,
    UPGrad,
)
from torchjd.autojac._jac_to_grad import (
    _can_skip_jacobian_combination,
    _has_forward_hook,
    jac_to_grad,
)


@mark.parametrize("aggregator", [Mean(), UPGrad(), PCGrad()])
def test_various_aggregators(aggregator: Aggregator):
    """Tests that jac_to_grad works for various aggregators."""

    t1 = tensor_(1.0, requires_grad=True)
    t2 = tensor_([2.0, 3.0], requires_grad=True)
    jac = tensor_([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    t1.__setattr__("jac", jac[:, 0])
    t2.__setattr__("jac", jac[:, 1:])
    expected_grad = aggregator(jac)
    g1 = expected_grad[0]
    g2 = expected_grad[1:]

    jac_to_grad([t1, t2], aggregator)

    assert_grad_close(t1, g1)
    assert_grad_close(t2, g2)


def test_single_tensor():
    """Tests that jac_to_grad works when a single tensor is provided."""

    aggregator = UPGrad()
    t = tensor_([2.0, 3.0, 4.0], requires_grad=True)
    jac = tensor_([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    t.__setattr__("jac", jac)
    g = aggregator(jac)

    jac_to_grad([t], aggregator)

    assert_grad_close(t, g)


def test_no_jac_field():
    """Tests that jac_to_grad fails when a tensor does not have a jac field."""

    aggregator = UPGrad()
    t1 = tensor_(1.0, requires_grad=True)
    t2 = tensor_([2.0, 3.0], requires_grad=True)
    jac = tensor_([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    t2.__setattr__("jac", jac[:, 1:])

    with raises(ValueError):
        jac_to_grad([t1, t2], aggregator)


def test_no_requires_grad():
    """Tests that jac_to_grad fails when a tensor does not require grad."""

    aggregator = UPGrad()
    t1 = tensor_(1.0, requires_grad=True)
    t2 = tensor_([2.0, 3.0], requires_grad=False)
    jac = tensor_([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    t1.__setattr__("jac", jac[:, 0])
    t2.__setattr__("jac", jac[:, 1:])

    with raises(ValueError):
        jac_to_grad([t1, t2], aggregator)


def test_row_mismatch():
    """Tests that jac_to_grad fails when the number of rows of the .jac is not constant."""

    aggregator = UPGrad()
    t1 = tensor_(1.0, requires_grad=True)
    t2 = tensor_([2.0, 3.0], requires_grad=True)
    t1.__setattr__("jac", tensor_([5.0, 6.0, 7.0]))  # 3 rows
    t2.__setattr__("jac", tensor_([[1.0, 2.0], [3.0, 4.0]]))  # 2 rows

    with raises(ValueError):
        jac_to_grad([t1, t2], aggregator)


def test_no_tensors():
    """Tests that jac_to_grad correctly does nothing when an empty list of tensors is provided."""

    jac_to_grad([], aggregator=UPGrad())


@mark.parametrize("retain_jac", [True, False])
def test_jacs_are_freed(retain_jac: bool):
    """Tests that jac_to_grad frees the jac fields if an only if retain_jac is False."""

    aggregator = UPGrad()
    t1 = tensor_(1.0, requires_grad=True)
    t2 = tensor_([2.0, 3.0], requires_grad=True)
    jac = tensor_([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    t1.__setattr__("jac", jac[:, 0])
    t2.__setattr__("jac", jac[:, 1:])

    jac_to_grad([t1, t2], aggregator, retain_jac=retain_jac)

    check = assert_has_jac if retain_jac else assert_has_no_jac
    check(t1)
    check(t2)


def test_has_forward_hook():
    """Tests that _has_forward_hook correctly detects the presence of forward hooks."""

    module = UPGrad()

    def dummy_forward_hook(_module, _input, _output):
        return _output

    def dummy_forward_pre_hook(_module, _input):
        return _input

    def dummy_backward_hook(_module, _grad_input, _grad_output):
        return _grad_input

    def dummy_backward_pre_hook(_module, _grad_output):
        return _grad_output

    # Module with no hooks or backward hooks only should return False
    assert not _has_forward_hook(module)
    module.register_full_backward_hook(dummy_backward_hook)
    assert not _has_forward_hook(module)
    module.register_full_backward_pre_hook(dummy_backward_pre_hook)
    assert not _has_forward_hook(module)

    # Module with forward hook should return True
    handle1 = module.register_forward_hook(dummy_forward_hook)
    assert _has_forward_hook(module)
    handle2 = module.register_forward_hook(dummy_forward_hook)
    assert _has_forward_hook(module)
    handle1.remove()
    assert _has_forward_hook(module)
    handle2.remove()
    assert not _has_forward_hook(module)

    # Module with forward pre-hook should return True
    handle3 = module.register_forward_pre_hook(dummy_forward_pre_hook)
    assert _has_forward_hook(module)
    handle4 = module.register_forward_pre_hook(dummy_forward_pre_hook)
    assert _has_forward_hook(module)
    handle3.remove()
    assert _has_forward_hook(module)
    handle4.remove()
    assert not _has_forward_hook(module)


_PARAMETRIZATIONS = [
    (AlignedMTL(), True),
    (DualProj(), True),
    (IMTLG(), True),
    (Krum(n_byzantine=1), True),
    (MGDA(), True),
    (PCGrad(), True),
    (UPGrad(), True),
    (ConFIG(), False),
    (Constant(tensor_([0.5, 0.5])), False),
    (GradDrop(), False),
    (Mean(), False),
    (Random(), False),
    (Sum(), False),
    (TrimmedMean(trim_number=1), False),
]

try:
    from torchjd.aggregation import CAGrad

    _PARAMETRIZATIONS.append((CAGrad(c=0.5), True))
except ImportError:
    pass

try:
    from torchjd.aggregation import NashMTL

    _PARAMETRIZATIONS.append((NashMTL(n_tasks=2), False))
except ImportError:
    pass


@mark.parametrize("aggregator, expected", _PARAMETRIZATIONS)
def test_can_skip_jacobian_combination(aggregator: Aggregator, expected: bool):
    """
    Tests that _can_skip_jacobian_combination correctly identifies when optimization can be used.
    """

    assert _can_skip_jacobian_combination(aggregator) == expected
    handle = aggregator.register_forward_hook(lambda module, input, output: output)
    assert not _can_skip_jacobian_combination(aggregator)
    handle.remove()
    handle = aggregator.register_forward_pre_hook(lambda module, input: input)
    assert not _can_skip_jacobian_combination(aggregator)
    handle.remove()
    assert _can_skip_jacobian_combination(aggregator) == expected
