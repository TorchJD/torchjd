from torch.ops import aten  # type: ignore

from torchjd.sparse import DiagonalSparseTensor

# pointwise functions applied to one Tensor with `0.0 → 0`
_POINTWISE_FUNCTIONS = [
    aten.abs.default,
    aten.absolute.default,
    aten.asin.default,
    aten.asinh.default,
    aten.atan.default,
    aten.atanh.default,
    aten.ceil.default,
    aten.erf.default,
    aten.erfinv.default,
    aten.expm1.default,
    aten.fix.default,
    aten.floor.default,
    aten.hardtanh.default,
    aten.leaky_relu.default,
    aten.log1p.default,
    aten.neg.default,
    aten.negative.default,
    aten.positive.default,
    aten.relu.default,
    aten.round.default,
    aten.sgn.default,
    aten.sign.default,
    aten.sin.default,
    aten.sinh.default,
    aten.sqrt.default,
    aten.square.default,
    aten.tan.default,
    aten.tanh.default,
    aten.trunc.default,
]

_IN_PLACE_POINTWISE_FUNCTIONS = [
    aten.abs_.default,
    aten.absolute_.default,
    aten.asin_.default,
    aten.asinh_.default,
    aten.atan_.default,
    aten.atanh_.default,
    aten.ceil_.default,
    aten.erf_.default,
    aten.erfinv_.default,
    aten.expm1_.default,
    aten.fix_.default,
    aten.floor_.default,
    aten.hardtanh_.default,
    aten.leaky_relu_.default,
    aten.log1p_.default,
    aten.neg_.default,
    aten.negative_.default,
    aten.relu_.default,
    aten.round_.default,
    aten.sgn_.default,
    aten.sign_.default,
    aten.sin_.default,
    aten.sinh_.default,
    aten.sqrt_.default,
    aten.square_.default,
    aten.tan_.default,
    aten.tanh_.default,
    aten.trunc_.default,
]


def _override_pointwise(op):
    @DiagonalSparseTensor.implements(op)
    def func_(t: DiagonalSparseTensor) -> DiagonalSparseTensor:
        assert isinstance(t, DiagonalSparseTensor)
        return DiagonalSparseTensor(op(t.physical), t.v_to_ps)

    return func_


def _override_inplace_pointwise(op):
    @DiagonalSparseTensor.implements(op)
    def func_(t: DiagonalSparseTensor) -> DiagonalSparseTensor:
        assert isinstance(t, DiagonalSparseTensor)
        op(t.physical)
        return t


for pointwise_func in _POINTWISE_FUNCTIONS:
    _override_pointwise(pointwise_func)

for pointwise_func in _IN_PLACE_POINTWISE_FUNCTIONS:
    _override_inplace_pointwise(pointwise_func)


@DiagonalSparseTensor.implements(aten.pow.Tensor_Scalar)
def pow_Tensor_Scalar(t: DiagonalSparseTensor, exponent: float) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)

    if exponent <= 0.0:
        # Need to densify because we don't have pow(0.0, exponent) = 0.0
        return aten.pow.Tensor_Scalar(t.to_dense(), exponent)

    new_physical = aten.pow.Tensor_Scalar(t.physical, exponent)
    return DiagonalSparseTensor(new_physical, t.v_to_ps)


# Somehow there's no pow_.Tensor_Scalar and pow_.Scalar takes tensor and scalar.
@DiagonalSparseTensor.implements(aten.pow_.Scalar)
def pow__Scalar(t: DiagonalSparseTensor, exponent: float) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)

    if exponent <= 0.0:
        # Need to densify because we don't have pow(0.0, exponent) = 0.0
        # Note sure if it's even possible to densify in-place, so let's just raise an error.
        raise ValueError(f"in-place pow with an exponent of {exponent} (<= 0) is not supported.")

    aten.pow_.Scalar(t.physical, exponent)
    return t


@DiagonalSparseTensor.implements(aten.div.Scalar)
def div_Scalar(t: DiagonalSparseTensor, divisor: float) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)

    new_physical = aten.div.Scalar(t.physical, divisor)
    return DiagonalSparseTensor(new_physical, t.v_to_ps)
