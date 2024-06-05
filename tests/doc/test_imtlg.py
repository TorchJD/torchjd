from torch.testing import assert_close


def test_imtlg():
    from torch import tensor
    from torchjd.aggregation import WeightedAggregator, IMTLGWeighting

    W = IMTLGWeighting()
    A = WeightedAggregator(W)
    J = tensor([[-4., 1., 1.], [6., 1., 1.]])

    assert_close(A(J), tensor([0.0767, 1.0000, 1.0000]), rtol=0, atol=1e-4)
    assert_close(W(J), tensor([0.5923, 0.4077]), rtol=0, atol=1e-4)
