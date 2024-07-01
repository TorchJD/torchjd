from torch.testing import assert_close


def test_multi_task_backward():
    import torch

    from torchjd import multi_task_backward
    from torchjd.aggregation import UPGrad

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    # Compute an arbitrary representation that is function of the shared parameter
    r = torch.tensor([-1.0, 1.0]) * p0
    y1 = r @ p1
    y2 = r @ p2

    multi_task_backward(
        tasks_losses=[y1, y2],
        shared_parameters=[p0],
        shared_representations=r,
        tasks_parameters=[[p1], [p2]],
        A=UPGrad(),
    )

    assert_close(p0.grad, torch.tensor([-2.0, 3.0]), rtol=0.0, atol=1e-04)
    assert_close(p1.grad, torch.tensor([-1.0, 2.0]), rtol=0.0, atol=1e-04)
    assert_close(p2.grad, torch.tensor([-1.0, 2.0]), rtol=0.0, atol=1e-04)
