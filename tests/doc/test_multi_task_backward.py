from torch.testing import assert_close


def test_multi_task_backward():
    import torch

    from torchjd import multi_task_backward
    from torchjd.aggregation import UPGrad

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)
    shared_parameters = [p0]
    tasks_parameters = [[p1], [p2]]

    # Compute arbitrary quantities that are function of the parameters
    r1 = torch.tensor([-1.0, 1.0]) @ p0
    r2 = (p0**2).sum() + p0.norm()
    shared_representations = [r1, r2]

    l1 = torch.stack((r1 * p1[0], r2 * p1[1]))
    l2 = r1 * p2[0] + r2 * p2[1]
    tasks_losses = [l1, l2]

    multi_task_backward(
        tasks_losses=tasks_losses,
        shared_parameters=shared_parameters,
        shared_representations=shared_representations,
        tasks_parameters=tasks_parameters,
        A=UPGrad(),
    )

    assert_close(p0.grad, torch.tensor([5.3416, 16.6833]), rtol=0.0, atol=1e-04)
    assert_close(p1.grad, torch.tensor([1.0000, 7.2361]), rtol=0.0, atol=1e-04)
    assert_close(p2.grad, torch.tensor([1.0000, 7.2361]), rtol=0.0, atol=1e-04)
