Multi-task Learning (MTL)
=========================


>>> import torch
>>> from torch.nn import Linear, MSELoss, ReLU, Sequential
>>> from torch.optim import SGD
>>>
>>> from torchjd import multi_task_backward
>>> from torchjd.aggregation import UPGrad
>>>
>>> shared_model = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
>>> task1_model = Linear(3, 1)
>>> task2_model = Linear(3, 1)
>>> parameters = (
...     list(shared_model.parameters())
...     + list(task1_model.parameters())
...     + list(task2_model.parameters())
... )
>>>
>>> loss_fn = MSELoss()
>>> optimizer = SGD(parameters, lr=0.1)
>>>
>>> A = UPGrad()
>>>
>>> input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
>>> target1 = torch.randn(16, 1)  # Batch of 16 targets for first task
>>> target2 = torch.randn(16, 1)  # Batch of 16 targets for second task
>>>
>>> shared_representation = shared_model(input)
>>> output1 = task1_model(shared_representation)
>>> output2 = task2_model(shared_representation)
>>> loss1 = loss_fn(output1, target1)
>>> loss2 = loss_fn(output2, target2)
>>>
>>> optimizer.zero_grad()
>>> multi_task_backward(
...     tasks_losses=[loss1, loss2],
...     shared_parameters=shared_model.parameters(),
...     shared_representations=shared_representation,
...     tasks_parameters=[task1_model.parameters(), task2_model.parameters()],
...     A=A,
>>> )
>>> optimizer.step()
