Multi-Task Learning (MTL)
=========================



>>> import torch
>>> from torch.nn import Linear, MSELoss, ReLU, Sequential
>>> from torch.optim import SGD
>>>
>>> from torchjd import mtl_backward
>>> from torchjd.aggregation import UPGrad
>>>
>>> shared_module = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
>>> task1_module = Linear(3, 1)
>>> task2_module = Linear(3, 1)
>>> params = [*shared_module.parameters(), *task1_module.parameters(), *task2_module.parameters()]
>>>
>>> loss_fn = MSELoss()
>>> optimizer = SGD(params, lr=0.1)
>>>
>>> A = UPGrad()
>>>
>>> inputs = torch.randn(8, 16, 10)  # 8 batches of 16 input random vectors of length 10
>>> task1_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for first task
>>> task2_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for second task
>>>
>>> for input, target1, target2 in zip(inputs, task1_targets, task2_targets):
>>>     features = shared_module(input)
>>>     output1 = task1_module(features)
>>>     output2 = task2_module(features)
>>>     loss1 = loss_fn(output1, target1)
>>>     loss2 = loss_fn(output2, target2)
>>>
>>>     optimizer.zero_grad()
>>>     mtl_backward(
...         features=features,
...         losses=[loss1, loss2],
...         shared_params=shared_module.parameters(),
...         tasks_params=[task1_module.parameters(), task2_module.parameters()],
...         A=A,
...     )
>>>     optimizer.step()
