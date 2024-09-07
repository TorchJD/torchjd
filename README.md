# ![image](docs/source/icons/favicon-32x32.png) TorchJD

[![Tests](https://github.com/TorchJD/torchjd/actions/workflows/tests.yml/badge.svg)](https://github.com/TorchJD/torchjd/actions/workflows/tests.yml)

TorchJD is a library enabling [Jacobian descent](https://arxiv.org/pdf/2406.16232) with PyTorch, to
train neural networks with multiple objectives. In particular, it can be used for multi-task
learning, with a wide variety of algorithms from the literature. It also enables the instance-wise
risk minimization paradigm, as proposed in
[Jacobian Descent For Multi-Objective Optimization](https://arxiv.org/pdf/2406.16232). The full
documentation is available at [torchjd.org](https://torchjd.org), with several usage examples.

## Installation
<!-- start installation -->
TorchJD can be installed directly with pip:
```bash
pip install torchjd
```
<!-- end installation -->

## Compatibility
TorchJD requires python 3.10, 3.11 or 3.12. It is only compatible with recent versions of PyTorch
(>= 2.0). For more information, read the `dependencies` in [pyproject.toml](./pyproject.toml).

## Usage

The main way to use TorchJD is to replace the usual call to `loss.backward()` by a call to
`torchjd.backward` or `torchjd.mtl_backward`, depending on the use-case.

The following example shows how to use TorchJD to train a multi-task model with Jacobian Descent,
using the [UPGrad](https://torchjd.org/docs/aggregation/upgrad/) aggregator.

```python
import torch
from torch.nn import Linear, MSELoss, ReLU, Sequential
from torch.optim import SGD

from torchjd import mtl_backward
from torchjd.aggregation import UPGrad

shared_module = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
task1_module = Linear(3, 1)
task2_module = Linear(3, 1)
params = [
    *shared_module.parameters(),
    *task1_module.parameters(),
    *task2_module.parameters(),
]

loss_fn = MSELoss()
optimizer = SGD(params, lr=0.1)
A = UPGrad()

inputs = torch.randn(8, 16, 10)  # 8 batches of 16 random input vectors of length 10
task1_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the first task
task2_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the second task

for input, target1, target2 in zip(inputs, task1_targets, task2_targets):
    features = shared_module(input)
    output1 = task1_module(features)
    output2 = task2_module(features)
    loss1 = loss_fn(output1, target1)
    loss2 = loss_fn(output2, target2)

    optimizer.zero_grad()
    mtl_backward(
        losses=[loss1, loss2],
        features=features,
        tasks_params=[task1_module.parameters(), task2_module.parameters()],
        shared_params=shared_module.parameters(),
        A=A,
    )
    optimizer.step()
```

> [!NOTE]
> In this example, the Jacobian is only with respect to the shared parameters. The task-specific
> parameters are simply updated via the gradient of their task’s loss with respect to them.

More usage examples can be found [here](https://torchjd.org/examples/).

## Contribution

Please read the [Contribution page](CONTRIBUTING.md).

## Citation
If you use TorchJD for your research, please cite:
```
@article{jacobian_descent,
  title={Jacobian Descent For Multi-Objective Optimization},
  author={Quinton, Pierre and Rey, Valérian},
  journal={arXiv preprint arXiv:2406.16232},
  year={2024}
}
```
