# ![image](docs/source/icons/favicon-32x32.png) TorchJD

[![Tests](https://github.com/TorchJD/torchjd/actions/workflows/tests.yml/badge.svg)](https://github.com/TorchJD/torchjd/actions/workflows/tests.yml)

TorchJD is a library extending autograd to enable
[Jacobian descent](https://arxiv.org/pdf/2406.16232) with PyTorch. In can be used to train neural
networks with multiple objectives. In particular, it supports multi-task learning, with a wide
variety of aggregators from the literature. It also enables the instance-wise risk minimization
paradigm. The full documentation is available at [torchjd.org](https://torchjd.org), with several usage examples.

## Installation
<!-- start installation -->
TorchJD can be installed directly with pip:
```bash
pip install torchjd
```
<!-- end installation -->
> [!NOTE]
> TorchJD requires python 3.10, 3.11 or 3.12. It is only compatible with recent versions of
> PyTorch (>= 2.0). For more information, read the `dependencies` in
> [pyproject.toml](./pyproject.toml).

## Usage

The main way to use TorchJD is to replace the usual call to `loss.backward()` by a call to
`torchjd.backward` or `torchjd.mtl_backward`, depending on the use-case.

The following example shows how to use TorchJD to train a multi-task model with Jacobian Descent,
using [UPGrad](https://torchjd.org/docs/aggregation/upgrad/).

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

## Supported Aggregators
TorchJD provides many existing aggregators from the literature, listed in the following table.

<!-- recommended aggregators first, then alphabetical order -->
| Aggregator                                                           | Publication                                                                                                                                                         |
|----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [UPGrad](https://torchjd.org/docs/aggregation/upgrad/) (recommended) | [Jacobian Descent For Multi-Objective Optimization](https://arxiv.org/pdf/2406.16232)                                                                               |
| [AlignedMTL](https://torchjd.org/docs/aggregation/aligned_mtl/)      | [Independent Component Alignment for Multi-Task Learning](https://arxiv.org/pdf/2305.19000)                                                                         |
| [CAGrad](https://torchjd.org/docs/aggregation/cagrad/)               | [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/pdf/2110.14048)                                                                        |
| [Constant](https://torchjd.org/docs/aggregation/constant/)           | -                                                                                                                                                                   |
| [DualProj](https://torchjd.org/docs/aggregation/dualproj/)           | [Gradient Episodic Memory for Continual Learning](https://arxiv.org/pdf/1706.08840)                                                                                 |
| [GradDrop](https://torchjd.org/docs/aggregation/graddrop/)           | [Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout](https://arxiv.org/pdf/2010.06808)                                                   |
| [IMTL-G](https://torchjd.org/docs/aggregation/imtl_g/)               | [Towards Impartial Multi-task Learning](https://discovery.ucl.ac.uk/id/eprint/10120667/)                                                                            |
| [Krum](https://torchjd.org/docs/aggregation/krum/)                   | [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf) |
| [Mean](https://torchjd.org/docs/aggregation/mean/)                   | -                                                                                                                                                                   |
| [MGDA](https://torchjd.org/docs/aggregation/mgda/)                   | [Multiple-gradient descent algorithm (MGDA) for multiobjective optimization](https://www.sciencedirect.com/science/article/pii/S1631073X12000738)                   |
| [Nash-MTL](https://torchjd.org/docs/aggregation/nash_mtl/)           | [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017)                                                                                        |
| [PCGrad](https://torchjd.org/docs/aggregation/pcgrad/)               | [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782)                                                                                        |
| [Random](https://torchjd.org/docs/aggregation/random/)               | [Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning](https://arxiv.org/pdf/2111.10603)                                             |
| [Sum](https://torchjd.org/docs/aggregation/sum/)                     | -                                                                                                                                                                   |
| [Trimmed Mean](https://torchjd.org/docs/aggregation/trimmed_mean/)   | [Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates](https://proceedings.mlr.press/v80/yin18a/yin18a.pdf)                                     |

The following example shows how to instantiate
[UPGrad](https://torchjd.org/docs/aggregation/upgrad/) and aggregate a simple matrix `J` with it.
```python
from torch import tensor
from torchjd.aggregation import UPGrad

A = UPGrad()
J = tensor([[-4., 1., 1.], [6., 1., 1.]])

A(J)
# Output: tensor([0.2929, 1.9004, 1.9004])
```

> [!TIP]
> When using TorchJD, you generally don't have to use aggregators directly. You simply instantiate
> one and pass it to the backward function (`torchjd.backward` or `torchjd.mtl_backward`), which
> will in turn apply it to the Jacobian matrix that it will compute.

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
