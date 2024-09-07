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
> [!NOTE]
> TorchJD requires python 3.10, 3.11 or 3.12. It is only compatible with recent versions of
> PyTorch (>= 2.0). For more information, read the `dependencies` in
> [pyproject.toml](./pyproject.toml).

## Supported Aggregators
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
