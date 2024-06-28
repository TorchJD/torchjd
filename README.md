# ![image](docs/source/icons/favicon-32x32.png) TorchJD

TorchJD is a library enabling [Jacobian descent](https://arxiv.org/pdf/2406.16232) with PyTorch, to
train neural networks with multiple objectives. The full documentation is available at
[torchjd.org](https://torchjd.org).

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
