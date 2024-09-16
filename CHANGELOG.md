# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). This changelog does not include internal
changes that do not affect the user.

## [0.2.1] - 2024-09-17

### Changed

- Removed upper cap on numpy version in the dependencies. This makes `torchjd` compatible with
  the most recent numpy versions too.

### Fixed

- **BREAKING** Prevented IMTLG from dividing by zero during its weight rescaling step. If the input
  matrix consists only of zeros, it will now return a vector of zeros instead of a vector of `nan`.

## [0.2.0] - 2024-09-05

### Added

- `autojac` package containing the backward pass functions and their dependencies.
- `mtl_backward` function to make a backward pass for multi-task learning.
- Multi-task learning example.

### Changed

- **BREAKING**: Moved the `backward` module to the `autojac` package. Some imports may have to be
  adapted.
- Improved documentation of `backward`.

### Fixed

- Fixed wrong tensor device with `IMTLG` in some rare cases.
- **BREAKING**: Removed the possibility of populating the `.grad` field of a tensor that does not
  expect it when calling `backward`. If an input `t` provided to backward does not satisfy
  `t.requires_grad and (t.is_leaf or t.retains_grad)`, an error is now raised.
- **BREAKING**: When using `backward`, aggregations are now accumulated into the `.grad` fields
  of the inputs rather than replacing those fields if they already existed. This is in line with the
  behavior of `torch.autograd.backward`.

## [0.1.0] - 2024-06-22

### Added

- Basic project structure.
- `aggregation` package:
  - `Aggregator` base class to aggregate Jacobian matrices.
  - `AlignedMTL` from [Independent Component
      Alignment for Multi-Task Learning](
      https://openaccess.thecvf.com/content/CVPR2023/papers/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.pdf>).
  - `CAGrad` from [Conflict-Averse Gradient Descent for Multi-task
      Learning](https://arxiv.org/pdf/2110.14048.pdf).
  - `Constant` to aggregate with constant weights.
  - `DualProj` adapted from [Gradient Episodic
      Memory for Continual Learning](https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf).
  - `GradDrop` from [Just Pick a Sign: Optimizing Deep
      Multitask Models with Gradient Sign Dropout](https://arxiv.org/pdf/2010.06808.pdf).
  - `IMTLG` from [Towards Impartial Multi-task Learning](https://discovery.ucl.ac.uk/id/eprint/10120667/).
  - `Krum` from [Machine Learning with Adversaries: Byzantine
      Tolerant Gradient Descent](https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf).
  - `Mean` to average the rows of the matrix.
  - `MGDA` from [Multiple-gradient descent algorithm (MGDA) for multiobjective optimization](https://www.sciencedirect.com/science/article/pii/S1631073X12000738/pdf?md5=2622857e4abde98b6f7ddc8a13a337e1&pid=1-s2.0-S1631073X12000738-main.pdf>).
  - `NashMTL` from [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017.pdf).
  - `PCGrad` from [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782.pdf).
  - `Random` from [Reasonable Effectiveness of Random Weighting: A
      Litmus Test for Multi-Task Learning](https://arxiv.org/pdf/2111.10603.pdf).
  - `Sum` to sum the rows of the matrix.
  - `TrimmedMean` from [Byzantine-Robust Distributed Learning: Towards
      Optimal Statistical Rates](https://proceedings.mlr.press/v80/yin18a/yin18a.pdf).
  - `UPGrad` from [Jacobian Descent for Multi-Objective Optimization](https://arxiv.org/pdf/2406.16232).
- `backward` function to perform a step of Jacobian descent.
- Documentation of the public API and of some usage examples.
- Tests:
  - Unit tests.
  - Documentation tests.
  - Plotting utilities to verify qualitatively that aggregators work as expected.
