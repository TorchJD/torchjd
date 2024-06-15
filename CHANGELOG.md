# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Basic project structure.
- `aggregation` package:
  - `Aggregator` base class to aggregate Jacobian matrices.
  - `Weighting` base class to get weights from Jacobian matrices.
  - `WeightedAggregator` for aggregators relying on weights.
  - `AlignedMTLWrapper` from [Independent Component
      Alignment for Multi-Task Learning](
      https://openaccess.thecvf.com/content/CVPR2023/papers/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.pdf>).
  - `CAGradWeighting` from [Conflict-Averse Gradient Descent for Multi-task
      Learning](https://arxiv.org/pdf/2110.14048.pdf).
  - `ConstantWeighting` to get constant weights.
  - `DualProjWrapper` adapted from [Gradient Episodic
      Memory for Continual Learning](https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf).
  - `GradDropAggregator` from [Just Pick a Sign: Optimizing Deep
      Multitask Models with Gradient Sign Dropout](https://arxiv.org/pdf/2010.06808.pdf).
  - `IMTLGWeighting` from [Towards Impartial Multi-task Learning](https://discovery.ucl.ac.uk/id/eprint/10120667/).
  - `KrumWeighting` from [Machine Learning with Adversaries: Byzantine
      Tolerant Gradient Descent](https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf).
  - `MeanWeighting` to get weights corresponding to an averaging.
  - `MGDAWeighting` from [Multiple-gradient descent algorithm (MGDA) for multiobjective optimization](https://www.sciencedirect.com/science/article/pii/S1631073X12000738/pdf?md5=2622857e4abde98b6f7ddc8a13a337e1&pid=1-s2.0-S1631073X12000738-main.pdf>).
  - `NashMTLWeighting` from [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017.pdf).
  - `NormalizingWrapper` to normalize the weights obtained by a wrapped `Weigthing`.
  - `PCGradWeighting` from [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782.pdf).
  - `RandomWeighting` from [Reasonable Effectiveness of Random Weighting: A
      Litmus Test for Multi-Task Learning](https://arxiv.org/pdf/2111.10603.pdf).
  - `SumWeighting` to get weights corresponding to a sum.
  - `TrimmedMeanAggregator` from [Byzantine-Robust Distributed Learning: Towards
      Optimal Statistical Rates](https://proceedings.mlr.press/v80/yin18a/yin18a.pdf).
  - `UPGradWrapper` from [Jacobian Descent for Multi-Objective Optimization](https://arxiv.org/search/?query=jacobian+descent+for+multi-objective+optimization&searchtype=all&source=header).
- `backward` function to perform a step of Jacobian descent.
- Documentation of the public API and of some usage examples.
- Tests:
  - Unit tests.
  - Documentation tests.
  - Computation time test.
  - Plotting utilities to verify qualitatively that aggregators work as expected.