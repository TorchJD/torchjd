import math

import torch
from torch import Tensor

from torchjd.aggregation.bases import _WeightedAggregator, _Weighting


class GradVac(_WeightedAggregator):
    r"""
    :class:~torchjd.aggregation.bases.Aggregator that implements the Gradient Vaccine (GradVac)
    strategy as described in
    "Gradient Vaccine: Investigating and Improving Multi-Task Optimization in Massively Multilingual Models"
    (<https://arxiv.org/pdf/2010.05874> :contentReference[oaicite:0]{index=0}>).

    If a constant target cosine similarity, :math:\phi^T, is provided via the `target` argument,
    that value is used. Otherwise (if `target` is `None`), the target is computed adaptively via an
    exponential moving average (EMA) as given in Equation (3) of the paper:

    .. math::

       \hat{\phi}{ij}^{(t)} = (1 - \beta)\,\hat{\phi}{ij}^{(t-1)} + \beta\,\phi_{ij}^{(t)},

    where :math:\beta is a decay rate.


    .. admonition::
        Example 1

        >>> from torch import tensor
        >>> from torchjd.aggregation import GradVac

        >>> A = GradVac()
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>> print(A(J))
        tensor([0.5924, 3.7914, 3.7914])

        Example 2

        from torch import tensor
        from torchjd.aggregation import GradVac

        A = GradVac(target=0)  # setting the target to 0 will produce PCGrad results.
        J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        print(A(J))
        tensor([0.5848, 3.8012, 3.8012])  # cf. PCGrad Example

        Example 3

        >>> from torch import tensor
        >>> from torchjd.aggregation import GradVac

        >>> A = GradVac(target=0.5)
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>> print(A(J))
        tensor([0.0596, 4.4696, 4.4696])



    """

    def __init__(self, target: float = None, beta: float = 1e-2):
        r"""
        :param target: The desired cosine similarity (`\phi^T`) between task gradients.
                       If `None`, an EMA is used to adaptively set the target per task pair.
        :param beta: The EMA decay rate (`\beta`); see Equation (3) in the paper.
        """
        super().__init__(weighting=_GradVacWeighting(target, beta))


class _GradVacWeighting(_Weighting):
    r"""
    :class:~torchjd.aggregation.bases._Weighting that computes weights using the Gradient Vaccine (GradVac)
    strategy as defined in Section 3.2 of
    "Gradient Vaccine: Investigating and Improving Multi-Task Optimization in Massively Multilingual Models"
    (<https://arxiv.org/pdf/2010.05874> :contentReference[oaicite:1]{index=1}>).

    For each task i (with gradient :math:g_i) and each other task j (with gradient :math:g_j),
    the cosine similarity is computed as

    .. math::

       \phi_{ij} = \frac{g_i \cdot g_j}{\|g_i\|\|g_j\|}.

    Then, if :math:\phi_{ij} is below the desired target, the gradient :math:g_i is adjusted
    along the direction of :math:g_j according to:

    .. math::

       g'i = g_i + \frac{\|g_i\|\Bigl(\phi^T\sqrt{1-\phi{ij}^2} - \phi_{ij}\sqrt{1-(\phi^T)^2}\Bigr)}
       {\|g_j\|\sqrt{1-(\phi^T)^2}}\,g_j.

    .. note::
    When no constant target is provided, the target is set adaptively for each pair using an EMA update,
    as described in Equation (3) of the paper.
    When the provided target constant is set to 0, the aggregator will behave exactly as PCGrad (which is GradVac's base case).
    Hence, GradVac can be conceived as a generalization of PCGrad.
    """

    def __init__(self, target: float = None, beta: float = 1e-2):
        super().__init__()
        self.constant_target = target  # if provided, a fixed value in [-1, 1]
        self.beta = beta
        self.ema = None  # will be initialized as a (dimension x dimension) tensor if target is not provided

    def forward(self, matrix: Tensor) -> Tensor:
        # Pre-compute all inner products between gradients:
        inner_products = matrix @ matrix.T

        device = matrix.device
        dtype = matrix.dtype
        cpu = torch.device("cpu")
        inner_products = inner_products.to(device=cpu)

        dimension = inner_products.shape[0]
        weights = torch.zeros(dimension, device=cpu, dtype=dtype)

        norms = torch.sqrt(torch.diag(inner_products) + 1e-8)

        if self.ema is None or self.ema.shape[0] != dimension:
            self.ema = torch.zeros((dimension, dimension), device=cpu, dtype=dtype)

        for i in range(dimension):
            permutation = torch.randperm(dimension)
            current_weights = torch.zeros(dimension, device=cpu, dtype=dtype)
            current_weights[i] = 1.0
            norm_i = norms[i]

            for j in permutation:
                if j == i:
                    continue
                norm_j = norms[j]

                phi = inner_products[i, j] / (norm_i * norm_j + 1e-8)

                # updating the exp. moving average target for the pair (i, j) (using Equation 3 from the paper):
                self.ema[i, j] = (1 - self.beta) * self.ema[i, j] + self.beta * phi

                # checj if a constant was provided and use it; else, use the EMA value.
                phi_target = (
                    self.constant_target if self.constant_target is not None else self.ema[i, j]
                )

                # if the observed similarity is lower than the target, compute an adjustment.
                if phi < phi_target:
                    # computing the adjustment factor (according to Equation (2) from the paper)
                    numerator = norm_i * (
                        phi_target * math.sqrt(max(0.0, 1 - phi * 2))
                        - phi * math.sqrt(max(0.0, 1 - phi_target * 2))
                    )
                    denominator = norm_j * math.sqrt(max(1e-8, 1 - phi_target**2))
                    factor = numerator / (denominator + 1e-8)

                    current_weights[j] += factor

            weights += current_weights

        return weights.to(device)
