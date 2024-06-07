from typing import Iterable

from torch import Tensor

from torchjd.aggregation import Aggregator
from torchjd.transform import Diagonalize, EmptyTensorDict, Init, Jac, Store, make_aggregation
from torchjd.transform.strategy import UnifyingStrategy


def backward(
    losses: Tensor,
    leaves: Iterable[Tensor],
    aggregator: Aggregator,
    parallel_chunk_size: int | None = None,
) -> None:
    """
    Computes the aggregation of the Jacobian of the ``losses`` wrt graph ``leaves``. Stores it in
    the ``.grad`` field of each leaf.

    .. admonition::
        Example

        The following code snippet showcases a simple usage of ``backward``.

        >>> import torch
        >>> from torch.nn import Sequential, Linear, ReLU, MSELoss
        >>>
        >>> from torchjd import backward
        >>> from torchjd.aggregation import WeightedAggregator, UPGradWrapper, MeanWeighting
        >>>
        >>> _ = torch.manual_seed(0)  # Set the seed to make this example deterministic
        >>>
        >>> model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
        >>> loss = MSELoss(reduction='none')
        >>>
        >>> W = UPGradWrapper(MeanWeighting())
        >>> A = WeightedAggregator(W)
        >>>
        >>> input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
        >>> target = input.sum(dim=1, keepdim=True)  # Batch of 16 targets
        >>>
        >>> output = model(input)
        >>> losses = loss(output, target)
        >>>
        >>> backward(losses, model.parameters(), A)

        The ``.grad`` field of each parameter of the model is now populated.

    :param losses: The losses to differentiate.
    :param leaves: The leaves of the graph to differentiate with respect to. These must have their
        ``requires_grad`` flag to ``True``.
    :param aggregator: Aggregator used for the aggregation of the Jacobian.
    :param parallel_chunk_size: The number of losses to differentiate simultaneously in the
        backward pass. If set to ``None``, all losses will be differentiated in parallel in one go.
        If set to `1`, all losses will be differentiated sequentially. A larger value results in
        faster differentiation, but also higher memory usage. Defaults to ``None``.
    """
    parameters = list(leaves)

    init = Init([losses])
    diag = Diagonalize([losses])
    jac = Jac([losses], parameters, chunk_size=parallel_chunk_size)
    aggregation = make_aggregation(UnifyingStrategy(aggregator, parameters))
    store = Store(parameters)

    transform = store << aggregation << jac << diag << init

    transform(EmptyTensorDict())
