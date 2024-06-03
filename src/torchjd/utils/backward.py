from typing import Iterable

from torch import Tensor

from torchjd.aggregation import Aggregator
from torchjd.transform import Diagonalize, EmptyTensorDict, Init, Jac, Store, make_aggregation
from torchjd.transform.strategy import UnifyingStrategy


def backward(
    tensor: Tensor,
    inputs: Iterable[Tensor],
    aggregator: Aggregator,
    parallel_chunk_size: int | None = None,
) -> None:
    """
    Computes the Jacobian of ``tensor`` with respect to ``inputs``. Computes its aggregation by
    ``A`` and stores it in the ``.grad`` fields of the inputs.

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

    :param tensor: The vector (1-dimensional tensor) to differentiate.
    :param inputs: The tensors with respect ot which the tensor values must be differentiated. These
        must have their ``requires_grad`` flag set to ``True``.
    :param aggregator: Aggregator to use for the aggregation of the Jacobian.
    :param parallel_chunk_size: The number of scalars to differentiate simultaneously in the
        backward pass. If set to ``None``, all coordinates of ``tensor`` will be differentiated in
        parallel at once. If set to `1`, all coordinates will be differentiated sequentially. A
        larger value results in faster differentiation, but also higher memory usage. Defaults to
        ``None``.
    """
    if not (parallel_chunk_size is None or parallel_chunk_size > 0):
        raise ValueError(
            f"`chunk_size` should be `None` or greater than `0`. (got {parallel_chunk_size})"
        )

    parameters = list(inputs)

    # Transform that creates gradients containing only ones
    init = Init([tensor])

    # Transform that turns the gradients into jacobians
    diag = Diagonalize([tensor])

    # Transform that computes the required jacobians
    jac = Jac([tensor], parameters, chunk_size=parallel_chunk_size)

    # Transform that defines the aggregation of the jacobians into gradients
    aggregation = make_aggregation(UnifyingStrategy(aggregator, parameters))

    # Transform that stores the gradients with respect to the model's parameters
    store = Store(parameters)

    backward_transform = store << aggregation << jac << diag << init

    backward_transform(EmptyTensorDict())
