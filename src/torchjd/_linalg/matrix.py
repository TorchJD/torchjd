from typing import Annotated

from torch import Tensor

Matrix = Annotated[Tensor, "ndim=2"]
PSDMatrix = Annotated[Matrix, "Positive semi-definite"]
