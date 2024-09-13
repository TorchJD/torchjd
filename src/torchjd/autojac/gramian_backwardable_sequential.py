import torch
from torch import Tensor
from torch.nn import Module

from ._transform import AccumulateGramian, Diagonalize, Gradients, Jac, Select


class GramianBackwardableSequential(Module):
    def __init__(self, *modules: Module, parallel_chunk_size: int | None = None):
        super().__init__()
        self.modules = modules
        self.transform = None
        self.output = None
        self.gramian = None
        self.parallel_chunk_size = parallel_chunk_size

    def forward(self, input: Tensor) -> Tensor:
        current_tensor = input
        tensors = list()
        for model in self.modules:
            current_tensor = model(current_tensor)
            if not isinstance(current_tensor, Tensor):
                raise RuntimeError("Requires models that output `Tensors`.")
            tensors = [current_tensor] + tensors
        self.output = current_tensor
        if self.output.dim != 1:
            raise ValueError("Last module should return a vector")

        self.gramian = torch.zeros([self.output.shape[0]] * 2)
        accumulate_gramian = AccumulateGramian(self.modules[0].parameters(), self.gramian)
        jac = Jac(
            [tensors[0]],
            self.modules[0].parameters(),
            self.parallel_chunk_size,
            retain_graph=True,
        )
        self.transform = accumulate_gramian << jac
        for i in range(1, len(self.modules)):
            params = list(self.modules[i].parameters())
            keys = params + [tensors[i - 1]]
            select_previous = Select([tensors[i - 1]], keys)
            select_params = Select(params, keys)
            accumulate_gramian = AccumulateGramian(params, self.gramian)
            jac = Jac(
                [tensors[i]],
                keys,
                self.parallel_chunk_size,
                retain_graph=True,
            )
            self.transform = (
                self.transform << select_previous | accumulate_gramian << select_params
            ) << jac

        self.transform = self.transform << Diagonalize([self.output])

        return self.output

    def backward_gramian(self, grad_output: Tensor) -> Tensor:
        gradients = Gradients({self.output: grad_output})
        self.transform(gradients)
        return self.gramian
