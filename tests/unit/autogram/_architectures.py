from functools import partial

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import Flatten, ReLU
from torch.utils._pytree import PyTree


class Cifar10Model(nn.Sequential):
    INPUT_SIZE = (3, 32, 32)
    OUTPUT_SIZE = (10,)

    def __init__(self):
        layers = [
            nn.Conv2d(3, 32, 3),
            ReLU(),
            nn.Conv2d(32, 64, 3, groups=32),
            nn.Sequential(nn.MaxPool2d(2), ReLU()),
            nn.Conv2d(64, 64, 3, groups=64),
            nn.Sequential(nn.MaxPool2d(3), ReLU(), Flatten()),
            nn.Linear(1024, 128),
            ReLU(),
            nn.Linear(128, 10),
        ]
        super().__init__(*layers)


class FlatNonSequentialNN(nn.Module):
    INPUT_SIZE = (9,)
    OUTPUT_SIZE = (514,)

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(9, 512)
        self.fc1 = nn.Linear(512, 513)
        self.fc2 = nn.Linear(513, 514)
        self.fc3 = nn.Linear(512, 514)

    def forward(self, input: Tensor) -> Tensor:
        common_input = self.relu(self.fc0(input))
        branch1 = self.fc2(self.relu(self.fc1(common_input)))
        branch2 = self.fc3(common_input)
        output = branch1 + branch2
        return output


class ModuleThatTakesString(nn.Module):
    INPUT_SIZE = (50,)
    OUTPUT_SIZE = (10,)

    def __init__(self):
        super().__init__()
        self.matrix1 = nn.Parameter(torch.randn(50, 10))
        self.matrix2 = nn.Parameter(torch.randn(50, 10))

    def forward(self, input: Tensor, string: str):
        if string == "test":
            return input @ self.matrix1
        else:
            return input @ self.matrix2


class ModelThatTakesString(nn.Module):
    INPUT_SIZE = (50,)
    OUTPUT_SIZE = (10,)

    def __init__(self):
        super().__init__()
        self.module = ModuleThatTakesString()

    def forward(self, input: Tensor):
        return self.module(input, "test") + self.module(input, "definitely not a test")


class MultiInputMultiOutputNN(nn.Module):
    INPUT_SIZE = (50,)
    OUTPUT_SIZE = ((60,), (70,))

    def __init__(self):
        super().__init__()
        self.matrix1 = nn.Parameter(torch.randn(50, 60))
        self.matrix2 = nn.Parameter(torch.randn(50, 70))

    def forward(self, *inputs: Tensor) -> tuple[Tensor, Tensor]:
        input = sum(inputs)
        return input @ self.matrix1, input @ self.matrix2


class MultiInputNN(nn.Module):
    INPUT_SIZE = (50,)
    OUTPUT_SIZE = (60,)

    def __init__(self):
        super().__init__()
        self.matrix1 = nn.Parameter(torch.randn(50, 60))

    def forward(self, *inputs: Tensor) -> tuple[Tensor, Tensor]:
        input = sum(inputs)
        return input @ self.matrix1


class SingleInputSingleOutputModel(nn.Module):
    INPUT_SIZE = (50,)
    OUTPUT_SIZE = (130,)

    def __init__(self):
        super().__init__()
        self.mimo = MultiInputMultiOutputNN()

    def forward(self, input: Tensor) -> Tensor:
        return torch.concatenate(list(self.mimo(input, input)), dim=1)


class SingleInputSingleOutputModel2(nn.Module):
    INPUT_SIZE = (50,)
    OUTPUT_SIZE = MultiInputNN.OUTPUT_SIZE

    def __init__(self):
        super().__init__()
        self.miso = MultiInputNN()

    def forward(self, input: Tensor) -> Tensor:
        return self.miso(input, input)


class PyTreeModule(nn.Module):
    INPUT_SIZE = (50,)
    OUTPUT_SIZE = {
        "first": ((50,), [(60,), (70,)]),
        "second": (80,),
        "third": ([((90,),)],),
    }

    def __init__(self):
        super().__init__()
        self.matrix1 = nn.Parameter(torch.randn(50, 50))
        self.matrix2 = nn.Parameter(torch.randn(50, 60))
        self.matrix3 = nn.Parameter(torch.randn(50, 70))
        self.matrix4 = nn.Parameter(torch.randn(50, 80))
        self.matrix5 = nn.Parameter(torch.randn(50, 90))

    def forward(self, input: Tensor) -> PyTree:
        return {
            "first": (input @ self.matrix1, [input @ self.matrix2, input @ self.matrix3]),
            "second": input @ self.matrix4,
            "third": ([(input @ self.matrix5,)],),
        }


class PyTreeModel(nn.Module):
    INPUT_SIZE = (50,)
    OUTPUT_SIZE = (350,)

    def __init__(self):
        super().__init__()
        self.pytree_module = PyTreeModule()

    def forward(self, input: Tensor):
        first, second, third = self.pytree_module(input).values()
        output1, output23 = first
        output2, output3 = output23
        output4 = second
        output5 = third[0][0][0]

        return torch.concatenate([output1, output2, output3, output4, output5], dim=1)


class ModuleWithParameterReuse(nn.Module):
    INPUT_SIZE = (50,)
    OUTPUT_SIZE = (10,)

    def __init__(self):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(50, 10))

    def forward(self, input: Tensor):
        return input @ self.matrix + (input**2) @ self.matrix


class MatMulModule(nn.Module):
    def __init__(self, matrix: nn.Parameter):
        super().__init__()
        self.matrix = matrix
        self.INPUT_SIZE = matrix.shape[0]

    def forward(self, input: Tensor):
        return input @ self.matrix


class ModelWithInterModuleParameterReuse(nn.Module):
    INPUT_SIZE = (50,)
    OUTPUT_SIZE = (10,)

    def __init__(self):
        super().__init__()
        matrix = nn.Parameter(torch.randn(50, 10))
        self.module1 = MatMulModule(matrix)
        self.module2 = MatMulModule(matrix)

    def forward(self, input: Tensor):
        return self.module1(input) + self.module2(input**2)


class ModelWithModuleReuse(nn.Module):
    INPUT_SIZE = (50,)
    OUTPUT_SIZE = (10,)

    def __init__(self):
        super().__init__()
        matrix = nn.Parameter(torch.randn(50, 10))
        self.module = MatMulModule(matrix)

    def forward(self, input: Tensor):
        return self.module(input) + self.module(input**2)


class ModelWithFreeParameter(nn.Module):
    INPUT_SIZE = (15,)
    OUTPUT_SIZE = (80,)

    def __init__(self):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(15, 16))  # Free parameter
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(16, 50)
        self.linear2 = nn.Linear(50, 60)
        self.linear3 = nn.Linear(60, 70)
        self.linear4 = nn.Linear(70, 80)

    def forward(self, input: Tensor):
        output = self.relu(input @ self.matrix)
        output = self.relu(self.linear1(output))
        output = self.relu(self.linear2(output))
        output = self.relu(self.linear3(output))
        output = self.linear4(output)
        return output


class ModelWithNoFreeParameter(nn.Module):
    INPUT_SIZE = (15,)
    OUTPUT_SIZE = (80,)

    def __init__(self):
        super().__init__()
        self.linear0 = nn.Linear(15, 16, bias=False)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(16, 50)
        self.linear2 = nn.Linear(50, 60)
        self.linear3 = nn.Linear(60, 70)
        self.linear4 = nn.Linear(70, 80)

    def forward(self, input: Tensor):
        output = self.relu(self.linear0(input))
        output = self.relu(self.linear1(output))
        output = self.relu(self.linear2(output))
        output = self.relu(self.linear3(output))
        output = self.linear4(output)
        return output


class ModuleWithUnusedParam(nn.Module):
    INPUT_SIZE = (50,)
    OUTPUT_SIZE = (10,)

    def __init__(self):
        super().__init__()
        self.unused_param = nn.Parameter(torch.randn(50, 10))
        self.matrix = nn.Parameter(torch.randn(50, 10))

    def forward(self, input: Tensor):
        return input @ self.matrix


class ModuleWithFrozenParam(nn.Module):
    INPUT_SIZE = (50,)
    OUTPUT_SIZE = (10,)

    def __init__(self):
        super().__init__()
        self.frozen_param = nn.Parameter(torch.randn(50, 10), requires_grad=False)
        self.matrix = nn.Parameter(torch.randn(50, 10))

    def forward(self, input: Tensor):
        return input @ self.matrix + (input**2) @ self.frozen_param


class ModuleWithBuffer(nn.Module):
    INPUT_SIZE = (27,)
    OUTPUT_SIZE = (27,)

    def __init__(self):
        super().__init__()
        self.buffer = nn.Buffer(torch.tensor(10.0))

    def forward(self, input: Tensor):
        return input * self.buffer


class ModelWithModuleWithBuffer(nn.Module):
    INPUT_SIZE = (27,)
    OUTPUT_SIZE = (10,)

    def __init__(self):
        super().__init__()
        self.module_with_buffer = ModuleWithBuffer()
        self.linear = nn.Linear(27, 10)

    def forward(self, input: Tensor):
        return self.linear(self.module_with_buffer(input))


class ResNet18(nn.Module):
    INPUT_SIZE = (3, 224, 224)
    OUTPUT_SIZE = (1000,)

    def __init__(self):
        super().__init__()
        self.resnet18 = torchvision.models.resnet18(
            norm_layer=partial(nn.InstanceNorm2d, track_running_stats=False, affine=True)
        )

    def forward(self, input: Tensor):
        return self.resnet18(input)
