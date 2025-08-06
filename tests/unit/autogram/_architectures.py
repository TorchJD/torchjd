from functools import partial

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import Flatten, ReLU
from torch.utils._pytree import PyTree


class ShapedModule(nn.Module):
    """Module class that guarantees subclasses to have INPUT_SHAPES and OUTPUT_SHAPES attributes."""

    INPUT_SHAPES: PyTree  # meant to be overridden
    OUTPUT_SHAPES: PyTree  # meant to be overridden

    def __init_subclass__(cls):
        super().__init_subclass__()
        if getattr(cls, "INPUT_SHAPES", None) is None:
            raise TypeError(f"{cls.__name__} must define INPUT_SHAPES")
        if getattr(cls, "OUTPUT_SHAPES", None) is None:
            raise TypeError(f"{cls.__name__} must define OUTPUT_SHAPES")


class Cifar10Model(ShapedModule):
    """
    Architecture for image classification on the CIFAR-10 dataset, similar to what we used in
    https://arxiv.org/pdf/2406.16232.
    """

    INPUT_SHAPES = (3, 32, 32)
    OUTPUT_SHAPES = (10,)

    def __init__(self):
        super().__init__()
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
        self.seq = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.seq(input)


class OverlyNestedModel(ShapedModule):
    """Model that contains many unnecessary levels of nested modules."""

    INPUT_SHAPES = (9,)
    OUTPUT_SHAPES = (14,)

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Sequential(
                nn.Sequential(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Linear(9, 10),
                            ReLU(),
                        ),
                        nn.Linear(10, 11),
                    ),
                    ReLU(),
                    nn.Linear(11, 12),
                ),
                ReLU(),
                nn.Linear(12, 13),
                ReLU(),
            ),
            nn.Linear(13, 14),
        )

    def forward(self, input: Tensor):
        return self.seq(input)


class MultiInputSingleOutputModule(ShapedModule):
    """Module that takes two inputs and returns one output."""

    INPUT_SHAPES = ((50,), (50,))
    OUTPUT_SHAPES = (60,)

    def __init__(self):
        super().__init__()
        self.matrix1 = nn.Parameter(torch.randn(50, 60))
        self.matrix2 = nn.Parameter(torch.randn(50, 60))

    def forward(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        input1, input2 = inputs
        output = input1 @ self.matrix1 + input2 @ self.matrix2
        return output


class MultiInputMultiOutputModule(ShapedModule):
    """Module that takes two inputs and returns two outputs that each depend on both inputs."""

    INPUT_SHAPES = ((50,), (50,))
    OUTPUT_SHAPES = ((60,), (70,))

    def __init__(self):
        super().__init__()
        self.matrix1_1 = nn.Parameter(torch.randn(50, 60))
        self.matrix2_1 = nn.Parameter(torch.randn(50, 60))
        self.matrix1_2 = nn.Parameter(torch.randn(50, 70))
        self.matrix2_2 = nn.Parameter(torch.randn(50, 70))

    def forward(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        input1, input2 = inputs
        output1 = input1 @ self.matrix1_1 + input2 @ self.matrix2_1
        output2 = input1 @ self.matrix1_2 + input2 @ self.matrix2_2
        return output1, output2


class SingleInputPyTreeOutputModule(ShapedModule):
    """Module taking a single input and returning a complex PyTree of tensors as output."""

    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = {
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


class PyTreeInputSingleOutputModule(ShapedModule):
    """Module taking a complex PyTree of tensors as input and returning a single output."""

    INPUT_SHAPES = {
        "one": [((10,), [(20,), (30,)]), (12,)],
        "two": (14,),
    }
    OUTPUT_SHAPES = (350,)

    def __init__(self):
        super().__init__()
        self.matrix1 = nn.Parameter(torch.randn(10, 50))
        self.matrix2 = nn.Parameter(torch.randn(20, 60))
        self.matrix3 = nn.Parameter(torch.randn(30, 70))
        self.matrix4 = nn.Parameter(torch.randn(12, 80))
        self.matrix5 = nn.Parameter(torch.randn(14, 90))

    def forward(self, inputs: PyTree) -> PyTree:
        input1 = inputs["one"][0][0]
        input2 = inputs["one"][0][1][0]
        input3 = inputs["one"][0][1][1]
        input4 = inputs["one"][1]
        input5 = inputs["two"]

        output1 = input1 @ self.matrix1
        output2 = input2 @ self.matrix2
        output3 = input3 @ self.matrix3
        output4 = input4 @ self.matrix4
        output5 = input5 @ self.matrix5
        output = torch.concatenate([output1, output2, output3, output4, output5], dim=1)

        return output


class PyTreeInputPyTreeOutputModule(ShapedModule):
    """
    Module taking a complex PyTree of tensors as input and returning a complex PyTree of tensors as
    output.
    """

    INPUT_SHAPES = {
        "one": [((10,), [(20,), (30,)]), (12,)],
        "two": (14,),
    }

    OUTPUT_SHAPES = {
        "first": ((50,), [(60,), (70,)]),
        "second": (80,),
        "third": ([((90,),)],),
    }

    def __init__(self):
        super().__init__()
        self.matrix1 = nn.Parameter(torch.randn(10, 50))
        self.matrix2 = nn.Parameter(torch.randn(20, 60))
        self.matrix3 = nn.Parameter(torch.randn(30, 70))
        self.matrix4 = nn.Parameter(torch.randn(12, 80))
        self.matrix5 = nn.Parameter(torch.randn(14, 90))

    def forward(self, inputs: PyTree) -> PyTree:
        input1 = inputs["one"][0][0]
        input2 = inputs["one"][0][1][0]
        input3 = inputs["one"][0][1][1]
        input4 = inputs["one"][1]
        input5 = inputs["two"]

        return {
            "first": (input1 @ self.matrix1, [input2 @ self.matrix2, input3 @ self.matrix3]),
            "second": input4 @ self.matrix4,
            "third": ([(input5 @ self.matrix5,)],),
        }


class EmptyOutputModule(nn.Module):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(shape))

    def forward(self, _: PyTree) -> None:
        return None


class SimpleBranchedModel(ShapedModule):
    """Model with one input and two branches that rejoin into one output."""

    INPUT_SHAPES = (9,)
    OUTPUT_SHAPES = (15,)

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(9, 13)
        self.fc1 = nn.Linear(13, 14)
        self.fc2 = nn.Linear(14, 15)
        self.fc3 = nn.Linear(13, 15)

    def forward(self, input: Tensor) -> Tensor:
        common_input = self.relu(self.fc0(input))
        branch1 = self.fc2(self.relu(self.fc1(common_input)))
        branch2 = self.fc3(common_input)
        output = branch1 + branch2
        return output


class MISOBranchedModel(ShapedModule):
    """
    Model taking a single input, branching it using a MultiInputSingleOutputModule, and returning
    its output.
    """

    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = MultiInputSingleOutputModule.OUTPUT_SHAPES

    def __init__(self):
        super().__init__()
        self.miso = MultiInputSingleOutputModule()

    def forward(self, input: Tensor) -> Tensor:
        return self.miso((input, input))


class MIMOBranchedModel(ShapedModule):
    """
    Model taking a single input, branching it using a MultiInputMultiOutputModule, and returning
    the concatenation of its outputs.
    """

    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = (130,)

    def __init__(self):
        super().__init__()
        self.mimo = MultiInputMultiOutputModule()

    def forward(self, input: Tensor) -> Tensor:
        return torch.concatenate(list(self.mimo((input, input))), dim=1)


class SIPOBranchedModel(ShapedModule):
    """
    Model taking a single input, branching it using a SingleInputPyTreeOutput, and returning the
    concatenation of its outputs.
    """

    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = (350,)

    def __init__(self):
        super().__init__()
        self.sipo = SingleInputPyTreeOutputModule()

    def forward(self, input: Tensor):
        first, second, third = self.sipo(input).values()
        output1, output23 = first
        output2, output3 = output23
        output4 = second
        output5 = third[0][0][0]

        return torch.concatenate([output1, output2, output3, output4, output5], dim=1)


class PIPOBranchedModel(ShapedModule):
    """
    Model taking a single input, splitting it, branching it using a PyTreeInputPyTreeOutput, and
    returning the concatenation of its outputs.
    """

    INPUT_SHAPES = (86,)
    OUTPUT_SHAPES = (350,)

    def __init__(self):
        super().__init__()
        self.pipo = PyTreeInputPyTreeOutputModule()

    def forward(self, input: Tensor) -> Tensor:
        input1 = input[:, 0:10]
        input2 = input[:, 10:30]
        input3 = input[:, 30:60]
        input4 = input[:, 60:72]
        input5 = input[:, 72:86]

        pytree_input = {
            "one": [(input1, [input2, input3]), input4],
            "two": input5,
        }

        pytree_output = self.pipo(pytree_input)

        first, second, third = pytree_output.values()
        output1, output23 = first
        output2, output3 = output23
        output4 = second
        output5 = third[0][0][0]

        return torch.concatenate([output1, output2, output3, output4, output5], dim=1)


class ModuleWithParameterReuse(ShapedModule):
    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = (10,)

    def __init__(self):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(50, 10))

    def forward(self, input: Tensor):
        return input @ self.matrix + (input**2) @ self.matrix


class MatMulModule(nn.Module):
    def __init__(self, matrix: nn.Parameter):
        super().__init__()
        self.matrix = matrix

    def forward(self, input: Tensor):
        return input @ self.matrix


class ModelWithInterModuleParameterReuse(ShapedModule):
    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = (10,)

    def __init__(self):
        super().__init__()
        matrix = nn.Parameter(torch.randn(50, 10))
        self.module1 = MatMulModule(matrix)
        self.module2 = MatMulModule(matrix)

    def forward(self, input: Tensor):
        return self.module1(input) + self.module2(input**2)


class ModelWithModuleReuse(ShapedModule):
    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = (10,)

    def __init__(self):
        super().__init__()
        matrix = nn.Parameter(torch.randn(50, 10))
        self.module = MatMulModule(matrix)

    def forward(self, input: Tensor):
        return self.module(input) + self.module(input**2)


class ModelWithFreeParameter(ShapedModule):
    INPUT_SHAPES = (15,)
    OUTPUT_SHAPES = (80,)

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


class ModelWithNoFreeParameter(ShapedModule):
    INPUT_SHAPES = (15,)
    OUTPUT_SHAPES = (80,)

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


class ModuleWithUnusedParam(ShapedModule):
    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = (10,)

    def __init__(self):
        super().__init__()
        self.unused_param = nn.Parameter(torch.randn(50, 10))
        self.matrix = nn.Parameter(torch.randn(50, 10))

    def forward(self, input: Tensor):
        return input @ self.matrix


class ModuleWithFrozenParam(ShapedModule):
    INPUT_SHAPES = (50,)
    OUTPUT_SHAPES = (10,)

    def __init__(self):
        super().__init__()
        self.frozen_param = nn.Parameter(torch.randn(50, 10), requires_grad=False)
        self.matrix = nn.Parameter(torch.randn(50, 10))

    def forward(self, input: Tensor):
        return input @ self.matrix + (input**2) @ self.frozen_param


class ModuleWithBuffer(ShapedModule):
    INPUT_SHAPES = (27,)
    OUTPUT_SHAPES = (27,)

    def __init__(self):
        super().__init__()
        self.buffer = nn.Buffer(torch.tensor(10.0))

    def forward(self, input: Tensor):
        return input * self.buffer


class ModelWithModuleWithBuffer(ShapedModule):
    INPUT_SHAPES = (27,)
    OUTPUT_SHAPES = (10,)

    def __init__(self):
        super().__init__()
        self.module_with_buffer = ModuleWithBuffer()
        self.linear = nn.Linear(27, 10)

    def forward(self, input: Tensor):
        return self.linear(self.module_with_buffer(input))


class ModelWithModuleWithoutOutput(ShapedModule):
    INPUT_SHAPES = (27,)
    OUTPUT_SHAPES = (10,)

    def __init__(self):
        super().__init__()
        self.module1 = EmptyOutputModule(self.INPUT_SHAPES + self.OUTPUT_SHAPES)
        matrix = nn.Parameter(torch.randn(self.INPUT_SHAPES + self.OUTPUT_SHAPES))
        self.module2 = MatMulModule(matrix)

    def forward(self, input: Tensor):
        self.module1(input)
        return self.module2(input)


class ResNet18(ShapedModule):
    INPUT_SHAPES = (3, 224, 224)
    OUTPUT_SHAPES = (1000,)

    def __init__(self):
        super().__init__()
        self.resnet18 = torchvision.models.resnet18(
            norm_layer=partial(nn.InstanceNorm2d, track_running_stats=False, affine=True)
        )

    def forward(self, input: Tensor):
        return self.resnet18(input)


class Cifar10ModelPart1(ShapedModule):
    INPUT_SHAPES = (3, 32, 32)
    OUTPUT_SHAPES = (1024,)

    def __init__(self):
        super().__init__()
        layers = [
            nn.Conv2d(3, 32, 3),
            ReLU(),
            nn.Conv2d(32, 64, 3, groups=32),
            nn.Sequential(nn.MaxPool2d(2), ReLU()),
            nn.Conv2d(64, 64, 3, groups=64),
            nn.Sequential(nn.MaxPool2d(3), ReLU(), Flatten()),
        ]
        self.seq = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.seq(input)


class Cifar10ModelPart2(ShapedModule):
    INPUT_SHAPES = (1024,)
    OUTPUT_SHAPES = (10,)

    def __init__(self):
        super().__init__()
        layers = [
            nn.Linear(1024, 128),
            ReLU(),
            nn.Linear(128, 10),
        ]
        self.seq = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.seq(input)
