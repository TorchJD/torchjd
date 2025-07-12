import torch
from torch import Tensor, nn
from torchviz import make_dot


class Cifar10Model(nn.Sequential):
    def __init__(self):
        layers = [
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, groups=32),
            nn.Sequential(nn.MaxPool2d(2), nn.ReLU()),
            nn.Conv2d(64, 64, 3, groups=64),
            nn.Sequential(nn.MaxPool2d(3), nn.ReLU(), nn.Flatten()),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ]
        super().__init__(*layers)


class FlatNonSequentialNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(9, 10)
        self.fc1 = nn.Linear(10, 12)
        self.fc2 = nn.Linear(12, 15)
        self.fc3 = nn.Linear(10, 15)

    def forward(self, input: Tensor) -> Tensor:
        common_input = self.relu(self.fc0(input))
        branch1 = self.fc2(self.relu(self.fc1(common_input)))
        branch2 = self.fc3(common_input)
        output = branch1 + branch2
        return output


def main():
    batch_size = 64
    input_shape = (batch_size, 3, 32, 32)
    input = torch.randn(input_shape)
    target = torch.randint(0, 10, (batch_size,))

    model = Cifar10Model()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    output = model(input)
    losses = criterion(output, target)

    graph = make_dot(losses, params=dict(model.named_parameters()))
    graph.view()


def main2():
    batch_size = 64
    input_shape = (batch_size, 9)
    input = torch.randn(input_shape)

    model = FlatNonSequentialNN()
    output = model(input)

    graph = make_dot(output, params=dict(model.named_parameters()))
    graph.view()


if __name__ == "__main__":
    main2()
