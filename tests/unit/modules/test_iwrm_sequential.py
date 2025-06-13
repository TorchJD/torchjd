import torch
from torch import nn
from torch.nn import ReLU

from torchjd._modules.iwrm_sequential import IWRMSequential
from torchjd.aggregation._mean import _MeanWeighting
from torchjd.aggregation._upgrad import _UPGradWrapper


def test_algo_3():

    batch_size = 4
    input_shape = (batch_size, 3, 32, 32)
    input = torch.randn(input_shape)
    target = torch.randint(0, 10, (batch_size,))

    weighting = _UPGradWrapper(_MeanWeighting(), 0.0001, 0.0001, "quadprog")

    layers = [
        nn.Conv2d(3, 32, 3),
        ReLU(),
        nn.Conv2d(32, 64, 3, groups=32),
        nn.MaxPool2d(2),
        ReLU(),
        nn.Conv2d(64, 64, 3, groups=64),
        nn.MaxPool2d(3),
        ReLU(),
        nn.Flatten(),
        nn.Linear(1024, 128),
        ReLU(),
        nn.Linear(128, 10),
    ]

    model = IWRMSequential(weighting, layers)
    criterion = torch.nn.CrossEntropyLoss()

    output = model(input)
    loss = criterion(output, target)

    loss.backward()
