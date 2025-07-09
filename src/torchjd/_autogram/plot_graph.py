import time

import torch
from torch import nn
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


def main():
    batch_size = 64
    input_shape = (batch_size, 3, 32, 32)
    input = torch.randn(input_shape)
    target = torch.randint(0, 10, (batch_size,))

    model = Cifar10Model()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    output = model(input)
    losses = criterion(output, target)

    def node_pre_hook_accumulate_grad(grad_outputs):
        print(f"Calling node AccumulateGrad with grad_outputs = {grad_outputs}")

    def node_pre_hook_relu(grad_outputs):
        print(f"Calling node ReLUBackward0 with grad_outputs = {grad_outputs}")

    def node_pre_hook_t_backward(grad_outputs):
        print(f"Calling node TBackward0 with grad_outputs = {grad_outputs}")

    def node_post_hook(grad_inputs, grad_outputs):
        print(grad_inputs)

    addmm_backward0 = losses.grad_fn.next_functions[0][0].next_functions[0][0]
    addmm_backward0.register_hook(node_post_hook)

    accumulate_grad = addmm_backward0.next_functions[0][0]
    relu_backward0 = addmm_backward0.next_functions[1][0]
    t_backward0 = addmm_backward0.next_functions[2][0]

    accumulate_grad.register_prehook(node_pre_hook_accumulate_grad)
    relu_backward0.register_prehook(node_pre_hook_relu)
    t_backward0.register_prehook(node_pre_hook_t_backward)

    graph = make_dot(
        losses, params=dict(model.named_parameters()), show_attrs=True, show_saved=True
    )
    graph.view()

    # torch.autograd.grad(losses, inputs=list(model[8].parameters()), grad_outputs=torch.ones_like(losses))
    torch.autograd.backward(losses, inputs=model[0].bias, grad_tensors=torch.ones_like(losses))


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.cuda_sync()
        self.start_time = time.perf_counter()
        return self  # This allows you to access the timer object within the 'with' block

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cuda_sync()
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        print(f"{self.name}: {self.elapsed_time:.5f} seconds")

    @staticmethod
    def cuda_sync():
        torch.cuda.synchronize()


def main2():
    size = 36000
    matrix_big = torch.randn((size, size), device="cuda", requires_grad=True)
    matrix_small = torch.randn((size, 1), device="cuda", requires_grad=True)

    product = matrix_big @ matrix_small
    loss = product.mean()

    graph = make_dot(loss, params={"matrix_big": matrix_big, "matrix_small": matrix_small})
    graph.view()

    # loss.backward(retain_graph=True)
    # for i in range(10):
    #     with Timer("Wrt all"):
    #         loss.backward(retain_graph=True)

    loss.backward(inputs=matrix_small, retain_graph=True)
    for i in range(10):
        with Timer("Wrt matrix_small only"):
            loss.backward(inputs=matrix_small, retain_graph=True)

    matrix_big.requires_grad_(False)
    product = matrix_big @ matrix_small
    loss = product.mean()

    loss.backward(inputs=matrix_small, retain_graph=True)
    for i in range(10):
        with Timer("With matrix big not on graph"):
            loss.backward(inputs=matrix_small, retain_graph=True)

    graph = make_dot(loss, params={"matrix_big": matrix_big, "matrix_small": matrix_small})
    graph.view()


if __name__ == "__main__":
    main2()
