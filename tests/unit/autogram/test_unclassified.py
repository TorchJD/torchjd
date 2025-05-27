import torch

from torchjd.autogram.differentiation_graph import get_node, topological_sort


def test_topological_sort():
    A = torch.randn([5, 5], requires_grad=True)
    A_T = A.T
    G = A @ A_T
    L, V = torch.linalg.eigh(G)
    x = V @ L

    outputs = [L, V, x]
    inputs = {A}

    roots = [get_node(output) for output in outputs]
    leaves = {get_node(input): input for input in inputs}

    sorted = tuple(topological_sort(roots, set(leaves.keys()), set()))

    expected_possibilities = {
        (x, L, V, G, A.T, A),
        (x, V, L, G, A.T, A),
    }
    expected_node_possibilities = {
        tuple([get_node(tensor) for tensor in possibility])
        for possibility in expected_possibilities
    }

    assert sorted in expected_node_possibilities
