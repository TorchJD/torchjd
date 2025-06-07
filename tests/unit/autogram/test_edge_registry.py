from torch.autograd.graph import get_gradient_edge
from utils.tensors import randn_

from torchjd.autogram._edge_registry import EdgeRegistry


def test_all_edges_are_leaves():
    a = randn_([3, 4], requires_grad=True)
    b = randn_([4], requires_grad=True)
    c = randn_([3], requires_grad=True)

    d = (a @ b) + c

    edge_registry = EdgeRegistry()

    registered_edges = {
        a: get_gradient_edge(a),
        b: get_gradient_edge(b),
        c: get_gradient_edge(c),
    }

    expected_leaves = {registered_edges[a], registered_edges[b], registered_edges[c]}

    for edge in registered_edges.values():
        edge_registry.register(edge)

    leaves = edge_registry.get_leaf_edges({get_gradient_edge(d)}, set())

    assert leaves == expected_leaves


def test_some_edges_are_not_leaves1():
    a = randn_([3, 4], requires_grad=True)
    b = randn_([4], requires_grad=True)
    c = randn_([4], requires_grad=True)
    d = randn_([4], requires_grad=True)

    e = a * b
    f = e + c
    g = f + d

    edge_registry = EdgeRegistry()

    registered_edges = {
        a: get_gradient_edge(a),
        b: get_gradient_edge(b),
        c: get_gradient_edge(c),
        d: get_gradient_edge(d),
        e: get_gradient_edge(e),
        f: get_gradient_edge(f),
        g: get_gradient_edge(g),
    }

    expected_leaves = {
        registered_edges[a],
        registered_edges[b],
        registered_edges[c],
        registered_edges[d],
    }

    for edge in registered_edges.values():
        edge_registry.register(edge)

    leaves = edge_registry.get_leaf_edges({get_gradient_edge(g)}, set())

    assert leaves == expected_leaves


def test_some_edges_are_not_leaves2():
    a = randn_([3, 4], requires_grad=True)
    b = randn_([4], requires_grad=True)
    c = randn_([4], requires_grad=True)
    d = randn_([4], requires_grad=True)

    e = a * b
    f = e + c
    g = f + d

    edge_registry = EdgeRegistry()

    registered_edges = {
        a: get_gradient_edge(a),
        c: get_gradient_edge(c),
        d: get_gradient_edge(d),
        e: get_gradient_edge(e),
        g: get_gradient_edge(g),
    }

    expected_leaves = {
        registered_edges[a],
        registered_edges[c],
        registered_edges[d],
    }

    for edge in registered_edges.values():
        edge_registry.register(edge)

    leaves = edge_registry.get_leaf_edges({get_gradient_edge(g)}, set())

    assert leaves == expected_leaves
