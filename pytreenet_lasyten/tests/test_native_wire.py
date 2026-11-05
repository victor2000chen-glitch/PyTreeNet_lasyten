"""Transport protocol checks with nonpicklable/nonexportable tensor doubles.

These exercise the public hdf5_io protocol, independently of the compiled
LASyTen tests in test_native_installed.py. They require hdf5_io to be available.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from examples.mpi_bug_smoke import make_problem
from pytreenet_lasyten.time_evolution import _mpi_serialization as wire
from pytreenet_lasyten.time_evolution.bug import BUGConfig
from pytreenet_lasyten.time_evolution.mpi_bug import _SubtreeUpdate
from pytreenet_lasyten.util import lasyten_backend as lb


class ProtectedSpace:
    def __init__(self, charges, is_dual=False):
        self.charges = tuple(charges)
        self.is_dual = is_dual
        self.dim = len(charges)
    def __reduce_ex__(self, protocol):
        raise AssertionError("A native representation space was pickled")


class ProtectedNative:
    def __init__(self, blocks, legs=None, labels=None, hidden_state=2j):
        self.blocks = np.array(blocks, copy=True)
        self.legs = (tuple(ProtectedSpace(range(n), bool(i % 2)) for i, n in enumerate(blocks.shape))
                     if legs is None else tuple(legs))
        self.labels = list(labels) if labels is not None else [None] * blocks.ndim
        self.hidden_state = hidden_state
        self.shape = self.blocks.shape
        self.num_legs = self.blocks.ndim
    def __reduce_ex__(self, protocol):
        raise AssertionError("A native tensor was pickled")
    def __array__(self, *a, **k):
        raise AssertionError("A native tensor was exported to NumPy")
    def save_hdf5(self, saver, group, path):
        saver.save(self.blocks, path + "blocks")
        saver.save([(p.charges, p.is_dual) for p in self.legs], path + "legs")
        saver.save(self.labels, path + "labels")
        saver.save(self.hidden_state, path + "hidden_state")
    @classmethod
    def from_hdf5(cls, loader, group, path):
        return cls(loader.load(path + "blocks"),
                   [ProtectedSpace(charges, dual) for charges, dual in loader.load(path + "legs")],
                   loader.load(path + "labels"), loader.load(path + "hidden_state"))


class ProtectedArray:
    def __init__(self, native):
        self.native = native
    @property
    def shape(self):
        return self.native.shape
    @property
    def ndim(self):
        return self.native.num_legs
    @property
    def legs(self):
        return self.native.legs
    def __array__(self, *a, **k):
        raise AssertionError("A native wrapper was exported to NumPy")
    def __reduce_ex__(self, protocol):
        raise AssertionError("A native wrapper was pickled")


@pytest.fixture
def protected(monkeypatch):
    pytest.importorskip("hdf5_io")
    monkeypatch.setattr(lb, "LasyTenArray", ProtectedArray)
    monkeypatch.setattr(lb, "_lasyten", lambda: SimpleNamespace(Tensor=ProtectedNative))
    state, operator, _ = make_problem()
    for network in (state, operator):
        for key, tensor in network.tensors.data.items():
            array = ProtectedArray(ProtectedNative(tensor, labels=[f"leg{i}" for i in range(tensor.ndim)]))
            permutation = list(network.nodes[key].leg_permutation)
            network.tensors.data[key] = array
            network.nodes[key].link_tensor(array)
            network.nodes[key]._leg_permutation = permutation
    return state, operator


def assert_same(left, right):
    assert left.root_id == right.root_id
    assert left.orthogonality_center_id == right.orthogonality_center_id
    for key in left.nodes:
        a, b = left.tensors.data[key].native, right.tensors.data[key].native
        np.testing.assert_array_equal(a.blocks, b.blocks)
        assert a.labels == b.labels and a.hidden_state == b.hidden_state
        assert [(p.charges, p.is_dual) for p in a.legs] == [(p.charges, p.is_dual) for p in b.legs]
        assert left.nodes[key].leg_permutation == right.nodes[key].leg_permutation
        assert left.nodes[key].parent == right.nodes[key].parent
        assert left.nodes[key].children == right.nodes[key].children


def test_native_task_round_trip_never_pickles_or_exports_tensors(protected):
    state, operator = protected
    message = wire.encode_task("a", state, state, {("a", "r"): state.tensors.data["a"]},
                               operator, .01, BUGConfig(), dict(rtol=1e-10))
    assert message["codec"] == "hdf5"
    result = wire.decode_task(message)
    assert result[0] == "a" and result[5] == .01 and result[7] == dict(rtol=1e-10)
    assert_same(state, result[1]); assert_same(state, result[2]); assert_same(operator, result[4])
    np.testing.assert_array_equal(result[3]["a", "r"].native.blocks, state.tensors.data["a"].native.blocks)


def test_native_result_restores_node_spaces_labels_and_hidden_state(protected):
    state, _ = protected
    node = state.nodes["a"]
    tensor = state.tensors.data["a"]
    value = _SubtreeUpdate("a", "r", {"a": (node, tensor)}, tensor, tensor)
    restored = wire.decode_result(wire.encode_result(value))
    assert restored.child_id == "a" and restored.parent_id == "r"
    restored_node, array = restored.node_tensors["a"]
    assert restored_node.parent == node.parent and restored_node.children == node.children
    assert restored_node.leg_permutation == node.leg_permutation
    np.testing.assert_array_equal(array.native.blocks, tensor.native.blocks)
    assert array.native.labels == tensor.native.labels
    assert array.native.hidden_state == tensor.native.hidden_state
    assert restored_node.leg_space(0).charges == node.leg_space(0).charges
