"""Recursive BUG updates against an independent three-spin dense Hamiltonian."""

from copy import deepcopy
from fractions import Fraction
import importlib
import os

import numpy as np
import pytest
from scipy.linalg import expm

PACKAGE = "pytreenet_lasyten"
Node = importlib.import_module(PACKAGE + ".core.node").Node
operators = importlib.import_module(PACKAGE + ".operators")
Hamiltonian, TensorProduct = operators.Hamiltonian, operators.TensorProduct
TTNO = importlib.import_module(PACKAGE + ".ttno").TreeTensorNetworkOperator
TTNOFinder = importlib.import_module(PACKAGE + ".ttno.state_diagram").TTNOFinder
TTNS = importlib.import_module(PACKAGE + ".ttns").TreeTensorNetworkState
bug = importlib.import_module(PACKAGE + ".time_evolution.bug")
BUG, BUGConfig = bug.BUG, bug.BUGConfig
TimeEvoMode = importlib.import_module(PACKAGE + ".time_evolution.time_evolution").TimeEvoMode
common_bug = importlib.import_module(PACKAGE + ".time_evolution.time_evo_util.common_bug")
root_update, CommonBUGConfig = common_bug.root_update, common_bug.CommonBUGConfig
soc = importlib.import_module(PACKAGE + ".contractions.state_operator_contraction")


def fixture(seed=9):
    rng = np.random.default_rng(seed)

    def random(shape):
        return rng.normal(size=shape) + 1j*rng.normal(size=shape)

    tree = TTNS()
    tree.add_root(Node(identifier="r"), random((2, 2, 1)))
    tree.add_child_to_parent(Node(identifier="a"), random((2, 2)), 0, "r", 0)
    tree.add_child_to_parent(Node(identifier="s"), random((2, 2, 2, 1)), 0, "r", 1)
    tree.add_child_to_parent(Node(identifier="b"), random((2, 2)), 0, "s", 1)
    tree.add_child_to_parent(Node(identifier="c"), random((2, 2)), 0, "s", 2)
    tree.canonical_form("r")
    tree.normalise()
    x = np.array([[0, 1], [1, 0]], complex)
    z = np.diag([1, -1]).astype(complex)
    y = np.array([[0, -1j], [1j, 0]])
    terms = [(.3, {"a": "x", "b": "x"}), (.2, {"a": "z", "c": "z"}),
             (-.13, {"b": "y", "c": "y"}), (.17, {"a": "z"}),
             (-.11, {"b": "x"}), (.09, {"c": "y"})]
    conversion = dict(x=x, y=y, z=z, I1=np.eye(1), I2=np.eye(2))
    h = Hamiltonian([(Fraction(1), f"t{i}", TensorProduct(term))
                     for i, (_, term) in enumerate(terms)],
                    conversion_dictionary=conversion,
                    coeffs_mapping={f"t{i}": coefficient for i, (coefficient, _) in enumerate(terms)})
    operator = TTNO.from_hamiltonian(h, tree, method=TTNOFinder.SGE)
    dense = np.zeros((8, 8), complex)
    for coefficient, term in terms:
        matrix = np.array([[1]])
        for site in ("a", "b", "c"):
            matrix = np.kron(matrix, conversion[term.get(site, "I2")])
        dense += coefficient * matrix
    return tree, operator, dense


def vector(tree):
    return np.einsum('asx,ay,sbcz,bv,cw->xyzvw',
                     tree.tensors["r"], tree.tensors["a"], tree.tensors["s"],
                     tree.tensors["b"], tree.tensors["c"]).reshape(-1)


@pytest.mark.parametrize("deep", [False, True])
@pytest.mark.parametrize("dt", [0., .03])
def test_recursive_update_against_full_dense(deep, dt):
    tree, h, dense = fixture()
    before = vector(tree).copy()
    op_topology = {key: (node.parent, tuple(node.children)) for key, node in h.nodes.items()}
    mode = TimeEvoMode.EXPM if dt == 0 else TimeEvoMode.RK45
    configuration = CommonBUGConfig(deep=deep, time_evo_mode=mode)
    updated = root_update(tree, h, dt, configuration,
                           solver_options=dict(rtol=1e-11, atol=1e-13))
    np.testing.assert_allclose(vector(updated), expm(-1j*dt*dense) @ before,
                               rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(vector(tree), before, atol=1e-13)
    assert updated.orthogonality_center_id == "r"
    assert not any(key.endswith("_basis_change_tensor") for key in updated.nodes)
    assert op_topology == {key: (node.parent, tuple(node.children)) for key, node in h.nodes.items()}


def test_full_numpy_bug_step_with_truncation():
    tree, h, dense = fixture(11)
    before = vector(tree).copy()
    config = BUGConfig(max_bond_dim=8, rel_tol=0, total_tol=0,
                        time_evo_mode=TimeEvoMode.RK45)
    evolution = BUG(tree, h, .03, .03, {}, config=config,
                     solver_options=dict(rtol=1e-11, atol=1e-13))
    evolution.run_one_time_step()
    np.testing.assert_allclose(vector(evolution.state), expm(-1j*.03*dense) @ before,
                               rtol=2e-9, atol=2e-11)


def test_solver_options_reach_every_recursive_node(monkeypatch):
    tree, h, _ = fixture()
    original = common_bug.single_site_time_evolution
    original_local = common_bug.effective_single_site_evolution
    seen = []
    options = dict(rtol=1e-11, atol=1e-13)

    def checked(node_id, *args, **kwargs):
        seen.append(node_id)
        assert kwargs["solver_options"] is options
        return original(node_id, *args, **kwargs)

    def checked_local(tensor, node, *args, **kwargs):
        seen.append(node.identifier)
        for key, value in options.items():
            assert kwargs[key] == value
        return original_local(tensor, node, *args, **kwargs)

    monkeypatch.setattr(common_bug, "single_site_time_evolution", checked)
    monkeypatch.setattr(common_bug, "effective_single_site_evolution", checked_local)
    config = CommonBUGConfig(time_evo_mode=TimeEvoMode.RK45)
    root_update(tree, h, .03, config, solver_options=options)
    assert sorted(seen) == sorted(tree.nodes)


@pytest.mark.parametrize("order", [soc.FirstContraction.KET, soc.FirstContraction.OPERATOR])
def test_temporary_parent_works_for_both_contraction_orders(monkeypatch, order):
    tree, h, dense = fixture()
    before = vector(tree).copy()
    monkeypatch.setattr(soc, "compare_contr_orders", lambda *args: order)
    config = CommonBUGConfig(time_evo_mode=TimeEvoMode.RK45)
    updated = root_update(tree, h, .03, config, solver_options=dict(rtol=1e-11, atol=1e-13))
    np.testing.assert_allclose(vector(updated), expm(-1j*.03*dense) @ before,
                               rtol=2e-9, atol=2e-11)
