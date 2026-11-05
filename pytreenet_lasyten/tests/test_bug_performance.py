from types import SimpleNamespace
import importlib
import numpy as np
import pytest
from pytreenet_lasyten.util import bug_performance as perf
from fake_native import backend, fixture, data, Charged


@pytest.mark.parametrize("v", range(1, 5))
@pytest.mark.parametrize("p", range(3))
def test_generic_environment_layouts(v, p):
    rng = np.random.default_rng(7 + v + p)
    dims = {i: int(rng.integers(1, 4)) for i in range(3 * v + 2 * p)}
    k = tuple(range(v))
    b = tuple(range(v, 2 * v))
    w = tuple(range(2 * v, 3 * v))
    pi = tuple(range(3 * v, 3 * v + p))
    po = tuple(range(3 * v + p, 3 * v + 2 * p))
    ignored = v // 2
    labels = [k + pi, tuple(reversed(w)) + po + pi, tuple(reversed(b)) + po]
    labels += [(k[i], w[i], b[i]) for i in range(v) if i != ignored]
    output = (k[ignored], w[ignored], b[ignored])
    operands = [
        rng.normal(size=tuple(dims[i] for i in ls))
        + 1j * rng.normal(size=tuple(dims[i] for i in ls))
        for ls in labels
    ]
    args = []
    for a, ls in zip(operands, labels):
        args.extend([a, list(ls)])
    expected = np.einsum(*args, list(output), optimize=True)
    plan = perf.plan_chain(
        tuple(map(tuple, labels)), tuple(a.shape for a in operands), tuple(output), 1
    )
    out = operands[0]
    for i, a, b in plan[0]:
        out = np.tensordot(out, operands[i], axes=(a, b))
    np.testing.assert_allclose(out.transpose(plan[1]), expected, atol=1e-9, rtol=2e-12)


@pytest.mark.parametrize("v", [1, 2, 3])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("charged", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
def test_native_environment_with_hidden_charge(v, p, charged, permuted, monkeypatch):
    lb, ct = backend()
    lb._axes = lambda axes, ndim: tuple(int(i) % ndim for i in axes)
    lb.conjugate = lambda a: lb.LasyTenArray(
        ct.permute_legs(ct.dagger(a.native), list(reversed(range(a.ndim))), [])
    )

    def dot(a, b, axes):
        return lb.LasyTenArray(
            ct.tdot(a.native, b.native, list(axes[0]), list(axes[1]))
        )

    monkeypatch.setattr(
        lb.LasyTenArray,
        "__mul__",
        lambda a, c: lb.LasyTenArray(a.native * c),
        raising=False,
    )
    util = importlib.import_module("pytreenet_lasyten.util")
    soc = importlib.import_module(
        "pytreenet_lasyten.contractions.state_operator_contraction"
    )
    monkeypatch.setattr(util, "lasyten_backend", lb)
    monkeypatch.setattr(soc, "lb", lb)
    monkeypatch.setattr(
        soc, "is_lasyten_tensor", lambda x: isinstance(x, lb.LasyTenArray)
    )
    monkeypatch.setattr(soc, "_dispatch_tensordot", dot)
    state, node, ham, hnode, cache, envs = fixture(
        lb,
        v,
        p,
        charge=int(p == 2),
        ham_order=list(reversed([f"v{i}" for i in range(v)])),
        charged=charged,
    )
    neighbours = node.neighbouring_nodes()
    ignored = neighbours[v // 2]
    bra = lb.conjugate(state)
    bnode = node
    optrafo = bratrafo = None
    if permuted:
        hids = ["H" + n for n in reversed(neighbours)]
        hnode = SimpleNamespace(
            neighbouring_nodes=lambda: hids,
            nopen_legs=lambda: 2,
            neighbour_index=lambda n: hids.index(n),
            nneighbours=lambda: v,
        )
        bra = lb.transpose(bra, list(reversed(range(v))) + [v])
        bids = ["B" + n for n in reversed(neighbours)]
        bnode = SimpleNamespace(neighbouring_nodes=lambda: bids, nopen_legs=lambda: 1)
        optrafo = lambda n: "H" + n
        bratrafo = lambda n: "B" + n
    before = [
        (data(x).copy(), list(x.native.labels))
        for x in [state, ham, bra, *envs.values()]
    ]
    ns, k, b, w, pi, po, hl = perf._layout(node, hnode, optrafo)
    bmap = dict(zip([bratrafo(n) if bratrafo else n for n in ns], b))
    labels = [k + pi, hl, tuple(bmap[n] for n in bnode.neighbouring_nodes()) + po]
    arrays = [data(state), data(ham), data(bra)]
    for i, n in enumerate(ns):
        if n != ignored:
            labels.append((k[i], w[i], b[i]))
            arrays.append(data(envs[n]))
    j = ns.index(ignored)
    args = []
    for a, ls in zip(arrays, labels):
        args.extend([a, list(ls)])
    expected = np.einsum(*args, [k[j], w[j], b[j]], optimize=True)
    actual = perf.contract_environment(
        ignored, node, state, hnode, ham, cache, bnode, bra, optrafo, bratrafo
    )
    np.testing.assert_allclose(data(actual), expected, atol=2e-10, rtol=2e-12)
    assert not isinstance(actual.native, Charged)
    for x, (values, labels) in zip([state, ham, bra, *envs.values()], before):
        np.testing.assert_array_equal(data(x), values)
        assert list(x.native.labels) == labels


def test_path_prediction_without_allocations():
    labels = ((0, 1, 2, 9), (6, 7, 8, 10, 9), (0, 6, 3), (1, 7, 4), (2, 8, 5))
    shapes = ((50, 50, 50, 1), (20, 20, 20, 1, 1), *((50, 20, 50),) * 3)
    plan = perf.plan_chain(labels, shapes, (3, 4, 5, 10), 16_777_216)
    assert plan[2] == 50_000_000 < 50**3 * 20**3


def test_lazy_copies_do_not_modify_source(monkeypatch):
    from test_bug_regression import fixture, vector
    from pytreenet_lasyten.time_evolution.bug import BUG, BUGConfig
    from pytreenet_lasyten.time_evolution.time_evolution import TimeEvoMode

    tree, ham, _ = fixture()
    initial = vector(tree).copy()
    results = []
    for enabled in ("0", "1"):
        monkeypatch.setenv("PYTREENET_BUG_LAZY_NODE_COPIES", enabled)
        evolution = BUG(
            tree,
            ham,
            0.02,
            0.04,
            {},
            config=BUGConfig(
                max_bond_dim=8, rel_tol=0, total_tol=0, time_evo_mode=TimeEvoMode.RK45
            ),
        )
        for _ in range(2):
            evolution.run_one_time_step()
        results.append(vector(evolution.state))
    np.testing.assert_array_equal(vector(tree), initial)
    np.testing.assert_allclose(results[1], results[0], atol=1e-12)


def test_mpi_encoding_same_payload():
    import pickle

    calls = []

    def encode(*args):
        calls.append(args)
        return dict(
            codec="hdf5", payload=b"coefficients", metadata=b"original", version=1
        )

    shared = {}
    a, b, c, d, e, f = [object() for _ in range(6)]
    env = {}
    args = (a, b, env, c, 0.01, d, e)
    first = perf.encode_frontier_task(shared, encode, "a", *args)
    second = perf.encode_frontier_task(shared, encode, "b", *args)
    assert len(calls) == 1 and second["payload"] is first["payload"]
    assert pickle.loads(second["metadata"])[0] == "b"
    perf.encode_frontier_task(shared, encode, "c", a, f, env, c, 0.01, d, e)
    assert len(calls) == 2
