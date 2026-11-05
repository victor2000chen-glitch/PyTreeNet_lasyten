import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg import expm

from examples.mpi_bug_smoke import make_problem, vector
from pytreenet_lasyten.util import tensor_splitting as ts
from pytreenet_lasyten.time_evolution.bug import BUG, BUGConfig
from pytreenet_lasyten.time_evolution.time_evolution import TimeEvoMode

PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "pytreenet_lasyten"


def test_imports_without_old_pytreenet_even_when_pyscf_is_present(tmp_path):
    script = '''
import importlib.abc,sys,types
class BlockOld(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name == "pytreenet" or name.startswith("pytreenet."):
   raise ModuleNotFoundError("Old package import: "+name)
sys.meta_path.insert(0,BlockOld())
fake=types.ModuleType("pyscf.lib")
fake.davidson=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("Wrong solver"))
sys.modules["pyscf"]=types.ModuleType("pyscf")
sys.modules["pyscf.lib"]=fake
from pytreenet_lasyten.time_evolution import BUG,MPIBUG
from pytreenet_lasyten.time_evolution.time_evo_enum import TimeEvoAlg
from pytreenet_lasyten.dmrg.dmrg_sparse import davidson
from pytreenet_lasyten.dmrg.dmrg_utils import davidson as adapted
assert davidson is adapted
assert TimeEvoAlg.BUG.get_class() is BUG
assert not any(n == "pytreenet" or n.startswith("pytreenet.") for n in sys.modules)
'''
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PROJECT), os.environ.get("PYTHONPATH", "")]))
    result = subprocess.run([sys.executable, "-c", script], env=env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_active_imports_of_old_package():
    for path in PACKAGE.rglob("*.py"):
        if "_lasyten_bug_speedups_backup" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            assert not any(m and (m == "pytreenet" or m.startswith("pytreenet.")) for m in modules), path


@pytest.mark.parametrize("shape,rows,cols", [((3, 5), (0,), (1,)), ((5, 3), (0,), (1,)),
                                           ((2, 3, 4), (2, 0), (1,))])
@pytest.mark.parametrize("mode", list(ts.SplitMode))
def test_qr_modes_reconstruct_and_have_expected_dimensions(shape, rows, cols, mode):
    tensor = np.random.default_rng(8).normal(size=shape) + 1j*np.random.default_rng(9).normal(size=shape)
    q, r = ts.tensor_qr_decomposition(tensor, rows, cols, mode)
    expected = tensor.transpose(rows + cols)
    actual = np.tensordot(q, r, axes=(-1, 0))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    m, n = np.prod([shape[i] for i in rows]), np.prod([shape[i] for i in cols])
    target = m if mode is ts.SplitMode.FULL else n if mode is ts.SplitMode.KEEP else min(m, n)
    assert q.shape[-1] == r.shape[0] == target


@pytest.mark.parametrize("mode", list(ts.SplitMode))
def test_svd_modes_reconstruct_rectangular_complex_tensor(mode):
    tensor = np.random.default_rng(3).normal(size=(2, 5, 3)).astype(complex)*(1+.3j)
    u, s, vh = ts.tensor_svd(tensor, (2, 0), (1,), mode)
    actual = np.tensordot(u[..., :len(s)]*s, vh[:len(s)], axes=(-1, 0))
    np.testing.assert_allclose(actual, tensor.transpose(2, 0, 1), atol=1e-12, rtol=1e-12)
    assert type(mode.numpy_svd_mode()) is bool


@pytest.mark.parametrize("mode", list(ts.SplitMode))
def test_native_decompositions_forward_modes_without_numpy_coercion(monkeypatch, mode):
    class Protected:
        ndim = 2
        def __array__(self, *a, **k):
            raise AssertionError("Native data were converted to NumPy")
    calls = []
    monkeypatch.setattr(ts, "is_lasyten_tensor", lambda a: isinstance(a, Protected))
    def qr(a, left, right, **options):
        calls.append(("qr", options)); return "q", "r"
    def svd(a, left, right, **options):
        calls.append(("svd", options)); return "u", "s", "v"
    monkeypatch.setattr(ts.lb, "tensor_qr_decomposition", qr)
    monkeypatch.setattr(ts.lb, "tensor_svd", svd)
    assert ts.tensor_qr_decomposition(Protected(), (0,), (1,), mode) == ("q", "r")
    assert ts.tensor_svd(Protected(), (0,), (1,), mode) == ("u", "s", "v")
    assert calls == [("qr", dict(mode=mode.value)), ("svd", dict(mode=mode.value, decomp="svd"))]


def test_two_full_bug_steps_match_independent_exponential():
    state, operator, matrix = make_problem()
    initial = vector(state).copy()
    config = BUGConfig(max_bond_dim=4, rel_tol=0, total_tol=0, time_evo_mode=TimeEvoMode.RK45)
    evolution = BUG(state, operator, .01, .02, {}, config=config,
                    solver_options=dict(rtol=1e-11, atol=1e-13))
    for step in (1, 2):
        evolution.run_one_time_step()
        np.testing.assert_allclose(vector(evolution.state), expm(-1j*.01*step*matrix) @ initial,
                                   atol=1e-11, rtol=1e-10)
    np.testing.assert_array_equal(vector(state), initial)


def test_one_site_dmrg_matches_exact_ground_energy():
    from pytreenet_lasyten.dmrg.dmrg_sparse import DMRGAlgorithm_sparse
    state, operator, matrix = make_problem()
    algorithm = DMRGAlgorithm_sparse(state, operator, num_sweeps=2, max_iter=80,
                                     svd_params=ts.SVDParameters(max_bond_dim=4), site="one-site")
    energies = algorithm.run()
    np.testing.assert_allclose(energies[-1], np.linalg.eigvalsh(matrix)[0], atol=2e-10, rtol=1e-10)


def test_native_two_site_dmrg_rejects_before_changing_state(monkeypatch):
    module = importlib.import_module("pytreenet_lasyten.dmrg.dmrg_sparse")
    state, operator, _ = make_problem()
    initial = vector(state).copy()
    center = state.orthogonality_center_id
    monkeypatch.setattr(module, "is_lasyten_tensor", lambda a: True)
    with pytest.raises(NotImplementedError, match="one-site"):
        module.DMRGAlgorithm_sparse(state, operator, 1, 20, ts.SVDParameters(), "two-site")
    assert state.orthogonality_center_id == center
    np.testing.assert_array_equal(vector(state), initial)
