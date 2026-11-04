"""Small exact-reference tests; compiled LASyTen execution is separate."""
from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import Complex_bug_impurity_solver_lasyten as model


class State:
    calls = 0

    def __init__(self, data):
        self.data = np.asarray(data, dtype=complex)

    def scalar_product(self, other=None, use_orthogonal_center=False):
        State.calls += 1
        return np.vdot(self.data if other is None else other.data, self.data)


def small_siam():
    operators = []
    for mode in range(4):
        a = np.zeros((16, 16), complex)
        for bits in range(16):
            if bits & (1 << mode):
                a[bits ^ (1 << mode), bits] = (-1) ** ((bits & ((1 << mode) - 1)).bit_count())
        operators.append(a)
    n = [a.conj().T @ a for a in operators]
    H = -n[0] - n[1] + 2 * n[0] @ n[1] + .15 * (n[2] + n[3])
    for i, b in [(0, 2), (1, 3)]:
        hopping = (.43 + .17j) * operators[i].conj().T @ operators[b]
        H += hopping + hopping.conj().T
    indices = [k for k in range(16)
               if sum((k >> m) & 1 for m in (0, 2)) == 1
               and sum((k >> m) & 1 for m in (1, 3)) == 1]
    energy, vectors = np.linalg.eigh(H[np.ix_(indices, indices)])
    ground = np.zeros(16, complex)
    ground[indices] = vectors[:, 0]
    return H, energy[0], operators[0] @ ground, operators[0].conj().T @ ground


class HelperTests(unittest.TestCase):
    def test_projection_orientation_residuals_and_cached_gram(self):
        rng = np.random.default_rng(37)
        V = rng.normal(size=(8, 5)) + 1j * rng.normal(size=(8, 5))
        A = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        H = (A + A.conj().T) / 2
        # A small nonsymmetric perturbation models inexact H|v> compression.
        action = H + 1e-4 * A
        states = [State(v) for v in V.T]
        State.calls = 0
        *_, X, S = model.pivoted_cholesky(states, tol=1e-13, return_gram=True)
        self.assertEqual(State.calls, 15)
        np.testing.assert_allclose(S, V.conj().T @ V, atol=1e-12)
        np.testing.assert_allclose(X.conj().T @ S @ X, np.eye(5), atol=1e-12)
        for full in (False, True):
            calls = []
            def apply(v, h, params):
                calls.append(v)
                return State(action @ v.data)
            with patch.object(model, 'apply_and_compress_native', side_effect=apply):
                E, Q, B, residuals, relative, diagnostics = model.krylov_projection_TTN(
                    H, X, states, Keep_all=True, gram_matrix=S,
                    full_residuals=full, return_diagnostics=True,
                )
            self.assertEqual(len(calls), 5)
            np.testing.assert_allclose(diagnostics['projected_hamiltonian'],
                                       V.conj().T @ action @ V, atol=1e-12)
            np.testing.assert_allclose(B, X @ Q, atol=1e-12)
            if full:
                expected = np.linalg.norm(action @ V @ B - (V @ B) * E, axis=0)
                np.testing.assert_allclose(residuals, expected, atol=1e-11)
                np.testing.assert_allclose(relative, expected / np.maximum(1., abs(E)))
            else:
                self.assertTrue(np.isnan(residuals).all())
                self.assertIsNone(diagnostics['hamiltonian_norm_matrix'])

    def test_dependent_snapshots_and_empty_subspace_errors(self):
        states = [State([1, 0]), State([0, 1]), State([1, 1])]
        L, permutation, diagonal, error, X, S = model.pivoted_cholesky(
            states, return_gram=True,
        )
        self.assertEqual(X.shape, (3, 2))
        np.testing.assert_allclose(X.conj().T @ S @ X, np.eye(2), atol=1e-12)
        with self.assertRaises(ValueError):
            model.pivoted_cholesky([])
        with self.assertRaises(ValueError):
            model.pivoted_cholesky(states, tol=100)

    def test_rk45_and_legacy_positional_argument(self):
        seen = []
        class ExactBUG:
            def __init__(self, **kwargs):
                seen.append(kwargs)
                self.state = deepcopy(kwargs['initial_state'])
            def run_one_time_step(self):
                self.state.data *= np.exp(-.1j)
        initial = State([1, 2j])
        with patch.object(model, 'BUG', ExactBUG):
            snapshots = model.Complex_time_evolution(
                initial, object(), .1, .2, 15, {}, 3, model.TimeEvoMode.RK45,
            )
            defaults = model.Complex_time_evolution(initial, object(), .1, .2, 15, {}, 1)
        self.assertEqual(len(seen), 2)
        self.assertEqual(len(snapshots), 4)
        self.assertEqual(len(defaults), 2)
        self.assertEqual(seen[0]['config'].time_evo_mode, model.TimeEvoMode.RK45)
        self.assertEqual(seen[1]['config'].time_evo_mode, model.TimeEvoMode.RK45)
        self.assertEqual(seen[0]['solver_options'], {'rtol': 1e-8, 'atol': 1e-10})
        np.testing.assert_allclose(snapshots[-1].data, initial.data * np.exp(-.6j))
        np.testing.assert_array_equal(initial.data, [1, 2j])
        with self.assertRaises(ValueError):
            model.Complex_time_evolution(initial, object(), .1, .25, 15, {}, 1)

    def test_actual_numpy_bug_rk45_matches_matrix_exponential(self):
        from scipy.linalg import expm
        from pytreenet_lasyten.core.node import Node
        from pytreenet_lasyten.ttns.ttns import TreeTensorNetworkState as TTNS
        from pytreenet_lasyten.ttno.ttno_class import TreeTensorNetworkOperator as TTNO
        initial = np.array([.8, .6j], complex)
        matrix = np.array([[.2, .3 + .1j], [.3 - .1j, -.4]], complex)
        state, operator = TTNS(), TTNO()
        state.add_root(Node(identifier='r'), initial.copy())
        operator.add_root(Node(identifier='r'), matrix)
        snapshots = model.Complex_time_evolution(
            state, operator, .01, .02, 4, {}, 2,
            time_evo_mode=model.TimeEvoMode.RK45,
        )
        self.assertEqual(len(snapshots), 3)
        np.testing.assert_allclose(snapshots[-1].tensors['r'],
                                   expm(-1j * matrix * .04) @ initial,
                                   rtol=1e-7, atol=1e-9)
        np.testing.assert_array_equal(state.tensors['r'], initial)

    def test_lehmann_against_exact_complex_siam_resolvents(self):
        H, E0, removal, addition = small_siam()
        omega, eta = np.linspace(-4, 4, 101), .2
        E, U = np.linalg.eigh(H)
        total = np.zeros_like(omega, dtype=complex)
        apply = lambda v, h, params: State(h @ v.data)
        for sign, raw in [(-1, removal), (1, addition)]:
            Z = float(np.vdot(raw, raw).real)
            seed = raw / np.sqrt(Z)
            # Complex-time snapshots span the accessible two-dimensional sector.
            snapshots = [State(U @ (np.exp(-1j * E * (.1 - .03j) * k) * (U.conj().T @ seed)))
                         for k in range(6)]
            *_, X, S = model.pivoted_cholesky(snapshots, tol=1e-12, svd_tol=1e-9, return_gram=True)
            with patch.object(model, 'apply_and_compress_native', side_effect=apply):
                poles, Q, B, residuals, _ = model.krylov_projection_TTN(
                    H, X, snapshots, Keep_all=True, gram_matrix=S,
                )
            G, captured = model.lehmann_sector_TTN(
                poles, B, snapshots, snapshots[0], E0, eta, omega, sign,
                overlaps=S[0], excitation_norm=S[0, 0],
            )
            uncached, _ = model.lehmann_sector_TTN(
                poles, B, snapshots, snapshots[0], E0, eta, omega, sign,
            )
            np.testing.assert_allclose(G, uncached, atol=1e-12)
            exact = np.array([np.vdot(raw, np.linalg.solve(
                (w + 1j * eta) * np.eye(16) - sign * (H - E0 * np.eye(16)), raw,
            )) for w in omega])
            np.testing.assert_allclose(Z * G, exact, rtol=1e-10, atol=1e-10)
            self.assertAlmostEqual(captured, 1., places=10)
            total += Z * G
        np.testing.assert_allclose(model.spectral_function(total), -total.imag / np.pi)

    def test_broadcast_serializes_records_and_preserves_root(self):
        import native_mpi_transport as transport
        message = []
        class Comm:
            def __init__(self, rank):self.rank = rank
            def Get_rank(self):return self.rank
            def bcast(self, value, root):
                if self.rank == root:message.append(deepcopy(value));return value
                return deepcopy(message[-1])
        state = State([1, 2j])
        record = {'blocks': [state.data.copy()]}
        with patch.object(transport, '_pack_network', return_value=record) as pack, \
             patch.object(transport, '_unpack_network', side_effect=lambda cls, r: cls(r['blocks'][0])):
            root = model.broadcast_ttn(state, Comm(0))
            worker = model.broadcast_ttn(None, Comm(1), network_class=State)
        self.assertIs(root, state)
        pack.assert_called_once_with(state)
        np.testing.assert_array_equal(worker.data, state.data)

    def test_driver_helper_call_signatures(self):
        import ast
        source = (Path(__file__).resolve().parents[1] / 'examples/hk/lasyten_HK_mpi.py').read_text()
        tree = ast.parse(source)
        checked = set()
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                name = call.func.id
                if name in {'Complex_time_evolution', 'pivoted_cholesky',
                            'krylov_projection_TTN', 'lehmann_sector_TTN',
                            'broadcast_ttn', 'construct_HK_with_bath',
                            'construct_fermionic_order'}:
                    inspect.signature(getattr(model, name)).bind(
                        *[object() for arg in call.args],
                        **{kw.arg: object() for kw in call.keywords},
                    )
                    checked.add(name)
        self.assertEqual(len(checked), 7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
