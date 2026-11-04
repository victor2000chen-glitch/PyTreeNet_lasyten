"""Check optimized BUG against the previous contractions and dense evolution.

From the repository root:
    python -m examples.bug_performance_check --backend numpy
    python -m examples.bug_performance_check --backend lasyten

This is a tiny correctness check, not an HK performance benchmark. Only this
eight-dimensional test problem is exported to dense arrays for comparison.
"""

from __future__ import annotations

from argparse import ArgumentParser
from contextlib import contextmanager
from copy import deepcopy
import os

import numpy as np
from scipy.linalg import expm

import pytreenet_lasyten
from pytreenet_lasyten.time_evolution.bug import BUG, BUGConfig
from pytreenet_lasyten.time_evolution.time_evolution import TimeEvoMode
from pytreenet_lasyten.util import bug_native_speedups as previous
from pytreenet_lasyten.util import bug_performance as perf

from .mpi_bug_smoke import make_problem


def _dense_vector(tree):
    arrays = [tree.tensors[key] for key in ("r", "a", "s", "b", "c")]
    arrays = [a.to_numpy() if hasattr(a, "native") else a for a in arrays]
    return np.einsum("asx,ay,sbcz,bv,cw->xyzvw", *arrays).reshape(-1)


@contextmanager
def _settings(enabled):
    changes = {
        "PYTREENET_BUG_" + name: "1" if enabled else "0"
        for name in (
            "PLANNED_ACTIONS",
            "PLANNED_ENVIRONMENTS",
            "CACHE_PATHS",
            "LAZY_NODE_COPIES",
        )
    }
    changes["PYTREENET_BUG_PROFILE"] = "1"
    saved_environment = {name: os.environ.get(name) for name in changes}
    saved_settings = previous.settings()
    try:
        os.environ.update(changes)
        # Exercise the large-operator action even though this problem is small.
        previous.configure(prepare_actions=True, max_operator_elements=0)
        yield
    finally:
        previous.configure(**saved_settings)
        for name, value in saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def check_case(backend, mode, dt):
    """Compare two steps from identical inputs, retaining norm and global phase."""
    state, hamiltonian, exact = make_problem(backend)
    initial = _dense_vector(state).copy()
    results = []
    for enabled in (False, True):
        with _settings(enabled):
            perf.reset_stats()
            config = BUGConfig(
                max_bond_dim=8,
                rel_tol=0,
                total_tol=0,
                time_evo_mode=mode,
            )
            operator = deepcopy(hamiltonian)
            # Encode the complex contour in H, as in the impurity workflow.
            operator.tensors[operator.root_id] *= dt / 0.02
            evolution = BUG(
                state,
                operator,
                0.02,
                0.04,
                {},
                config=config,
                solver_options={"rtol": 1e-10, "atol": 1e-12},
            )
            for _ in range(2):
                evolution.run_one_time_step()
            results.append(_dense_vector(evolution.state))
            np.testing.assert_allclose(_dense_vector(state), initial, atol=1e-13)

    np.testing.assert_allclose(results[1], results[0], atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(
        results[1],
        expm(-2j * dt * exact) @ initial,
        atol=2e-8,
        rtol=2e-8,
    )
    diagnostics = perf.stats()
    if backend == "lasyten":
        assert diagnostics["counts"].get("planned_matvecs", 0) > 0
        assert diagnostics["counts"].get("planned_environments", 0) > 0
    error = float(np.linalg.norm(results[1] - results[0]) / np.linalg.norm(results[0]))
    return error, diagnostics


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "lasyten"), default="numpy")
    args = parser.parse_args()
    print("PyTreeNet package:", pytreenet_lasyten.__file__, flush=True)
    if args.backend == "lasyten":
        import lasyten

        print("Compiled LASyTen:", lasyten.__file__, flush=True)
    for mode in (TimeEvoMode.RK23, TimeEvoMode.RK45):
        for dt in (0.02, 0.02 - 0.004j):
            error, diagnostics = check_case(args.backend, mode, dt)
            print(
                f"PASS {args.backend} {mode.value} dt={dt} "
                f"relative state error={error:.3e}",
                flush=True,
            )
    print("Diagnostics (final case):", diagnostics)


if __name__ == "__main__":
    main()
