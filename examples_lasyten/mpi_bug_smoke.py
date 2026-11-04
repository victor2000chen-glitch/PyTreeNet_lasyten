"""Small U(1)-conserving problem for serial/process/MPI installation checks.

    python examples/mpi_bug_smoke.py --executor process --verify
    mpiexec -n 3 python examples/mpi_bug_smoke.py --backend lasyten --verify

Native MPI requires LASyTen plus its hdf5_io dependency on every rank.
This tests a three-site spin/occupation Hamiltonian, not the HK impurity model.
"""
from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from fractions import Fraction
import multiprocessing
from time import perf_counter

import numpy as np

from pytreenet_lasyten.core.node import Node
from pytreenet_lasyten.operators import Hamiltonian, TensorProduct
from pytreenet_lasyten.ttns.ttns import TreeTensorNetworkState as TTNS
from pytreenet_lasyten.ttno.ttno_class import TreeTensorNetworkOperator as TTNO
from pytreenet_lasyten.ttno.state_diagram import TTNOFinder
from pytreenet_lasyten.time_evolution.bug import BUG, BUGConfig
from pytreenet_lasyten.time_evolution.fixed_bug import FixedBUG, FixedBUGConfig
from pytreenet_lasyten.time_evolution.mpi_bug import MPIBUG, MPIFixedBUG
from pytreenet_lasyten.time_evolution.time_evolution import TimeEvoMode


def make_problem(backend="numpy", seed=11):
    rng = np.random.default_rng(seed)
    def random(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)
    tree = TTNS()
    tree.add_root(Node(identifier="r"), random((2, 2, 1)))
    tree.add_child_to_parent(Node(identifier="a"), random((2, 2)), 0, "r", 0)
    tree.add_child_to_parent(Node(identifier="s"), random((2, 2, 2, 1)), 0, "r", 1)
    tree.add_child_to_parent(Node(identifier="b"), random((2, 2)), 0, "s", 1)
    tree.add_child_to_parent(Node(identifier="c"), random((2, 2)), 0, "s", 2)
    down = np.array([[0, 1], [0, 0]], complex)
    terms = [(.31, {"a": "up", "b": "down"}),
             (.31, {"a": "down", "b": "up"}),
             (-.17, {"b": "up", "c": "down"}),
             (-.17, {"b": "down", "c": "up"}),
             (.23, {"a": "n"}), (-.11, {"b": "n"}), (.07, {"c": "n"}),
             (.19, {"a": "n", "c": "n"})]
    conversion = dict(up=down.T, down=down, n=np.diag([0., 1.]),
                      I1=np.eye(1), I2=np.eye(2))
    h = Hamiltonian([(Fraction(1), f"t{i}", TensorProduct(term))
                     for i, (_, term) in enumerate(terms)],
                    conversion_dictionary=conversion,
                    coeffs_mapping={f"t{i}": value for i, (value, _) in enumerate(terms)})
    if backend == "lasyten":
        from pytreenet_lasyten.util.u1_t3ns import random_u1_state_like
        from pytreenet_lasyten.util.u1_ttno import u1_ttno_from_hamiltonian
        state = random_u1_state_like(tree, total_particles=1, max_bond_dim=4, seed=seed)
        operator = u1_ttno_from_hamiltonian(h, state)
    elif backend == "numpy":
        tree.canonical_form("r")
        tree.normalise()
        state = tree
        operator = TTNO.from_hamiltonian(h, state, method=TTNOFinder.SGE)
    else:
        raise ValueError("backend must be numpy or lasyten")
    dense = np.zeros((8, 8), complex)
    for coefficient, term in terms:
        matrix = np.array([[1.]])
        for key in ("a", "b", "c"):
            matrix = np.kron(matrix, conversion[term.get(key, "I2")])
        dense += coefficient * matrix
    return state, operator, dense


def vector(tree):
    """Independent NumPy contraction for this tiny topology only."""
    return np.einsum("asx,ay,sbcz,bv,cw->xyzvw", tree.tensors["r"], tree.tensors["a"],
                     tree.tensors["s"], tree.tensors["b"], tree.tensors["c"]).reshape(-1)


def agreement(actual, reference):
    n1 = actual.scalar_product(use_orthogonal_center=False).real
    n2 = reference.scalar_product(use_orthogonal_center=False).real
    overlap = actual.scalar_product(reference, use_orthogonal_center=False)
    fidelity = float(abs(overlap) ** 2 / (n1 * n2))
    if not np.isfinite(fidelity) or not -1e-10 <= fidelity <= 1 + 1e-10:
        raise AssertionError(f"Invalid fidelity {fidelity}")
    np.testing.assert_allclose(n1, n2, rtol=1e-9, atol=1e-11)
    # Fidelity discards global phase, which matters for later Green functions.
    np.testing.assert_allclose(overlap / np.sqrt(n1*n2), 1, rtol=1e-9, atol=1e-10)
    if 1 - fidelity > 1e-9:
        raise AssertionError(f"Serial/MPI fidelity mismatch: {fidelity}")
    return float(np.clip(fidelity, 0, 1))


def run_checks(executor, args):
    state, operator, _ = make_problem(args.backend)
    cls = MPIFixedBUG if args.fixed_rank else MPIBUG
    serial_cls = FixedBUG if args.fixed_rank else BUG
    config = (FixedBUGConfig(time_evo_mode=TimeEvoMode.RK45) if args.fixed_rank else
              BUGConfig(max_bond_dim=4, rel_tol=0, total_tol=0, time_evo_mode=TimeEvoMode.RK45))
    options = dict(rtol=1e-10, atol=1e-12)
    arguments = (state, operator, .01, .01 * args.steps, {})
    if executor is None:
        evolution = serial_cls(*arguments, config=config, solver_options=options)
    else:
        evolution = cls(*arguments, config=config, solver_options=options,
                        executor=executor, subtree_tasks=args.subtree_tasks)
    reference = (serial_cls(*arguments, config=deepcopy(config), solver_options=options)
                 if args.verify else None)
    for step in range(args.steps):
        start = perf_counter()
        evolution.run_one_time_step()
        elapsed = perf_counter() - start
        text = f"step={step + 1}, backend={args.backend}, executor={args.executor}, seconds={elapsed:.6f}"
        if reference is not None:
            reference.run_one_time_step()
            text += f", fidelity={agreement(evolution.state, reference.state):.12g}"
        print(text, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "lasyten"), default="numpy")
    parser.add_argument("--executor", choices=("mpi", "process", "serial"), default="mpi")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--subtree-tasks", type=int, default=2)
    parser.add_argument("--fixed-rank", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.workers, args.subtree_tasks) < 1:
        parser.error("steps, workers and subtree-tasks must be positive")
    if args.executor == "mpi":
        from mpi4py import MPI
        from mpi4py.futures import MPICommExecutor
        if MPI.COMM_WORLD.size < 2:
            parser.error("Use mpiexec -n 2 (or more) for the MPI executor.")
        with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
            if executor is not None:
                run_checks(executor, args)
    else:
        context = (ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"))
                   if args.executor == "process" else nullcontext(None))
        with context as executor:
            run_checks(executor, args)


if __name__ == "__main__":
    main()
