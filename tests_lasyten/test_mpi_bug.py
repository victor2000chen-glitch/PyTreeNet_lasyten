from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
import multiprocessing

import numpy as np
import pytest

from examples.mpi_bug_smoke import make_problem, vector
from pytreenet_lasyten.time_evolution.bug import BUG, BUGConfig
from pytreenet_lasyten.time_evolution.fixed_bug import FixedBUG, FixedBUGConfig
from pytreenet_lasyten.time_evolution.mpi_bug import MPIBUG, MPIFixedBUG
from pytreenet_lasyten.time_evolution.time_evolution import TimeEvoMode
from pytreenet_lasyten.time_evolution import _mpi_serialization as wire
from pytreenet_lasyten.contractions.sandwich_caching import SandwichCache


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("tasks", [1, 2, 3, 8])
def test_spawned_workers_match_serial_over_two_steps(fixed, tasks):
    state, operator, _ = make_problem()
    before = vector(state).copy()
    config = (FixedBUGConfig(time_evo_mode=TimeEvoMode.RK45) if fixed else
              BUGConfig(max_bond_dim=4, rel_tol=0, total_tol=0, time_evo_mode=TimeEvoMode.RK45))
    options = dict(rtol=1e-11, atol=1e-13)
    args = (state, operator, .01, .02, {})
    reference = (FixedBUG if fixed else BUG)(*args, config=deepcopy(config), solver_options=options)
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as executor:
        parallel = (MPIFixedBUG if fixed else MPIBUG)(*args, config=deepcopy(config),
                      solver_options=options, executor=executor, subtree_tasks=tasks)
        for _ in range(2):
            parallel.run_one_time_step()
            reference.run_one_time_step()
            np.testing.assert_allclose(vector(parallel.state), vector(reference.state),
                                       atol=1e-11, rtol=1e-10)
            assert parallel.state.orthogonality_center_id == state.root_id
            assert not any(key.endswith("_basis_change_tensor") for key in parallel.state.nodes)
    np.testing.assert_array_equal(vector(state), before)


def test_frozen_task_is_independent_of_later_sender_mutation():
    state, operator, _ = make_problem()
    cache = SandwichCache.init_cache_but_one(state, operator, state.root_id)
    before = vector(state).copy()
    message = wire.encode_task("a", state, state, dict(cache), operator, .01,
                               BUGConfig(), dict(rtol=1e-10))
    assert message["codec"] == "pickle"
    state.tensors["a"] *= 2
    _, restored, parent, _, _, _, _, _ = wire.decode_task(message)
    np.testing.assert_array_equal(vector(restored), before)
    np.testing.assert_array_equal(vector(parent), before)


def test_worker_failure_does_not_replace_evolution_state():
    from concurrent.futures import Future
    class FailingExecutor:
        def submit(self, *args, **kwargs):
            future = Future()
            future.set_exception(RuntimeError("worker solve failed"))
            return future
    state, operator, _ = make_problem()
    evolution = MPIBUG(state, operator, .01, .01, {}, config=BUGConfig(),
                       executor=FailingExecutor(), subtree_tasks=3)
    old = evolution.state
    initial = vector(old).copy()
    with pytest.raises(RuntimeError, match="worker solve failed"):
        evolution.run_one_time_step()
    assert evolution.state is old
    np.testing.assert_array_equal(vector(evolution.state), initial)


def test_serial_bug_rejects_unsupported_thread_count():
    state, operator, _ = make_problem()
    evolution = BUG(state, operator, .01, .01, {}, config=BUGConfig(subtree_workers=2))
    initial = vector(evolution.state).copy()
    with pytest.raises(ValueError, match="subtree_workers=1"):
        evolution.run_one_time_step()
    np.testing.assert_array_equal(vector(evolution.state), initial)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_task_counts_are_rejected(value):
    state, operator, _ = make_problem()
    with ThreadPoolExecutor(1) as executor:
        with pytest.raises(ValueError):
            MPIBUG(state, operator, .01, .01, {}, executor=executor, subtree_tasks=value)
