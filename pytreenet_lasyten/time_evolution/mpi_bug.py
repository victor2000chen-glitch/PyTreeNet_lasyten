"""MPI task-pool implementations of recursive BUG time evolution.

Create MPIBUG or MPIFixedBUG on the coordinator inside an MPICommExecutor
context. Other ranks serve subtree tasks; they must not instantiate the
integrator. The supplied executor is kept alive across all time steps.

The module itself does not import mpi4py, so MPI remains an optional dependency.
NumPy jobs use frozen pickle messages. Native LASyTen jobs use in-memory HDF5
messages, preserving sparse blocks, leg spaces and hidden charges. h5py >= 3.13
and the same hdf5_io/LASyTen build as checkpoints are required on every rank.
This implementation sends full snapshots per frontier task; it does not provide
streaming/distributed tensor storage.
"""
from __future__ import annotations

from concurrent.futures import Executor
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from numpy import ndarray

from .bug import BUG
from .fixed_bug import FixedBUG
from .time_evo_util.bug_util import basis_change_tensor_id
from .time_evo_util.common_bug import (
    CommonBUGConfig, _finish_non_leaf_update,
    _finish_root_update, _prepare_node_update, update_node,
)
from ._mpi_serialization import encode_task, decode_task, encode_result, decode_result
from ..contractions.sandwich_caching import SandwichCache
from ..contractions.tree_cach_dict import PartialTreeCachDict
from ..core.node import Node
from ..ttno import TreeTensorNetworkOperator
from ..ttns import TreeTensorNetworkState

__all__ = ["MPIBUG", "MPIFixedBUG", "root_update_with_executor"]


def _positive_task_count(value: int):
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or value < 1):
        raise ValueError("subtree_tasks must be a positive integer")


def _select_subtree_roots(state: TreeTensorNetworkState,
                          target: int) -> list[str]:
    """Select disjoint subtrees, splitting the largest splittable one first.

    Unary ancestors are kept on the coordinator only when this exposes an
    actual branch. The final task count may exceed the target for high-degree
    nodes, or fall short when the tree has too few leaves.
    """
    _positive_task_count(target)
    frontier = list(state.nodes[state.root_id].children)
    while len(frontier) < target:
        candidates = []
        for index, node_id in enumerate(frontier):
            branch_id = node_id
            children = state.nodes[branch_id].children
            while len(children) == 1:
                branch_id = children[0]
                children = state.nodes[branch_id].children
            if len(children) > 1:
                cost = sum(state.tensors.data[key].size
                           for key in state.find_subtree_of_node(node_id))
                candidates.append((cost, -index, branch_id))
        if not candidates:
            break
        _, minus_index, branch_id = max(candidates)
        index = -minus_index
        frontier[index:index + 1] = state.nodes[branch_id].children
    return frontier


@dataclass
class _SubtreeUpdate:
    """Only the updated subtree and its boundary, never outside neighbours."""
    child_id: str
    parent_id: str
    node_tensors: dict[str, tuple[Node, ndarray]]
    environment: ndarray
    basis_change: ndarray


def _update_subtree_task(node_id: str,
                          new_state: TreeTensorNetworkState,
                          parent_state: TreeTensorNetworkState,
                          parent_environments: dict,
                          hamiltonian: TreeTensorNetworkOperator,
                          time_step_size: float,
                          bug_config: CommonBUGConfig,
                          solver_options: dict | None) -> _SubtreeUpdate:
    """Picklable worker entry point; no MPI collectives or nested submissions."""
    cache = SandwichCache(parent_state, hamiltonian, parent_environments)
    state, environment, basis_change = update_node(
        node_id, new_state, parent_state, cache, hamiltonian,
        time_step_size, bug_config, solver_options=solver_options)
    boundary_id = basis_change_tensor_id(node_id)
    node_tensors = {key: state[key]
                    for key in state.find_subtree_of_node(boundary_id)}
    return _SubtreeUpdate(node_id, parent_state.nodes[node_id].parent,
                           node_tensors, environment, basis_change)



def _update_subtree_message(message):
    """Process/MPI entry point with a portable bytes-only tensor payload."""
    return encode_result(_update_subtree_task(*decode_task(message)))

def _merge_subtree_update(state: TreeTensorNetworkState,
                           result: _SubtreeUpdate):
    for node_id, (node, tensor) in result.node_tensors.items():
        state.nodes[node_id] = node
        state.tensors[node_id] = tensor
    state.nodes[result.parent_id].replace_child(
        result.child_id, basis_change_tensor_id(result.child_id))


def root_update_with_executor(current_state: TreeTensorNetworkState,
                               hamiltonian: TreeTensorNetworkOperator,
                               time_step_size: float,
                               bug_config: CommonBUGConfig,
                               executor: Executor,
                               subtree_tasks: int,
                               solver_options: dict | None = None
                               ) -> TreeTensorNetworkState:
    """Run one recursive BUG update with an MPI-compatible task executor.

    The coordinator prepares old-state gauges down to a disjoint frontier,
    submits all frontier subtrees, merges their results, and updates the
    remaining ancestors bottom-up. Remote tasks use the serial BUG kernel.
    Existing tensor arrays are frozen in submission snapshots because task
    serialization completes before coordinator-side preparation continues.

    The first implementation sends a full state/Hamiltonian snapshot per task
    and receives only subtree tensors and boundary data. It distributes
    computation, not tensor storage. Native tensors use their block-preserving
    HDF5 format; their Node representation spaces are never pickled. Configs
    and custom solver options must be picklable.
    """
    _positive_task_count(subtree_tasks)
    if executor is None or not callable(getattr(executor, "submit", None)):
        raise ValueError("An active task executor is required on the coordinator")
    if bug_config.subtree_workers != 1:
        raise ValueError("Use subtree_workers=1 with MPI; select MPI ranks instead")
    root_id = current_state.root_id
    assert root_id is not None, "The state has no root node!"
    assert current_state.orthogonality_center_id == root_id, \
        "The orthogonality center is not the root node!"
    frontier = set(_select_subtree_roots(current_state, subtree_tasks))
    new_state = deepcopy(current_state)
    root_cache = SandwichCache.init_cache_but_one(
        current_state, hamiltonian, root_id)
    ancestors = []
    futures = []
    shared_encoding = {}
    from ..util.bug_performance import encode_frontier_task

    def prepare(node_id, parent_state, parent_cache):
        if node_id in frontier:
            # Serialize synchronously so later gauge preparation cannot mutate
            # an object while a ProcessPool/MPI sender is still pickling it.
            message = encode_frontier_task(
                shared_encoding, encode_task, node_id, new_state, parent_state,
                parent_cache, hamiltonian, time_step_size, bug_config, solver_options)
            futures.append(executor.submit(_update_subtree_message, message))
            return
        state, cache = _prepare_node_update(
            node_id, parent_state, parent_cache, bug_config)
        ancestors.append((node_id, state, cache, parent_state))
        for child_id in tuple(state.nodes[node_id].children):
            prepare(child_id, state, cache)

    try:
        for child_id in tuple(current_state.nodes[root_id].children):
            prepare(child_id, current_state, root_cache)
        # No partial result is committed if submission or a remote solve fails.
        results = [decode_result(future.result()) for future in futures]
    except BaseException:
        for future in futures:
            future.cancel()
        raise

    boundaries = {}
    for result in results:
        _merge_subtree_update(new_state, result)
        boundaries[result.child_id] = (result.environment, result.basis_change)

    def child_caches(node_id):
        environments = PartialTreeCachDict()
        basis_changes = PartialTreeCachDict()
        for child_id in current_state.nodes[node_id].children:
            environment, basis_change = boundaries[child_id]
            environments.add_entry(child_id, node_id, environment)
            basis_changes.add_entry(child_id, node_id, basis_change)
        return environments, basis_changes

    for node_id, state, cache, parent_state in reversed(ancestors):
        environments, basis_changes = child_caches(node_id)
        _, environment, basis_change = _finish_non_leaf_update(
            node_id, state, new_state, parent_state, cache, hamiltonian,
            time_step_size, bug_config, environments, basis_changes,
            solver_options=solver_options)
        boundaries[node_id] = (environment, basis_change)

    environments, _ = child_caches(root_id)
    return _finish_root_update(
        current_state, new_state, root_cache, hamiltonian, time_step_size,
        bug_config, environments, solver_options=solver_options)


class _MPIBUGMixin:
    """Share MPI task scheduling between adaptive and fixed-rank BUG."""

    def __init__(self, *args: Any, executor: Executor,
                 subtree_tasks: int | None = None, **kwargs: Any):
        if executor is None or not callable(getattr(executor, "submit", None)):
            raise ValueError("Create the integrator only on the MPI coordinator")
        if subtree_tasks is None:
            subtree_tasks = getattr(executor, "num_workers", 1)
        _positive_task_count(subtree_tasks)
        self.subtree_executor = executor
        self.subtree_tasks = subtree_tasks
        super().__init__(*args, **kwargs)
        if self.config.subtree_workers != 1:
            raise ValueError("Use subtree_workers=1 with MPI; select MPI ranks instead")

    def recursive_update(self):
        self.state = root_update_with_executor(
            self.state, self.hamiltonian, self.time_step_size, self.config,
            self.subtree_executor, self.subtree_tasks,
            solver_options=self.solver_options)


class LegacyMPIBUG(_MPIBUGMixin, BUG):
    """Rank-adaptive BUG with MPI subtree tasks and coordinator-side truncation.

    Use the same arguments as BUG, plus ``executor`` from MPICommExecutor and
    optionally ``subtree_tasks`` (default: the executor's worker count).
    Keep the executor open for the complete evolution.
    """


class LegacyMPIFixedBUG(_MPIBUGMixin, FixedBUG):
    """Fixed-rank BUG using the same MPI task scheduling as MPIBUG."""

# Public worker classes use scoped sparse-block messages and persistent caches.
from ._mpi_bug_fast import MPIBUG, MPIFixedBUG
