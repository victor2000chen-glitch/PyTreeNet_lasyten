"""Exact-contraction BUG optimizations; no dense export of native tensors.

The planner estimates dense-equivalent work, not block FLOPs. Its memory
budget is a path-selection target, NOT an RSS cap. If no chain fits, the
minimum-peak chain is used and counted. No slicing or new truncation occurs.
"""
from __future__ import annotations
from collections import Counter
from collections.abc import MutableMapping
from copy import copy, deepcopy
from functools import lru_cache, wraps
from itertools import permutations
from math import prod
import os
import pickle
from time import perf_counter

_COUNTS = Counter()
_TIMES = Counter()
_LOCAL = {}
_PATHS = {}


def _enabled(name, default=True):
    return os.environ.get('PYTREENET_BUG_' + name, '1' if default else '0').lower() not in ('0', 'false', 'off')


def reset_stats():
    _COUNTS.clear()
    _TIMES.clear()
    _LOCAL.clear()


def stats():
    return dict(counts=dict(_COUNTS), seconds=dict(_TIMES),
                local_solves={k: dict(v) for k, v in _LOCAL.items()},
                plan_cache=plan_chain.cache_info()._asdict())


def _budget(itemsize):
    mib = float(os.environ.get('PYTREENET_BUG_PATH_MIB', '256'))
    if not 0 < mib < float('inf'):
        raise ValueError('PYTREENET_BUG_PATH_MIB must be finite and positive.')
    return max(1, int(mib * 2**20 / itemsize))


@lru_cache(maxsize=2048)
def plan_chain(labels, shapes, output, budget):
    """Enumerate connected chains anchored on the ket, with no outer products.

    Only layouts/shapes are cached. No tensor values, spaces or environments
    are cached across solves. This is deliberately a bounded search for T3NS;
    high-degree nodes use a greedy connected chain instead of factorial work.
    """
    dimensions = {}
    counts = Counter()
    for legs, shape in zip(labels, shapes):
        if len(legs) != len(shape) or len(set(legs)) != len(legs):
            raise ValueError('Malformed contraction labels.')
        for leg, dim in zip(legs, shape):
            if dimensions.setdefault(leg, dim) != dim:
                raise ValueError('Contraction dimensions disagree.')
            counts[leg] += 1
    if any(n != (1 if leg in output else 2) for leg, n in counts.items()):
        raise ValueError('Expected an ordinary pairwise tensor network.')
    if len(set(output)) != len(output) or set(output) != {x for x, n in counts.items() if n == 1}:
        raise ValueError('Invalid output legs.')

    reached = {0}
    while True:
        more = {j for j in range(len(labels)) if any(set(labels[j]) & set(labels[i]) for i in reached)}
        if more <= reached:
            break
        reached |= more
    disconnected = len(reached) != len(labels)

    def advance(current, index):
        right = labels[index]
        shared = tuple(x for x in current if x in right)
        if not shared and not disconnected:
            return None
        left_axes = tuple(current.index(x) for x in shared)
        right_axes = tuple(right.index(x) for x in shared)
        result = tuple(x for x in current if x not in shared) + tuple(x for x in right if x not in shared)
        size = prod(dimensions[x] for x in result)
        work = size * prod(dimensions[x] for x in shared)
        return result, (index, left_axes, right_axes), size, work

    if len(labels) <= 7:
        orders = permutations(range(1, len(labels)))
    else:
        current, remaining, order = labels[0], set(range(1, len(labels))), []
        while remaining:
            possible = [(advance(current, i), i) for i in remaining]
            possible = [(x, i) for x, i in possible if x is not None]
            if not possible:
                raise ValueError('Disconnected contraction network.')
            step, i = min(possible, key=lambda pair: (pair[0][2] > budget, pair[0][3], pair[0][2], pair[1]))
            current = step[0]
            remaining.remove(i)
            order.append(i)
        orders = [order]
    best = None
    for order in orders:
        current, instructions, peak, work = labels[0], [], 0, 0
        for index in order:
            value = advance(current, index)
            if value is None:
                break
            current, instruction, size, cost = value
            instructions.append(instruction)
            peak, work = max(peak, size), work + cost
        else:
            if set(current) != set(output):
                raise ValueError('Contraction did not produce requested output.')
            score = (0, work, peak) if peak <= budget else (1, peak, work)
            if best is None or score < best[0]:
                best = score, tuple(instructions), tuple(current.index(x) for x in output), peak, work
    if best is None:
        raise ValueError('No connected ket-anchored contraction chain exists.')
    return best[1:]


def _make_plan(operands, labels, output):
    import numpy as np
    shapes = tuple(tuple(int(d) for d in a.shape) for a in operands)
    itemsize = max(np.dtype(a.dtype).itemsize for a in operands)
    budget = _budget(itemsize)
    plan = plan_chain(tuple(map(tuple, labels)), shapes, tuple(output), budget)
    _COUNTS['plans_used'] += 1
    _COUNTS['largest_planned_result_bytes'] = max(_COUNTS['largest_planned_result_bytes'], plan[2] * itemsize)
    if plan[2] > budget:
        _COUNTS['plans_over_target'] += 1
    return plan


def _layout(state_node, ham_node, id_trafo_op=None):
    neighbours = tuple(state_node.neighbouring_nodes())
    v, p = len(neighbours), state_node.nopen_legs()
    transform = id_trafo_op or (lambda x: x)
    mapped = tuple(transform(n) for n in neighbours)
    axes = tuple(ham_node.neighbour_index(n) for n in mapped)
    if len(set(mapped)) != v or sorted(axes) != list(range(v)) or ham_node.nneighbours() != v:
        raise ValueError('Ket/operator neighbours must match bijectively.')
    if ham_node.nopen_legs() != 2*p:
        raise ValueError('Operator physical legs must be output followed by input.')
    ket, bra, op = tuple(range(v)), tuple(range(v, 2*v)), tuple(range(2*v, 3*v))
    pin, pout = tuple(range(3*v, 3*v+p)), tuple(range(3*v+p, 3*v+2*p))
    ordered = [None] * v
    for axis, label in zip(axes, op):
        ordered[axis] = label
    ham_labels = tuple(ordered) + pout + pin
    return neighbours, ket, bra, op, pin, pout, ham_labels


class PlannedAction:
    """Prepared native action. Every chain includes the trial tensor from step 1."""
    local_operator = None

    def __init__(self, old, state, node, ham, hnode, cache, wrap_result, counters):
        self.old = old
        self.wrap_result, self.counters = wrap_result, counters
        neighbours, ket, bra, op, pin, pout, hlegs = _layout(node, hnode)
        self.operands = (None, old.operator, *old.environments)
        labels = (ket + pin, hlegs, *((ket[i], op[i], bra[i]) for i in range(len(neighbours))))
        envs = [cache.get_entry(n, node.identifier) for n in neighbours]
        self.plan = _make_plan((state, ham, *envs), labels, bra + pout)
        _COUNTS['planned_actions'] += 1

    def __call__(self, time, tensor):
        ct = self.old.ct
        result = tensor.native
        for index, left, right in self.plan[0]:
            result = ct.tdot(result, self.operands[index], list(left), list(right))
            _COUNTS['action_tdots'] += 1
        permutation = self.plan[1]
        if permutation != tuple(range(len(permutation))):
            result = ct.permute_legs(result, codomain=list(permutation), domain=[])
        result.set_labels(self.old.labels)
        _COUNTS['planned_matvecs'] += 1
        # Keep the pre-existing public speedup counters useful.
        self.counters['prepared_matvecs'] += 1
        self.counters['prepared_tdot_calls'] += len(self.plan[0])
        return self.wrap_result(tensor, result)


def improve_action(old, state, node, ham, hnode, cache, wrap_result, counters):
    if old is None or old.local_operator is not None or not _enabled('PLANNED_ACTIONS'):
        return old
    return PlannedAction(old, state, node, ham, hnode, cache, wrap_result, counters)


def contract_environment(ignored_node_id, ket_node, ket_tensor, op_node, op_tensor,
                         dictionary, bra_node=None, bra_tensor=None,
                         id_trafo_op=None, id_trafo_bra=None):
    """Plan the complete native sandwich; explicit bra is already conjugated."""
    from . import lasyten_backend as lb
    from ..contractions.state_operator_contraction import tensordot
    if bra_node is None and bra_tensor is None:
        bra_node, bra_tensor = ket_node, lb.conjugate(ket_tensor)
    elif bra_node is None or bra_tensor is None:
        raise ValueError('Supply both bra_node and bra_tensor, or neither.')
    neighbours, ket, bra, op, pin, pout, hlegs = _layout(ket_node, op_node, id_trafo_op)
    if ignored_node_id not in neighbours:
        raise ValueError('Ignored node is not a ket neighbour.')
    transform = id_trafo_bra or (lambda x: x)
    mapped = tuple(transform(n) for n in neighbours)
    if len(set(mapped)) != len(mapped) or set(mapped) != set(bra_node.neighbouring_nodes()):
        raise ValueError('Ket/bra neighbours must match bijectively.')
    if bra_node.nopen_legs() != len(pout):
        raise ValueError('Bra physical legs disagree.')
    bmap = dict(zip(mapped, bra))
    blegs = tuple(bmap[n] for n in bra_node.neighbouring_nodes()) + pout
    operands, labels = [ket_tensor, op_tensor, bra_tensor], [ket + pin, hlegs, blegs]
    for i, n in enumerate(neighbours):
        if n != ignored_node_id:
            operands.append(dictionary.get_entry(n, ket_node.identifier))
            labels.append((ket[i], op[i], bra[i]))
    i = neighbours.index(ignored_node_id)
    plan = _make_plan(operands, labels, (ket[i], op[i], bra[i]))
    # Own labels while sharing immutable numerical blocks.
    operands = [lb.LasyTenArray(a.native.copy(deep=False)) for a in operands]
    for a in operands:
        a.native.set_labels([None] * a.ndim)
    result = operands[0]
    for index, left, right in plan[0]:
        # The existing sandwich tdot explicitly closes hidden ket/bra charges.
        result = tensordot(result, operands[index], axes=(left, right))
    if plan[1] != tuple(range(len(plan[1]))):
        result = lb.transpose(result, plan[1])
    _COUNTS['planned_environments'] += 1
    return result


def use_environment(ket, op, bra):
    if not _enabled('PLANNED_ENVIRONMENTS'):
        return False
    from .tensor_dispatch import is_lasyten_tensor
    return all(is_lasyten_tensor(x) for x in (ket, op)) and (bra is None or is_lasyten_tensor(bra))


class LazyNodeCopies(MutableMapping):
    """Snapshot dictionary, cloning mutable node metadata only when accessed.

    The mapping itself is copied, so additions/removals cannot leak. This
    reduces node cloning, not dictionary-copy asymptotics. Numerical tensors
    continue to follow the existing copy_state_for_update ownership rules.
    """
    def __init__(self, source):
        # Snapshot effective references without forcing every lazy node clone.
        self._data = dict(source._data if isinstance(source, LazyNodeCopies) else source)
        self._owned = set()

    def __getitem__(self, key):
        node = self._data[key]
        if key not in self._owned:
            node = copy(node)
            for name, value in vars(node).items():
                if isinstance(value, (list, dict, set)):
                    setattr(node, name, copy(value))
            self._data[key] = node
            self._owned.add(key)
            _COUNTS['node_metadata_clones'] += 1
        return node

    def __setitem__(self, key, value):
        self._data[key] = value
        self._owned.add(key)

    def __delitem__(self, key):
        del self._data[key]
        self._owned.discard(key)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data

    def __deepcopy__(self, memo):
        result = deepcopy(self._data, memo)
        memo[id(self)] = result
        return result


def copy_nodes(source):
    # Opt-in: recursive working views are supported, arbitrary concurrent
    # mutations of an ancestor working state are not. Default keeps old copies.
    if _enabled('LAZY_NODE_COPIES', False):
        return LazyNodeCopies(source)
    result = {}
    for key, node in source.items():
        cloned = copy(node)
        for name, value in vars(node).items():
            if isinstance(value, (list, dict, set)):
                setattr(cloned, name, copy(value))
        result[key] = cloned
    return result


def cached_truncation_path(tree, original_find_path):
    if not _enabled('CACHE_PATHS'):
        return original_find_path(tree)
    key = (tree.root_id, tuple((k, n.parent, tuple(n.children)) for k, n in tree.nodes.items()))
    path = _PATHS.get(key)
    if path is None:
        path = tuple(original_find_path(tree))
        if len(_PATHS) >= 16:
            _PATHS.pop(next(iter(_PATHS)))
        _PATHS[key] = path
        _COUNTS['truncation_path_misses'] += 1
    else:
        _COUNTS['truncation_path_hits'] += 1
    return list(path)


def encode_frontier_task(shared, encode, node_id, new_state, parent_state,
                         environments, hamiltonian, dt, config, options):
    """Reuse serialization only when all task payload objects are identical.

    The caller owns this cache for one root update and does not mutate these
    shared trees/environments during frontier submission. No cross-step cache.
    The native HDF5 payload does not contain node_id; metadata does.
    """
    key = tuple(id(x) for x in (new_state, parent_state, environments, hamiltonian, config, options)) + (dt,)
    if _enabled('SHARE_MPI_ENCODING') and key in shared:
        message = shared[key]
        metadata = pickle.dumps((node_id, dt, config, options), protocol=5)
        _COUNTS['mpi_encoding_reused'] += 1
        return dict(message, metadata=metadata)
    message = encode(node_id, new_state, parent_state, dict(environments), hamiltonian, dt, config, options)
    if message['codec'] == 'hdf5' and _enabled('SHARE_MPI_ENCODING'):
        shared[key] = message
    return message


def profile_local(function):
    @wraps(function)
    def wrapped(state_tensor, state_node, *args, **kwargs):
        if not _enabled('PROFILE', False):
            return function(state_tensor, state_node, *args, **kwargs)
        from .tensor_dispatch import is_lasyten_tensor
        record = None
        if is_lasyten_tensor(state_tensor):
            record = kwargs.get('stats')
            if record is None:
                record = {}
                kwargs['stats'] = record
        start = perf_counter()
        result = function(state_tensor, state_node, *args, **kwargs)
        elapsed = perf_counter() - start
        row = _LOCAL.setdefault(state_node.identifier, Counter())
        row['calls'] += 1
        row['seconds'] += elapsed
        for name in ('nfev', 'n_accepted', 'n_rejected'):
            if record is not None:
                row[name] += record.get(name, 0)
        return result
    return wrapped


def timed_call(name, function, *args, **kwargs):
    if not _enabled('PROFILE', False):
        return function(*args, **kwargs)
    start = perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        _TIMES[name] += perf_counter() - start
