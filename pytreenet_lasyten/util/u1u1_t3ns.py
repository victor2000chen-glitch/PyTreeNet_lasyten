"""Initialize spin-resolved fixed-charge trees directly in allowed native blocks."""
from collections import Counter
from collections.abc import Mapping
from heapq import heappop, heappush
from math import comb

import numpy as np

from . import lasyten_backend as lb
from .abelian_charges import (add_charge, charge_tuple, charge_space, charge_metadata,
                              occupation_charges, symmetry_for, check_native_charges)
from .u1_t3ns import _topology, _integer


def _choose_pair_sectors(capacities, budget, required, expected):
    if capacities.get(required, 0) < 1:
        raise ValueError('The reference occupation path is not feasible.')
    distance = lambda q: sum((a-b)**2 for a, b in zip(q, expected))
    ranked = [required] + sorted((q for q in capacities if q != required), key=lambda q: (distance(q), q))
    kept = {q: 1 for q in ranked[:budget]}
    remaining = budget - len(kept)
    while remaining:
        available = [q for q in kept if kept[q] < capacities[q]]
        if not available:
            break
        for q in available:
            kept[q] += 1
            remaining -= 1
            if not remaining:
                break
    return dict(sorted(kept.items()))


def _allocate_pair_sectors(order, children, physical, budgets, target, occupations):
    """Pure bookkeeping; bond budgets are TOTAL dimensions across charge pairs."""
    total_modes = tuple(sum(len(physical[i]) == 2 and physical[i][1][k] == 1 for i in order) for k in (0, 1))
    if any(q < 0 or q > m for q, m in zip(target, total_modes)):
        raise ValueError('Requested spin populations exceed the available modes.')
    root = order[0]
    sizes, reference, sectors = {}, {}, {}
    for i in reversed(order):
        unit = physical[i][-1]
        sizes[i] = tuple(unit[k] + sum(sizes[c][k] for c in children[i]) for k in (0, 1))
        local = tuple(occupations.get(i, 0) * q for q in unit)
        reference[i] = tuple(local[k] + sum(reference[c][k] for c in children[i]) for k in (0, 1))
        if i == root:
            continue
        budget = budgets[i]
        capacities = Counter(physical[i])
        for child in children[i]:
            combined = Counter()
            for q, d in capacities.items():
                for qc, dc in sectors[child].items():
                    key = add_charge(q, qc)
                    combined[key] = min(budget, combined[key] + d*dc)
            capacities = combined
        feasible = {}
        for q, d in capacities.items():
            outside = tuple(m-s for m, s in zip(total_modes, sizes[i]))
            needed = tuple(n-v for n, v in zip(target, q))
            if all(0 <= n <= m for n, m in zip(needed, outside)):
                complement = comb(outside[0], needed[0]) * comb(outside[1], needed[1])
                feasible[q] = min(d, budget, complement)
        expected = tuple(target[k]*sizes[i][k]/max(1, total_modes[k]) for k in (0, 1))
        sectors[i] = _choose_pair_sectors(feasible, budget, reference[i], expected)
    if reference[root] != target:
        raise ValueError('Reference occupations disagree with the target spin populations.')
    return sectors


def _qr_parent_space(tensor, allocated, budget, node_id):
    """Use QR's actual bond basis rather than requiring allocation identity.

    QR can change the public ordering of a virtual basis or reduce a sector
    to the space supported by its rows. This is a new random initializer:
    its parent tensor has not been generated yet, so no R factor needs to
    be absorbed. The parent must, however, use the returned Q leg's dual.
    Only small charge-label arrays are inspected; tensor blocks stay native.
    """
    space = tensor.legs[0].dual
    if space.is_dual or not space.symmetry.is_equivalent_to(symmetry_for(2)):
        raise RuntimeError(f'QR returned an invalid parent-space orientation or symmetry at {node_id}.')
    basis = np.asarray(space.sectors_of_basis)
    if basis.ndim != 2 or basis.shape != (int(space.dim), 2) or basis.dtype.kind not in 'iu':
        raise RuntimeError(f'QR returned an invalid pair-charge table at {node_id}.')
    actual = dict(sorted(Counter(tuple(int(v) for v in q) for q in basis).items()))
    if not 0 < sum(actual.values()) <= budget:
        raise RuntimeError(f'QR returned an empty or over-budget parent space at {node_id}.')
    if any(d > allocated.get(q, 0) for q, d in actual.items()):
        raise RuntimeError(f'QR expanded an unallocated pair-charge sector at {node_id}.')
    return space, actual


def random_u1u1_state_like(topology, *, total_particles_by_spin, max_bond_dim=16,
                          seed=None, reference_occupations=None, spin_by_node=None,
                          tree_factory=None):
    target = charge_tuple(total_particles_by_spin)
    order, dimensions = _topology(topology)
    root = order[0]
    children = {i: list(topology.nodes[i].children) for i in order}
    physical = {i: occupation_charges(i, dimensions[i], 2, spin_by_node) for i in order}
    active = [i for i in order if dimensions[i] == 2]
    by_spin = [[i for i in active if physical[i][1][k] == 1] for k in (0, 1)]
    if any(n < 0 or n > len(nodes) for n, nodes in zip(target, by_spin)):
        raise ValueError('Requested spin populations exceed the available modes.')
    if isinstance(max_bond_dim, Mapping):
        if set(max_bond_dim) != set(order[1:]):
            raise ValueError('Supply one bond cap for every nonroot node.')
        budgets = {i: _integer(max_bond_dim[i], 'bond cap', 1) for i in order[1:]}
    else:
        cap = _integer(max_bond_dim, 'max_bond_dim', 1)
        budgets = dict.fromkeys(order[1:], cap)
    rng = np.random.default_rng(seed)
    if reference_occupations is None:
        occupied = set()
        for n, nodes in zip(target, by_spin):
            occupied.update(rng.choice(nodes, size=n, replace=False).tolist())
        occupations = {i: int(i in occupied) for i in active}
    else:
        if set(reference_occupations) != set(active):
            raise ValueError('reference_occupations must specify every physical mode.')
        occupations = {i: _integer(reference_occupations[i], 'occupation') for i in active}
        if any(n > 1 for n in occupations.values()) or tuple(sum(occupations[i] for i in nodes) for nodes in by_spin) != target:
            raise ValueError('Reference occupations must be 0/1 with the requested spin populations.')
    sectors = _allocate_pair_sectors(order, children, physical, budgets, target, occupations)
    spaces = {i: charge_space([q for q, d in sectors[i].items() for _ in range(d)]) for i in order[1:]}
    physical_spaces = {i: charge_space(physical[i]) for i in order}
    ct = lb._lasyten()
    backend = ct.get_backend(symmetry=symmetry_for(2), block_backend='numpy')
    dtype = ct.Dtype.from_numpy_dtype(np.dtype('complex128'))
    gaussian = lambda shape: (rng.standard_normal(shape) + 1j*rng.standard_normal(shape))/np.sqrt(2)
    tensors = {}
    qr_adjusted_bonds = []
    for i in reversed(order):
        outward = ([] if i == root else [spaces[i].dual]) + [spaces[c] for c in children[i]] + [physical_spaces[i]]
        kwargs = dict(func=gaussian, codomain=outward, backend=backend, dtype=dtype)
        if i == root:
            a = lb.LasyTenArray(ct.ChargedTensor.from_block_func(
                charge=charge_space([target]), charged_state=np.ones(1, dtype=complex), **kwargs))
            norm = lb.norm(a)
            if not np.isfinite(norm) or norm <= 0:
                raise RuntimeError('The initialized root charge sector is zero or nonfinite.')
            a = a / norm
        else:
            a = lb.LasyTenArray(ct.SymmetricTensor.from_block_func(**kwargs))
            q, _ = lb.tensor_qr_decomposition(a, tuple(range(1, a.ndim)), (0,))
            a = lb.transpose(q, (q.ndim-1,) + tuple(range(q.ndim-1)))
            if a.legs[0] != spaces[i].dual:
                qr_adjusted_bonds.append(i)
            # Children are processed before their parent. Propagate Q's real
            # space now, before generating that parent's random blocks.
            spaces[i], sectors[i] = _qr_parent_space(a, sectors[i], budgets[i], i)
        tensors[i] = a
    state = (tree_factory or type(topology))(backend='lasyten')
    state.add_root(root, tensors[root])
    rank = {i: k for k, i in enumerate(topology.nodes)}
    ready = [(rank[root], root)]
    while ready:
        _, i = heappop(ready)
        parent = topology.nodes[i].parent
        if i != root:
            siblings = children[parent]
            position = siblings.index(i)
            state.add_child_to_parent(i, tensors[i], 0, parent, int(parent != root) + position)
            if position+1 < len(siblings):
                child = siblings[position+1]
                heappush(ready, (rank[child], child))
        if children[i]:
            child = children[i][0]
            heappush(ready, (rank[child], child))
    state.orthogonality_center_id = root
    state.u1_info = charge_metadata(target, active)
    state.u1_info.update(reference_occupations=occupations, bond_sectors=sectors,
                         physical_charges=physical, qr_adjusted_bonds=tuple(qr_adjusted_bonds))
    check_u1u1_state(state)
    return state


def check_u1u1_state(state, *, atol=1e-10):
    charge = check_native_charges(state)
    if len(charge) != 2:
        raise ValueError('Expected U(1) x U(1).')
    worst = 0.
    for i in state.nodes:
        if i == state.root_id:
            continue
        a = state.tensors[i]
        axes = tuple(range(1, a.ndim))
        gram = lb.tensordot(lb.conjugate(a), a, axes=(axes, axes))
        identity = lb.LasyTenArray(lb._lasyten().SymmetricTensor.from_eye([a.legs[0].dual], backend=a.native.backend))
        worst = max(worst, lb.norm(gram-identity))
    if worst > atol or abs(lb.norm(state.tensors[state.root_id])-1) > atol:
        raise ValueError(f'State is not normalized in root canonical form (residual {worst:g}).')
    return worst
