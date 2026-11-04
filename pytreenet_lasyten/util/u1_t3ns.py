"""Fixed-particle-number U(1) states on an existing PyTreeNet topology.

Requires the accompanying LASyTen storage adapter. Physical axes use the
occupation basis |0>, |1>; an axis of dimension one is a charge-zero dummy.
This initializes a new state; it does not project the old tensor values.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from heapq import heappop, heappush
from math import comb
from numbers import Integral

import numpy as np

from pytreenet_lasyten.util import lasyten_backend as lb


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _topology(tree):
    root = tree.root_id
    if root is None:
        raise ValueError("The template must contain a root.")
    order, seen, dimensions = [], set(), {}

    def visit(node_id, parent):
        if node_id in seen:
            raise ValueError("The template must be an acyclic tree.")
        seen.add(node_id)
        node = tree.nodes[node_id]
        if node.parent != parent:
            raise ValueError(f"Inconsistent parent for {node_id}.")
        dims = tuple(node.open_dimensions())
        if len(dims) != 1 or dims[0] not in (1, 2):
            raise ValueError(f"{node_id} needs one physical leg of dimension 1 or 2.")
        dimensions[node_id] = dims[0]
        order.append(node_id)
        for child in node.children:
            visit(child, node_id)

    visit(root, None)
    if seen != set(tree.nodes):
        raise ValueError("The template contains disconnected nodes.")
    return order, dimensions


def _choose_sectors(capacities, budget, required, expected):
    """Allocate degeneracies while retaining one feasible occupation path."""
    if capacities.get(required, 0) < 1:
        raise ValueError("The reference charge is unreachable from the child spaces.")
    ranked = [required] + sorted((q for q in capacities if q != required),
                                key=lambda q: (abs(q - expected), q))
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


def random_u1_state_like(topology, *, total_particles, max_bond_dim=16,
                         seed=None, reference_occupations=None, tree_factory=None):
    """Build a normalized U(1) state with the template's rooted topology.

    Args:
        topology: The already-working T3NS tree; its tensor values are ignored.
        total_particles: Fixed total number of occupied spin orbitals.
        max_bond_dim: Total dimension across all sectors on each bond, or a
            mapping from every nonroot node identifier to that bond's cap.
        seed: NumPy random seed, used for occupation selection and block values.
        reference_occupations: Optional mapping of every dimension-two node to
            0 or 1, summing to total_particles. Its subtree charges are retained
            even when the bond cap cannot accommodate every feasible sector.
        tree_factory: Optional constructor accepting backend="lasyten";
            defaults to type(topology). Nodes are inserted using identifiers.

    Every nonroot tensor is an isometry towards its parent. The root is a
    ChargedTensor carrying total_particles; its norm fixes the network norm.
    Tensors are generated block by block and QR stays inside LASyTen. No full
    dense site tensor or many-body vector is constructed during initialization.
    """
    order, dimensions = _topology(topology)
    root = order[0]
    active = [i for i in topology.nodes if dimensions[i] == 2]
    modes = len(active)
    number = _integer(total_particles, "total_particles")
    if number > modes:
        raise ValueError(f"total_particles cannot exceed the {modes} spin orbitals.")
    if isinstance(max_bond_dim, Mapping):
        if set(max_bond_dim) != set(order[1:]):
            raise ValueError("The bond-cap mapping must contain every nonroot node exactly once.")
        budgets = {i: _integer(max_bond_dim[i], f"bond cap for {i}", 1) for i in order[1:]}
    else:
        cap = _integer(max_bond_dim, "max_bond_dim", 1)
        budgets = dict.fromkeys(order[1:], cap)
    rng = np.random.default_rng(seed)
    if reference_occupations is None:
        occupied = set(rng.choice(active, size=number, replace=False).tolist())
        occupations = {i: int(i in occupied) for i in active}
    else:
        if set(reference_occupations) != set(active):
            raise ValueError("reference_occupations must specify every physical dimension-two node.")
        occupations = {i: _integer(reference_occupations[i], f"occupation of {i}") for i in active}
        if any(n > 1 for n in occupations.values()) or sum(occupations.values()) != number:
            raise ValueError("Occupations must be 0 or 1 and sum to total_particles.")

    sectors, sizes, reference = {}, {}, {}
    for i in reversed(order):
        children = topology.nodes[i].children
        sizes[i] = int(dimensions[i] == 2) + sum(sizes[c] for c in children)
        reference[i] = occupations.get(i, 0) + sum(reference[c] for c in children)
        if i == root:
            continue
        budget = budgets[i]
        capacities = {0: 1, 1: 1} if dimensions[i] == 2 else {0: 1}
        for c in children:
            combined = Counter()
            for q, d in capacities.items():
                for qc, dc in sectors[c].items():
                    combined[q + qc] = min(budget, combined[q + qc] + d * dc)
            capacities = combined
        lower, upper = max(0, number - (modes - sizes[i])), min(number, sizes[i])
        capacities = {q: min(d, budget, comb(modes - sizes[i], number - q))
                      for q, d in capacities.items() if lower <= q <= upper}
        sectors[i] = _choose_sectors(capacities, budget, reference[i],
                                     number * sizes[i] / max(1, modes))

    ct = lb._lasyten()
    symmetry = lb._symmetry(ct.U1())
    backend = ct.get_backend(symmetry=symmetry, block_backend="numpy")
    dtype = ct.Dtype.from_numpy_dtype(np.dtype("complex128"))
    spaces = {i: lb.u1_space([q for q, d in sectors[i].items() for _ in range(d)])
              for i in order[1:]}
    physical_spaces = {1: lb.u1_space([0]), 2: lb.u1_space([0, 1])}

    def gaussian(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)

    tensors = {}
    for i in reversed(order):
        outward = ([] if i == root else [spaces[i].dual])
        outward += [spaces[c] for c in topology.nodes[i].children]
        outward += [physical_spaces[dimensions[i]]]
        args = dict(func=gaussian, codomain=outward, backend=backend, dtype=dtype)
        if i == root:
            tensor = lb.LasyTenArray(ct.ChargedTensor.from_block_func(
                charge=lb.u1_space([number]), charged_state=np.ones(1, dtype=complex), **args))
            length = lb.norm(tensor)
            if not np.isfinite(length) or length == 0:
                raise RuntimeError("Initialization produced an empty root charge sector.")
            tensor = tensor / length
        else:
            tensor = lb.LasyTenArray(ct.SymmetricTensor.from_block_func(**args))
            q, _ = lb.tensor_qr_decomposition(tensor, tuple(range(1, tensor.ndim)), (0,))
            tensor = lb.transpose(q, (q.ndim - 1,) + tuple(range(q.ndim - 1)))
            if tensor.legs[0] != spaces[i].dual:
                raise RuntimeError(f"QR changed the allocated bond space at {i}.")
        tensors[i] = tensor

    state = (tree_factory or type(topology))(backend="lasyten")
    state.add_root(root, tensors[root])
    # Preserve the source insertion order whenever it respects the current
    # parent/sibling order (as it does for the supplied topology). Rerooted
    # templates can require a different insertion sequence.
    rank = {i: k for k, i in enumerate(topology.nodes)}
    ready = [(rank[root], root)]
    while ready:
        _, i = heappop(ready)
        parent = topology.nodes[i].parent
        if i != root:
            siblings = topology.nodes[parent].children
            position = siblings.index(i)
            parent_axis = int(parent != root) + position
            state.add_child_to_parent(i, tensors[i], 0, parent, parent_axis)
            if position + 1 < len(siblings):
                sibling = siblings[position + 1]
                heappush(ready, (rank[sibling], sibling))
        children = topology.nodes[i].children
        if children:
            heappush(ready, (rank[children[0]], children[0]))
    state.orthogonality_center_id = root
    state.u1_info = dict(total_particles=number, physical_nodes=tuple(active),
                         reference_occupations=occupations, bond_sectors=sectors)
    return state


def check_u1_state(state, *, atol=1e-10):
    """Check native U(1) spaces, dual bonds, subtree isometries and root norm.

    Returns the largest isometry residual; no many-body vector is constructed.
    Requires an unmodified state returned by random_u1_state_like.
    """
    from .abelian_charges import is_pair_state
    if is_pair_state(state):
        from .u1u1_t3ns import check_u1u1_state
        return check_u1u1_state(state, atol=atol)
    order, _ = _topology(state)
    ct = lb._lasyten()
    symmetry = lb._symmetry(ct.U1())
    worst = 0.0
    for i in order:
        a = state.tensors[i]
        if not a.native.symmetry.is_equivalent_to(symmetry):
            raise ValueError(f"{i} does not use U(1).")
        for axis, child in enumerate(state.nodes[i].children, start=int(i != state.root_id)):
            if a.legs[axis] != state.tensors[child].legs[0].dual:
                raise ValueError(f"Incompatible U(1) spaces at {i} -> {child}.")
        if i != state.root_id:
            other = tuple(range(1, a.ndim))
            gram = lb.tensordot(lb.conjugate(a), a, axes=(other, other))
            identity = lb.LasyTenArray(ct.SymmetricTensor.from_eye(
                [a.legs[0].dual], backend=a.native.backend))
            residual = lb.norm(gram - identity)
            worst = max(worst, residual)
    root = state.tensors[state.root_id]
    if not isinstance(root.native, ct.ChargedTensor):
        raise ValueError("The root must carry the total particle charge.")
    if root.native.charge_leg != lb.u1_space([state.u1_info["total_particles"]]):
        raise ValueError("The root charge disagrees with total_particles.")
    if worst > atol or abs(lb.norm(root) - 1) > atol:
        raise ValueError(f"State is not normalized in root canonical form (residual {worst:g}).")
    return worst


def dense_state_for_checks(state, *, max_physical_sites=16):
    """Explicit small-system check: return a dense occupation vector and order.

    Only this diagnostic helper densifies. Dimension-one dummy axes are removed.
    """
    order, dimensions = _topology(state)
    count = sum(d == 2 for d in dimensions.values())
    if count > max_physical_sites:
        raise ValueError(f"Dense check limited to {max_physical_sites} physical sites; got {count}.")
    subtrees = {}
    for i in reversed(order):
        a = state.tensors[i]
        labels = ([None] if i != state.root_id else []) + list(state.nodes[i].children) + [i]
        for child in state.nodes[i].children:
            b, b_labels = subtrees.pop(child)
            axis = labels.index(child)
            a = lb.tensordot(a, b, axes=([axis], [0]))
            labels = labels[:axis] + labels[axis + 1:] + b_labels[1:]
        subtrees[i] = a, labels
    a, labels = subtrees[state.root_id]
    physical_order = tuple(i for i in labels if dimensions[i] == 2)
    return a.to_numpy().reshape((2,) * count), physical_order
