"""Local fermion excitations of fixed-N U(1) TTNS, with explicit JW strings.

Charge is transported to the root along the operated site's ancestor path.
All site tensors stay native. Only local 2x2 operators and bond identity maps
are initialized densely; no site tensor or many-body state is exported.
"""
from copy import deepcopy
from numbers import Integral

import numpy as np

from . import lasyten_backend as lb
from ..ttns.ttns import TreeTensorNetworkState
from .tensor_splitting import SplitMode

def _state_copy(state):
    result = deepcopy(state)
    if not isinstance(result, TreeTensorNetworkState):
        info = getattr(result, 'u1_info', None)
        result = TreeTensorNetworkState.from_ttn(result)
        if info is not None:
            result.u1_info = info
    return result


def _fixed_number(state):
    ct = lb._lasyten()
    root = state.root_id
    if root is None:
        raise ValueError('The input tree must contain a root.')
    symmetry = lb._symmetry(ct.U1())
    for i, node in state.nodes.items():
        a = lb.as_lasyten(state.tensors[i])
        if not a.native.symmetry.is_equivalent_to(symmetry):
            raise ValueError('Excitations require native bosonic U(1) tensors.')
        if len(node.open_legs) != 1:
            raise ValueError('Each node must have one physical occupation/dummy leg.')
        if i != root and not isinstance(a.native, ct.SymmetricTensor):
            raise ValueError('Only the root may carry a ChargedTensor.')
    a = lb.as_lasyten(state.tensors[root])
    if not isinstance(a.native, ct.ChargedTensor):
        raise ValueError('The root must carry the fixed total-N ChargedTensor.')
    charge = a.native.charge_leg
    sectors = np.asarray(charge.sector_decomposition)
    if charge.dim != 1 or charge.is_dual or sectors.shape != (1, 1):
        raise ValueError('The root must carry one nondual, one-dimensional U(1) charge sector.')
    return int(sectors[0, 0])


def _store(state, node_id, a):
    # The preceding getter has materialized logical axis order. Relinking is
    # necessary because the virtual spaces change, even though their sizes do not.
    a.native.set_labels([None] * a.ndim)
    state.nodes[node_id].link_tensor(a)
    state.tensors[node_id] = a


def _small_operator(data, spaces, like):
    native = lb._lasyten().SymmetricTensor.from_dense_block(
        data, codomain=spaces, backend=like.native.backend,
        device=like.native.device, tol=1e-12)
    return lb.LasyTenArray(native)


def _shift_root_charge(a):
    """Move the last public charge-wire leg into the root's charge space."""
    ct = lb._lasyten()
    if not isinstance(a.native, ct.ChargedTensor):
        raise ValueError('Charge transport must end at the charged root.')
    npublic = a.ndim - 1  # exclude the extra public charge-wire leg
    invariant = a.native.invariant_part
    invariant = ct.permute_legs(
        invariant, codomain=list(range(npublic)),
        domain=[npublic + 1, npublic])
    invariant = ct.combine_legs(
        invariant, [npublic, npublic + 1], pipe_dualities=True)
    invariant = ct.flatten_pipe_leg(invariant, npublic)
    invariant.set_labels([None] * npublic + ['!'])
    return lb.LasyTenArray(ct.ChargedTensor(invariant, a.native.charged_state))


def apply_local_charge_operator(state, operator, *, node_id, delta,
                                 parity_nodes=()):
    """Return an unnormalized copy with an explicit JW local operator applied.

    ``operator`` uses basis |0>, |1> and has definite particle transfer delta.
    Every site in parity_nodes receives diag(1,-1). Their order is immaterial;
    callers must supply exactly the preceding modes in their JW ordering.
    The root's charge changes from N to N+delta. Bond sizes remain unchanged.
    """
    from .abelian_charges import is_pair_state
    if is_pair_state(state):
        from .u1u1_excitations import apply_local_pair_charge_operator
        return apply_local_pair_charge_operator(state, operator, node_id=node_id,
                                                delta=delta, parity_nodes=parity_nodes)
    if isinstance(delta, bool) or not isinstance(delta, Integral):
        raise TypeError('delta must be an integer particle transfer.')
    delta = int(delta)
    data = np.asarray(operator, dtype=complex)
    if data.shape != (2, 2) or not np.all(np.isfinite(data)):
        raise ValueError('The local operator must be a finite 2x2 matrix.')
    rows, columns = np.nonzero(data)
    if len(rows) and np.any(rows - columns != delta):
        raise ValueError('The local operator does not have the declared particle transfer.')
    parity_nodes = tuple(parity_nodes)
    if len(set(parity_nodes)) != len(parity_nodes) or node_id in parity_nodes:
        raise ValueError('Parity nodes must be distinct and exclude the operated site.')
    result = _state_copy(state)
    number = _fixed_number(result)
    physical = lb.u1_space([0, 1])
    for i in (node_id,) + parity_nodes:
        if i not in result.nodes:
            raise ValueError(f'Unknown physical node {i!r}.')
        a = result.tensors[i]
        if a.legs[-1] != physical:
            raise ValueError(f'{i!r} must use occupation basis |0>, |1> with charges 0,1.')

    for i in parity_nodes:
        a = result.tensors[i]
        parity = _small_operator(np.diag([1., -1.]), [physical, physical.dual], a)
        _store(result, i, lb.tensordot(a, parity, axes=([-1], [1])))

    a = result.tensors[node_id]
    wire = lb.u1_space([delta]).dual
    local = _small_operator(data[:, :, None], [physical, physical.dual, wire], a)
    a = lb.tensordot(a, local, axes=([-1], [1]))
    current = node_id
    while current != result.root_id:
        # Parent leg plus a one-dimensional wire: same size, shifted charges.
        a = lb.fuse_legs(a, (0, a.ndim - 1), pipe_dualities=True)
        _store(result, current, a)
        new_child_space = a.legs[0].dual
        parent = result.nodes[current].parent
        parent_node, b = result[parent]
        axis = parent_node.neighbour_index(current)
        rank = b.ndim
        shift = _small_operator(
            np.eye(b.shape[axis], dtype=complex)[:, :, None],
            [new_child_space, b.legs[axis].dual, wire], b)
        a = lb.tensordot(b, shift, axes=([axis], [1]))
        permutation = (list(range(axis)) + [rank - 1]
                       + list(range(axis, rank - 1)) + [rank])
        a = lb.transpose(a, permutation)
        current = parent
    _store(result, current, _shift_root_charge(a))
    result.orthogonality_center_id = None
    # Initialization-only occupation and isometry metadata do not describe the
    # new noncanonical state. Actual charges always live on native leg spaces.
    result.u1_info = dict(
        total_particles=number + delta,
        physical_nodes=tuple(i for i, n in result.nodes.items()
                             if tuple(n.open_dimensions()) == (2,)),
    )
    result.excitation_info = dict(node_id=node_id, delta=delta,
                                  parity_nodes=parity_nodes)
    return result


def create_impurity_excitations(ground_state, impurity_node='Impurity_1_up',
                                parity_nodes=(), normalise=True,
                                *, zero_tol=1e-14):
    """Return removal, addition, Z_removal, Z_addition for a normalized input.

    The ground state is copied and normalized internally. Returned excitation
    states are normalized if requested. A zero-weight branch raises as in the
    original topology helper. No canonicalization kernels are needed.
    """
    from .abelian_charges import is_pair_state
    if is_pair_state(ground_state):
        from .u1u1_excitations import create_pair_impurity_excitations
        return create_pair_impurity_excitations(ground_state, impurity_node=impurity_node,
            parity_nodes=parity_nodes, normalise=normalise, zero_tol=zero_tol)
    if not np.isfinite(zero_tol) or zero_tol < 0:
        raise ValueError('zero_tol must be finite and nonnegative.')
    ground = _state_copy(ground_state)
    ground.ensure_root_orth_center(mode=SplitMode.REDUCED)
    _fixed_number(ground)
    norm_sq = float(ground.scalar_product(use_orthogonal_center=False).real)
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        raise ValueError('The input state must have a finite nonzero norm.')
    ground.tensors[ground.root_id] = ground.tensors[ground.root_id] / np.sqrt(norm_sq)
    ground.orthogonality_center_id = None
    d = np.array([[0., 1.], [0., 0.]])
    removal = apply_local_charge_operator(ground, d, node_id=impurity_node,
                                          delta=-1, parity_nodes=parity_nodes)
    addition = apply_local_charge_operator(ground, d.T, node_id=impurity_node,
                                           delta=1, parity_nodes=parity_nodes)
    z_removal = float(removal.scalar_product(use_orthogonal_center=False).real)
    z_addition = float(addition.scalar_product(use_orthogonal_center=False).real)
    if z_removal <= zero_tol:
        raise ValueError('d|GS> has approximately zero norm; skip this zero-weight branch.')
    if z_addition <= zero_tol:
        raise ValueError('d^dagger|GS> has approximately zero norm; skip this zero-weight branch.')
    if normalise:
        removal.tensors[removal.root_id] = removal.tensors[removal.root_id] / np.sqrt(z_removal)
        addition.tensors[addition.root_id] = addition.tensors[addition.root_id] / np.sqrt(z_addition)
    return removal, addition, z_removal, z_addition
