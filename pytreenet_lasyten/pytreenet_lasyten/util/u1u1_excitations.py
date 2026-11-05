"""Spin-resolved local excitations; root charge transfer stays native."""
import numpy as np
from numbers import Integral

from . import lasyten_backend as lb
from .abelian_charges import (charge_tuple, charge_space, add_charge, fixed_charge,
                              physical_basis, charge_metadata, check_native_charges)
from .tensor_splitting import SplitMode
from .u1_excitations import _state_copy, _store, _small_operator, _shift_root_charge


def apply_local_pair_charge_operator(state, operator, *, node_id, delta, parity_nodes=()):
    result = _state_copy(state)
    result.ensure_root_orth_center(mode=SplitMode.REDUCED)
    charge = check_native_charges(result)
    if len(charge) != 2:
        raise ValueError('Expected two U(1) charges.')
    data = np.asarray(operator, dtype=complex)
    if data.shape != (2, 2) or not np.all(np.isfinite(data)):
        raise ValueError('The local operator must be a finite 2x2 matrix.')
    if node_id not in result.nodes:
        raise ValueError(f'Unknown physical node {node_id!r}.')
    physical = result.tensors[node_id].legs[-1]
    basis = physical_basis(physical)
    if len(basis) != 2:
        raise ValueError('The operated site must be a physical occupation mode.')
    if isinstance(delta, (bool, np.bool_)):
        raise ValueError('delta must be an integer or an integer charge pair.')
    transfer = tuple(int(delta)*v for v in basis[1]) if isinstance(delta, Integral) else charge_tuple(delta)
    rows, cols = np.nonzero(data)
    if any(tuple(basis[r][k]-basis[c][k] for k in (0, 1)) != transfer for r, c in zip(rows, cols)):
        raise ValueError('The local operator does not have the declared charge transfer.')
    parity_nodes = tuple(parity_nodes)
    if len(set(parity_nodes)) != len(parity_nodes) or node_id in parity_nodes:
        raise ValueError('Parity nodes must be distinct and exclude the operated site.')
    for i in parity_nodes:
        if i not in result.nodes:
            raise ValueError(f'Unknown parity node {i!r}.')
        a = result.tensors[i]
        p = a.legs[-1]
        if len(physical_basis(p)) != 2:
            raise ValueError('JW parity nodes must be physical occupation modes.')
        parity = _small_operator(np.diag([1., -1.]), [p, p.dual], a)
        _store(result, i, lb.tensordot(a, parity, axes=([-1], [1])))
    a = result.tensors[node_id]
    wire = charge_space([transfer]).dual
    local = _small_operator(data[:, :, None], [physical, physical.dual, wire], a)
    a = lb.tensordot(a, local, axes=([-1], [1]))
    current = node_id
    while current != result.root_id:
        a = lb.fuse_legs(a, (0, a.ndim-1), pipe_dualities=True)
        _store(result, current, a)
        new_child_space = a.legs[0].dual
        parent = result.nodes[current].parent
        parent_node, b = result[parent]
        axis, rank = parent_node.neighbour_index(current), b.ndim
        shift = _small_operator(np.eye(b.shape[axis], dtype=complex)[:, :, None],
                               [new_child_space, b.legs[axis].dual, wire], b)
        a = lb.tensordot(b, shift, axes=([axis], [1]))
        permutation = list(range(axis)) + [rank-1] + list(range(axis, rank-1)) + [rank]
        a = lb.transpose(a, permutation)
        current = parent
    _store(result, current, _shift_root_charge(a))
    result.orthogonality_center_id = None
    new_charge = add_charge(charge, transfer)
    result.u1_info = charge_metadata(new_charge, (i for i, n in result.nodes.items() if tuple(n.open_dimensions()) == (2,)))
    result.excitation_info = dict(node_id=node_id, delta=transfer, parity_nodes=parity_nodes)
    check_native_charges(result)
    return result


def create_pair_impurity_excitations(ground_state, impurity_node='Impurity_1_up',
                                     parity_nodes=(), normalise=True, *, zero_tol=1e-14):
    if not np.isfinite(zero_tol) or zero_tol < 0:
        raise ValueError('zero_tol must be finite and nonnegative.')
    ground = _state_copy(ground_state)
    ground.ensure_root_orth_center(mode=SplitMode.REDUCED)
    check_native_charges(ground)
    norm_sq = float(ground.scalar_product(use_orthogonal_center=False).real)
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        raise ValueError('The ground state must have a finite nonzero norm.')
    ground.tensors[ground.root_id] = ground.tensors[ground.root_id] / np.sqrt(norm_sq)
    ground.orthogonality_center_id = None
    d = np.array([[0., 1.], [0., 0.]])
    removal = apply_local_pair_charge_operator(ground, d, node_id=impurity_node, delta=-1, parity_nodes=parity_nodes)
    addition = apply_local_pair_charge_operator(ground, d.T, node_id=impurity_node, delta=1, parity_nodes=parity_nodes)
    zr = float(removal.scalar_product(use_orthogonal_center=False).real)
    za = float(addition.scalar_product(use_orthogonal_center=False).real)
    if not np.isfinite(zr+za) or zr <= zero_tol or za <= zero_tol:
        raise ValueError('An excitation has zero/nonfinite weight; skip that branch explicitly.')
    if normalise:
        removal.tensors[removal.root_id] = removal.tensors[removal.root_id] / np.sqrt(zr)
        addition.tensors[addition.root_id] = addition.tensors[addition.root_id] / np.sqrt(za)
    return removal, addition, zr, za
