"""Shared charge bookkeeping for bosonic U(1) and U(1) x U(1).

The two factor order is always (N_up, N_down). Fermionic statistics continue
to be represented by the model's existing Jordan-Wigner strings.
"""
from numbers import Integral
from functools import lru_cache
import re

import numpy as np


def charge_tuple(value, rank=2):
    values = (value,) if isinstance(value, Integral) else tuple(value)
    if len(values) != rank or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in values):
        raise ValueError(f'Expected {rank} integer charge components, got {value!r}.')
    return tuple(int(v) for v in values)


def add_charge(a, b):
    if len(a) != len(b):
        raise ValueError('Cannot add charges from different symmetry groups.')
    return tuple(x + y for x, y in zip(a, b))


@lru_cache(maxsize=2)
def symmetry_for(rank):
    from . import lasyten_backend as lb
    ct = lb._lasyten()
    if rank == 1:
        return lb._symmetry(ct.U1())
    if rank == 2:
        return lb._symmetry(ct.U1() * ct.U1())
    raise ValueError('Only one or two bosonic U(1) factors are supported.')


def symmetry_rank(symmetry):
    for rank in (1, 2):
        if symmetry.is_equivalent_to(symmetry_for(rank)):
            return rank
    raise ValueError('Expected bosonic U(1) or U(1) x U(1).')


def charge_space(charges, rank=2, *, dual=False):
    from . import lasyten_backend as lb
    charges = list(charges)
    if not charges:
        raise ValueError('A charge space needs at least one basis vector.')
    data = np.array([charge_tuple(q, rank) for q in charges], dtype=np.int64)
    space = lb._lasyten().ElementarySpace.from_basis(symmetry_for(rank), data)
    return space.dual if dual else space


def spin_index(node_id, spin_by_node=None):
    spin = None if spin_by_node is None else spin_by_node.get(node_id)
    if spin is None:
        match = re.search(r'_(up|down)(?:_\d+)?$', str(node_id))
        spin = match.group(1) if match else None
    if spin not in ('up', 'down'):
        raise ValueError(f'Cannot identify spin of {node_id!r}; supply spin_by_node with up/down labels.')
    return int(spin == 'down')


def occupation_charges(node_id, dimension, rank=2, spin_by_node=None):
    zero = (0,) * rank
    if dimension == 1:
        return (zero,)
    if dimension != 2:
        raise ValueError('Each physical leg must be a dimension-one dummy or a two-state occupation mode.')
    index = 0 if rank == 1 else spin_index(node_id, spin_by_node)
    unit = tuple(int(k == index) for k in range(rank))
    return zero, unit


def physical_basis(space):
    rank = symmetry_rank(space.symmetry)
    if space.is_dual:
        raise ValueError('State physical occupation legs must be nondual.')
    data = np.asarray(space.sectors_of_basis, dtype=np.int64)
    if data.shape != (int(space.dim), rank):
        raise ValueError('Invalid physical charge table.')
    result = tuple(tuple(int(v) for v in row) for row in data)
    if result == ((0,) * rank,):
        return result
    if len(result) != 2 or result[0] != (0,) * rank or sum(result[1]) != 1 or any(q not in (0, 1) for q in result[1]):
        raise ValueError('Physical legs must use |0>, |1> with one occupied spin-mode charge.')
    return result


def fixed_charge(state):
    from . import lasyten_backend as lb
    root = state.root_id
    if root is None:
        raise ValueError('The state has no root.')
    native = state.tensors[root].native
    if not isinstance(native, lb._lasyten().ChargedTensor):
        raise ValueError('Move the charged orthogonality centre to the root first.')
    rank = symmetry_rank(native.symmetry)
    space = native.charge_leg
    values = np.asarray(space.sector_decomposition)
    if space.dim != 1 or space.is_dual or values.shape != (1, rank):
        raise ValueError('Expected one nondual, one-dimensional fixed charge representation.')
    return charge_tuple(values[0], rank)


def is_pair_state(state):
    from . import lasyten_backend as lb
    root = state.root_id
    if root is None:
        return False
    tensor = state.tensors.data[root]
    return (isinstance(tensor, lb.LasyTenArray)
            and tensor.native.symmetry.is_equivalent_to(symmetry_for(2)))


def charge_metadata(charge, physical_nodes):
    charge = tuple(charge)
    info = dict(total_particles=sum(charge), physical_nodes=tuple(physical_nodes),
                charge_labels=('N',) if len(charge) == 1 else ('N_up', 'N_down'))
    if len(charge) == 2:
        info['total_particles_by_spin'] = charge
    return info


def check_native_charges(state, operator=None):
    """Check actual tensor spaces after initialization, DMRG or BUG; return tuple."""
    from . import lasyten_backend as lb
    ct = lb._lasyten()
    charge = fixed_charge(state)
    rank = len(charge)
    if operator is not None and (operator.root_id != state.root_id or set(operator.nodes) != set(state.nodes)):
        raise ValueError('State and operator topologies differ.')
    for network, is_operator in ((state, False), (operator, True)):
        if network is None:
            continue
        for identifier, node in network.nodes.items():
            tensor = network.tensors[identifier]
            if not isinstance(tensor, lb.LasyTenArray) or symmetry_rank(tensor.native.symmetry) != rank:
                raise ValueError(f'Inconsistent native symmetry at {identifier}.')
            physical = state.tensors[identifier].legs[-1]
            physical_basis(physical)
            expected = ct.SymmetricTensor if is_operator or identifier != state.root_id else ct.ChargedTensor
            if not isinstance(tensor.native, expected):
                raise ValueError(f'Invalid charge carrier at {identifier}.')
            if is_operator:
                if tensor.legs[-2] != physical or tensor.legs[-1] != physical.dual:
                    raise ValueError(f'Operator physical charge spaces differ at {identifier}.')
                other = state.nodes[identifier]
                if node.parent != other.parent or set(node.children) != set(other.children):
                    raise ValueError(f'Operator connectivity differs at {identifier}.')
            for axis, child in enumerate(node.children, start=int(node.parent is not None)):
                if tensor.legs[axis] != network.tensors[child].legs[0].dual:
                    raise ValueError(f'Bond charge spaces are not dual at {identifier} -> {child}.')
    info = getattr(state, 'u1_info', {})
    if 'total_particles' in info and info['total_particles'] != sum(charge):
        raise ValueError('Total particle metadata disagrees with the native root charge.')
    if rank == 2 and 'total_particles_by_spin' in info and tuple(info['total_particles_by_spin']) != charge:
        raise ValueError('Spin population metadata disagrees with the native root charge.')
    if operator is not None:
        value = operator.u1_operator_info.get('total_charge')
        if charge_tuple(value, rank) != (0,) * rank:
            raise ValueError('The TTNO must conserve every charge component.')
    return charge
