"""T3NS topology shared by one- and two-charge HK calculations."""
import numpy as np

from ..core.node import Node
from ..ttns.ttns import TreeTensorNetworkState
from .u1_t3ns import random_u1_state_like, check_u1_state, _integer
from .u1u1_t3ns import random_u1u1_state_like
from .abelian_charges import charge_tuple, check_native_charges


def build_u1_t3ns(*, norb, nb, virtual_bond_dimension, bath_bond_dimension,
                  total_particles=None, seed=11, symmetry=None, total_particles_by_spin=None):
    """Build the existing three-arm topology; chi is total over all sectors.

    symmetry='u1' retains the old behavior. symmetry='u1xu1' fixes both spin
    populations; with an even total N its default is (N/2,N/2). For odd N,
    provide total_particles_by_spin explicitly. Providing that pair alone
    selects u1xu1. No existing state is relabelled or projected.
    """
    if norb != 3 or nb < 1:
        raise ValueError('This topology requires norb=3 and nb>=1.')
    mode = symmetry or ('u1xu1' if total_particles_by_spin is not None else 'u1')
    if mode not in ('u1', 'u1xu1'):
        raise ValueError('symmetry must be u1 or u1xu1.')
    if mode == 'u1' and total_particles_by_spin is not None:
        raise ValueError('Spin populations require symmetry=u1xu1.')
    modes = 2*norb*(nb+1)
    number = modes//2 if total_particles is None else _integer(total_particles, 'total_particles')
    pair = None
    if mode == 'u1xu1':
        if total_particles_by_spin is None:
            if number % 2:
                raise ValueError('For odd N, specify total_particles_by_spin explicitly.')
            pair = number//2, number//2
        else:
            pair = charge_tuple(total_particles_by_spin)
            if total_particles is not None and sum(pair) != number:
                raise ValueError('total_particles and total_particles_by_spin disagree.')
            number = sum(pair)
    children = {'node1': [f'Impurity_{m}_up' for m in range(1, 4)]}
    for m in range(1, 4):
        branch = f'node{m+1}'
        children[f'Impurity_{m}_up'] = [branch]
        children[branch] = [f'Impurity_{m}_down', f'Bath_{m}_up_0']
        children[f'Impurity_{m}_down'] = [f'Bath_{m}_down_0']
        for spin in ('up', 'down'):
            for k in range(nb):
                children[f'Bath_{m}_{spin}_{k}'] = [f'Bath_{m}_{spin}_{k+1}'] if k+1 < nb else []
    template = TreeTensorNetworkState(backend='numpy')

    def insert(identifier, parent=None, parent_axis=None):
        d = 1 if identifier.startswith('node') else 2
        shape = (1,)*(len(children[identifier]) + int(parent is not None)) + (d,)
        data = np.zeros(shape, dtype=complex)
        node = Node(identifier=identifier)
        if parent is None:
            template.add_root(node, data)
        else:
            template.add_child_to_parent(node, data, 0, parent, parent_axis)
        for axis, child in enumerate(children[identifier], start=int(parent is not None)):
            insert(child, identifier, axis)
    insert('node1')
    caps = {i: bath_bond_dimension if i.startswith('Bath_') and int(i.rsplit('_', 1)[1]) > 0 else virtual_bond_dimension
            for i in template.nodes if i != template.root_id}
    if mode == 'u1':
        state = random_u1_state_like(template, total_particles=number, max_bond_dim=caps,
                                    seed=seed, tree_factory=TreeTensorNetworkState)
    else:
        state = random_u1u1_state_like(template, total_particles_by_spin=pair,
                                      max_bond_dim=caps, seed=seed, tree_factory=TreeTensorNetworkState)
    check_u1_state(state)
    return state


def check_native_u1(state, operator=None):
    """Compatibility API: return total N for either symmetry, validate both charges."""
    return sum(check_native_charges(state, operator))
