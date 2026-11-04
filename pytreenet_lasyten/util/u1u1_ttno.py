"""Exact spin-resolved TTNO construction, including charge refinement of SGE channels."""
from itertools import product
from math import prod

import numpy as np

from ..operators import Hamiltonian, TensorProduct
from ..ttno import TreeTensorNetworkOperator
from ..ttno.state_diagram import StateDiagram, TTNOFinder
from . import lasyten_backend as lb
from .abelian_charges import add_charge, charge_space, physical_basis, symmetry_rank
from .u1_t3ns import _topology
from .u1_ttno import _insert

ZERO = (0, 0)


def _local_pair_delta(matrix, basis, site, label):
    a = np.asarray(matrix)
    if a.shape != (len(basis), len(basis)) or not np.all(np.isfinite(a)):
        raise ValueError(f'Invalid operator {label!r} at {site}.')
    rows, cols = np.nonzero(a)
    changes = {tuple(basis[r][k]-basis[c][k] for k in (0, 1)) for r, c in zip(rows, cols)}
    if not changes:
        return None
    if len(changes) != 1:
        raise ValueError(f'Operator {label!r} at {site} mixes spin-resolved transfers; split its symbol into definite-charge operators.')
    return changes.pop()


def _prepare_pair_hamiltonian(hamiltonian, reference, dimensions, physical):
    conversion = {k: np.asarray(v, dtype=complex) for k, v in hamiltonian.conversion_dictionary.items()}
    coefficients = dict(hamiltonian.coeffs_mapping)
    for d in set(dimensions.values()):
        label = f'I{d}'
        if label in conversion and not np.array_equal(conversion[label], np.eye(d)):
            raise ValueError(f'{label} must be the padding identity.')
        conversion[label] = np.eye(d, dtype=complex)
    basis = {i: physical_basis(p) for i, p in physical.items()}
    terms, deltas = [], {}
    for fraction, coefficient, operators in hamiltonian.terms:
        value = float(fraction)*complex(coefficients[coefficient])
        if not np.isfinite(value):
            raise ValueError(f'Nonfinite coefficient {coefficient!r}.')
        if value == 0:
            continue
        total, zero = ZERO, False
        for site, label in operators.items():
            if site not in dimensions or not isinstance(label, str):
                raise ValueError('Hamiltonian terms must use known sites and symbolic operators.')
            key = site, label
            if key not in deltas:
                deltas[key] = _local_pair_delta(conversion[label], basis[site], site, label)
            if deltas[key] is None:
                zero = True
                break
            total = add_charge(total, deltas[key])
        if zero:
            continue
        if total != ZERO:
            raise ValueError(f'Hamiltonian term {dict(operators)} changes (N_up,N_down) by {total}.')
        terms.append((fraction, coefficient, TensorProduct(operators)))
    padded = Hamiltonian(terms, conversion_dictionary=conversion, coeffs_mapping=coefficients).pad_with_identities(reference)
    for _, _, operators in padded.terms:
        for site, label in operators.items():
            deltas.setdefault((site, label), ZERO)
    return padded, deltas


def _expand_charge_channels(order, nodes, hyperedges, deltas):
    """Refine each compressed virtual channel into (original channel, charge).

    No original channel is merged. Each local tensor is projected onto all
    compatible charge transfers; root transfer is zero. For a conserving H
    this is an exact decomposition, including when SGE merged spin transfers.
    """
    root = order[0]
    charges, records = {}, {}
    for i in reversed(order):
        node = nodes[i]
        entries = []
        for edge in hyperedges[i]:
            child_vertices = [edge.find_vertex(c).identifier for c in node.children]
            options = [sorted(charges[v]) for v in child_vertices]
            for child_charges in product(*options):
                total = deltas[i, edge.label]
                for q in child_charges:
                    total = add_charge(total, q)
                if i == root:
                    if total != ZERO:
                        continue
                    parent_key = None
                else:
                    vertex = edge.find_vertex(node.parent).identifier
                    charges.setdefault(vertex, set()).add(total)
                    parent_key = vertex, total
                entries.append((parent_key, tuple(zip(child_vertices, child_charges)), edge))
        records[i] = entries
    if not records[root]:
        raise ValueError('No charge-conserving root channels were found.')
    # Retain only charge channels reachable from the neutral root.
    used = {i: set() for i in order[1:]}
    for i in order:
        active = records[i] if i == root else [entry for entry in records[i] if entry[0] in used[i]]
        records[i] = active
        for _, child_keys, _ in active:
            for child, key in zip(nodes[i].children, child_keys):
                used[child].add(key)
    return records, used


def u1u1_ttno_from_hamiltonian(hamiltonian, reference_tree, *, method=TTNOFinder.SGE,
                              ttno_factory=None, max_local_elements=5_000_000):
    if method == TTNOFinder.TREE:
        raise ValueError('Use SGE or BASE for general coefficients.')
    order, dimensions = _topology(reference_tree)
    root = order[0]
    physical = {i: reference_tree.tensors[i].legs[-1] for i in order}
    if any(symmetry_rank(p.symmetry) != 2 for p in physical.values()):
        raise ValueError('Every reference physical leg must use U(1) x U(1).')
    h, deltas = _prepare_pair_hamiltonian(hamiltonian, reference_tree, dimensions, physical)
    factory = ttno_factory or TreeTensorNetworkOperator
    tensors, bond_charges = {}, {}
    if not h.terms:
        neutral = charge_space([ZERO])
        for i in order:
            node = reference_tree.nodes[i]
            virtual = ([] if i == root else [neutral.dual]) + [neutral]*len(node.children)
            d, p = dimensions[i], physical[i]
            data = np.eye(d, dtype=complex).reshape((1,)*len(virtual) + (d, d))
            if i == root:
                data *= 0
            tensors[i] = lb.from_numpy(data, spaces=virtual + [p, p.dual])
            if i != root:
                bond_charges[i] = (ZERO,)
    else:
        diagram = StateDiagram.from_hamiltonian(h, reference_tree, method)
        diagram.set_all_vertex_indices()
        hyperedges = {i: diagram.hyperedge_colls[i].contained_hyperedges for i in order}
        records, used = _expand_charge_channels(order, reference_tree.nodes, hyperedges, deltas)
        positions = {}
        for i in order[1:]:
            vertices = diagram.get_vertex_coll_two_ids(reference_tree.nodes[i].parent, i).contained_vertices
            old_index = {v.identifier: v.index for v in vertices}
            keys = sorted(used[i], key=lambda key: (old_index[key[0]], key[1]))
            if not keys:
                raise ValueError(f'Empty refined TTNO bond at {i}.')
            positions[i] = {key: k for k, key in enumerate(keys)}
            bond_charges[i] = tuple(key[1] for key in keys)
        spaces = {i: charge_space(q) for i, q in bond_charges.items()}
        for i in order:
            node = reference_tree.nodes[i]
            virtual = ([] if i == root else [spaces[i].dual]) + [spaces[c] for c in node.children]
            d, p = dimensions[i], physical[i]
            shape = tuple(int(s.dim) for s in virtual) + (d, d)
            if max_local_elements is not None and prod(shape) > max_local_elements:
                raise ValueError(f'Refined TTNO tensor at {i} needs {prod(shape)} initialization entries; raise max_local_elements explicitly if intended.')
            data = np.zeros(shape, dtype=complex)
            for parent_key, child_keys, edge in records[i]:
                axes = ([] if i == root else [positions[i][parent_key]])
                axes += [positions[c][key] for c, key in zip(node.children, child_keys)]
                coefficient = float(edge.lambda_coeff)*complex(h.coeffs_mapping[edge.gamma_coeff])
                data[tuple(axes)] += coefficient*h.conversion_dictionary[edge.label]
            tensors[i] = lb.from_numpy(data, spaces=virtual + [p, p.dual])
    operator = _insert(reference_tree, tensors, factory)
    operator.u1_operator_info = dict(total_charge=ZERO, bond_charge_changes=bond_charges,
        physical_nodes=tuple(i for i in order if dimensions[i] == 2), method=method.value,
        charge_labels=('N_up', 'N_down'))
    return operator
