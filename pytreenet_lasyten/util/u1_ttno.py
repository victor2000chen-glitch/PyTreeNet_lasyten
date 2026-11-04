"""Exact number-conserving TTNO construction using the LASyTen adapter.

Use the existing symbolic Hamiltonian (including its Jordan-Wigner strings).
SGE constructs the operator graph. Each graph vertex is labelled by particle
transfer through that edge before site tensors are converted to native U(1).
Only individual site tensors are temporarily dense during construction.
"""
from __future__ import annotations

from heapq import heappop, heappush
from math import prod

import numpy as np

from ..operators import Hamiltonian, TensorProduct
from ..ttno import TreeTensorNetworkOperator
from ..ttno.state_diagram import StateDiagram, TTNOFinder
from . import lasyten_backend as lb
from .u1_t3ns import _topology


def _local_delta(matrix, dimension, site, label):
    a = np.asarray(matrix)
    if a.shape != (dimension, dimension) or not np.all(np.isfinite(a)):
        raise ValueError(f"Invalid local operator {label!r} at {site}: expected {(dimension, dimension)}.")
    rows, columns = np.nonzero(a)
    if not len(rows):
        return None
    # Dimension-two occupation basis |0>, |1>; dimension-one dummy charge 0.
    changes = np.unique(rows - columns)
    if len(changes) != 1:
        raise ValueError(f"Operator {label!r} at {site} mixes particle transfers {changes.tolist()}; split it into definite-charge symbols.")
    return int(changes[0])


def _prepare(hamiltonian, reference, dimensions):
    conversion = {label: np.asarray(matrix, dtype=complex)
                  for label, matrix in hamiltonian.conversion_dictionary.items()}
    coefficients = dict(hamiltonian.coeffs_mapping)
    for d in set(dimensions.values()):
        label = f"I{d}"
        if label in conversion and not np.array_equal(conversion[label], np.eye(d)):
            raise ValueError(f"{label} must be the identity used for Hamiltonian padding.")
        conversion[label] = np.eye(d, dtype=complex)
    terms, deltas = [], {}
    for fraction, coefficient, operators in hamiltonian.terms:
        value = float(fraction) * complex(coefficients[coefficient])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite coefficient {coefficient!r}.")
        if value == 0:
            continue
        total = 0
        zero = False
        for site, label in operators.items():
            if site not in dimensions:
                raise ValueError(f"Hamiltonian site {site!r} is absent from the reference tree.")
            if not isinstance(label, str):
                raise TypeError("This constructor expects symbolic local operators in Hamiltonian.terms.")
            key = site, label
            if key not in deltas:
                deltas[key] = _local_delta(conversion[label], dimensions[site], site, label)
            if deltas[key] is None:
                zero = True
                break
            total += deltas[key]
        if zero:
            continue
        if total:
            raise ValueError(f"Hamiltonian term {dict(operators)} changes total particle number by {total}.")
        terms.append((fraction, coefficient, TensorProduct(operators)))
    clean = Hamiltonian(terms, conversion_dictionary=conversion, coeffs_mapping=coefficients)
    padded = clean.pad_with_identities(reference)
    for _, _, operators in padded.terms:
        for site, label in operators.items():
            deltas.setdefault((site, label), 0)  # newly inserted identities
    return padded, deltas


def _insert(reference, tensors, factory):
    result = factory()
    rank = {i: k for k, i in enumerate(reference.nodes)}
    ready = [(rank[reference.root_id], reference.root_id)]
    while ready:
        _, i = heappop(ready)
        node = reference.nodes[i]
        if i == reference.root_id:
            result.add_root(i, tensors.pop(i))
        else:
            siblings = reference.nodes[node.parent].children
            position = siblings.index(i)
            parent_axis = int(node.parent != reference.root_id) + position
            result.add_child_to_parent(i, tensors.pop(i), 0, node.parent, parent_axis)
            if position + 1 < len(siblings):
                sibling = siblings[position + 1]
                heappush(ready, (rank[sibling], sibling))
        if node.children:
            heappush(ready, (rank[node.children[0]], node.children[0]))
    return result


def u1_ttno_from_hamiltonian(hamiltonian, reference_tree, *,
                              method=TTNOFinder.SGE, ttno_factory=None,
                              max_local_elements=5_000_000):
    """Preserve a symbolic Hamiltonian exactly in a native U(1) TTNO.

    The reference is a TTNS with one occupation leg per node, dimension two
    (charges 0,1) or one (charge 0). Its state values and virtual spaces are
    ignored. The returned operator acts on all total-N sectors; it has total
    charge zero and does not inherit the state's fixed particle number.

    Local axes are (parent if present, children..., physical output, input).
    An operator bond carries Delta N in the descendant subtree. Both signs
    are allowed. Output uses the occupation space P; input uses P.dual.

    SGE is accepted only if its virtual channels have definite charge. A
    compression that mixes charges raises rather than silently projecting.
    TTNOFinder.BASE is an exact, uncompressed alternative for small examples.
    The graph and the input Hamiltonian are not modified.
    """
    from .abelian_charges import is_pair_state
    if is_pair_state(reference_tree):
        from .u1u1_ttno import u1u1_ttno_from_hamiltonian
        return u1u1_ttno_from_hamiltonian(hamiltonian, reference_tree, method=method,
            ttno_factory=ttno_factory, max_local_elements=max_local_elements)
    if method == TTNOFinder.TREE:
        raise ValueError("TREE does not handle general coefficients; use SGE or BASE.")
    order, dimensions = _topology(reference_tree)
    h, deltas = _prepare(hamiltonian, reference_tree, dimensions)
    factory = ttno_factory or TreeTensorNetworkOperator
    root = reference_tree.root_id
    physical = {1: lb.u1_space([0]), 2: lb.u1_space([0, 1])}
    bond_charges = {}
    tensors = {}
    if not h.terms:
        zero = lb.u1_space([0])
        for i in order:
            node = reference_tree.nodes[i]
            virtual = ([] if i == root else [zero.dual]) + [zero] * len(node.children)
            d = dimensions[i]
            data = np.eye(d, dtype=complex).reshape((1,) * len(virtual) + (d, d))
            if i == root:
                data = data * 0
            tensors[i] = lb.from_numpy(data, spaces=virtual + [physical[d], physical[d].dual])
            if i != root:
                bond_charges[i] = (0,)
    else:
        diagram = StateDiagram.from_hamiltonian(h, reference_tree, method)
        diagram.set_all_vertex_indices()
        vertex_charge = {}
        for i in reversed(order):
            node = reference_tree.nodes[i]
            for edge in diagram.hyperedge_colls[i].contained_hyperedges:
                charge = deltas[i, edge.label]
                charge += sum(vertex_charge[edge.find_vertex(c).identifier] for c in node.children)
                if i == root:
                    if charge:
                        raise ValueError("The compressed diagram has a nonzero root charge.")
                else:
                    vertex = edge.find_vertex(node.parent)
                    old = vertex_charge.setdefault(vertex.identifier, charge)
                    if old != charge:
                        raise ValueError(f"Compressed bond at {i} mixes particle transfers {old} and {charge}; try TTNOFinder.BASE for a small uncompressed check.")
        for i in order[1:]:
            vertices = diagram.get_vertex_coll_two_ids(reference_tree.nodes[i].parent, i).contained_vertices
            charges = [None] * len(vertices)
            for v in vertices:
                charges[v.index] = vertex_charge[v.identifier]
            if any(q is None for q in charges):
                raise RuntimeError(f"Incomplete bond charge labelling at {i}.")
            bond_charges[i] = tuple(charges)
        spaces = {i: lb.u1_space(q) for i, q in bond_charges.items()}
        for i in order:
            node = reference_tree.nodes[i]
            shape = diagram.obtain_tensor_shape(i, h.conversion_dictionary)
            if max_local_elements is not None and prod(shape) > max_local_elements:
                raise ValueError(f"Local TTNO tensor at {i} needs {prod(shape)} dense initialization entries; increase max_local_elements explicitly if intended.")
            data = np.zeros(shape, dtype=complex)
            for edge in diagram.hyperedge_colls[i].contained_hyperedges:
                coefficient = float(edge.lambda_coeff) * complex(h.coeffs_mapping[edge.gamma_coeff])
                data[edge.find_tensor_position(reference_tree)] += coefficient * h.conversion_dictionary[edge.label]
            virtual = ([] if i == root else [spaces[i].dual]) + [spaces[c] for c in node.children]
            p = physical[dimensions[i]]
            tensors[i] = lb.from_numpy(data, spaces=virtual + [p, p.dual])
    operator = _insert(reference_tree, tensors, factory)
    operator.u1_operator_info = dict(total_charge=0, bond_charge_changes=bond_charges,
                                     physical_nodes=tuple(i for i in reference_tree.nodes if dimensions[i] == 2),
                                     method=method.value)
    return operator


def _operator_topology(operator):
    order = []
    dimensions = {}
    def visit(i):
        node = operator.nodes[i]
        dims = tuple(node.open_dimensions())
        if dims not in ((1, 1), (2, 2)):
            raise ValueError(f"{i} does not have occupation output/input legs.")
        dimensions[i] = dims[0]
        order.append(i)
        for child in node.children:
            visit(child)
    visit(operator.root_id)
    return order, dimensions


def dense_ttno_for_checks(operator, *, site_order=None, max_modes=8):
    """Explicit small-system reference matrix, using native tree contractions.

    This diagnostic alone exports the many-body operator to NumPy. The normal
    TTNO constructor never builds that matrix. Dummy dimensions are removed.
    """
    order, dimensions = _operator_topology(operator)
    active = [i for i in order if dimensions[i] == 2]
    if len(active) > max_modes:
        raise ValueError(f"Dense operator check limited to {max_modes} spin orbitals; got {len(active)}.")
    if site_order is None:
        site_order = active
    if len(site_order) != len(active) or set(site_order) != set(active):
        raise ValueError("site_order must list every dimension-two site once.")
    subtrees = {}
    for i in reversed(order):
        node = operator.nodes[i]
        a = operator.tensors[i]
        labels = ([None] if i != operator.root_id else []) + list(node.children) + [(i, 'out'), (i, 'in')]
        for child in node.children:
            b, blabels = subtrees.pop(child)
            axis = labels.index(child)
            a = lb.tensordot(a, b, axes=([axis], [0]))
            labels = labels[:axis] + labels[axis + 1:] + blabels[1:]
        subtrees[i] = a, labels
    a, labels = subtrees[operator.root_id]
    labels = [label for label in labels if dimensions[label[0]] == 2]
    desired = [(i, 'out') for i in site_order] + [(i, 'in') for i in site_order]
    data = a.to_numpy().reshape((2,) * len(labels))
    data = data.transpose([labels.index(label) for label in desired])
    return data.reshape(2 ** len(active), 2 ** len(active))


def product_matrix_element_for_checks(operator, bra, ket, *, max_local_elements=5_000_000):
    """Evaluate a product-basis matrix element without a many-body matrix.

    Diagnostic: exports one local TTNO tensor at a time to NumPy. Specify
    occupations for exactly every physical (dimension-two) site. Dummy sites
    are implicitly zero. Useful even when a full operator matrix is too large.
    """
    order, dimensions = _operator_topology(operator)
    active = {i for i in order if dimensions[i] == 2}
    for occupations in (bra, ket):
        if set(occupations) != active or any(v not in (0, 1) for v in occupations.values()):
            raise ValueError("Occupations must give 0 or 1 for every physical site.")
    subtrees = {}
    for i in reversed(order):
        a = operator.tensors[i]
        if prod(a.shape) > max_local_elements:
            raise ValueError(f"Local diagnostic download at {i} exceeds max_local_elements.")
        data = a.to_numpy()[..., bra.get(i, 0), ket.get(i, 0)]
        children = operator.nodes[i].children
        for child in reversed(children):
            data = np.tensordot(data, subtrees.pop(child), axes=([-1], [0]))
        subtrees[i] = data
    return complex(np.asarray(subtrees[operator.root_id]).item())
