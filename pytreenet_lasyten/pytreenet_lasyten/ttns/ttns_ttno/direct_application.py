"""
This module implements the direct application of a TTNO to a TTNS.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import numpy as np

from ...core.node import relative_leg_permutation, NodeEnum
from ...core.tree_structure import LinearisationMode
from ..ttns import TTNS
from ...core.node import Node
from ...util import lasyten_backend as lb
from ...util.tensor_dispatch import is_lasyten_tensor, tensordot, transpose

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...ttno.ttno_class import TTNO

__all__ = ["direct"]

def direct(ttns: TTNS,
          ttno: TTNO,
          id_mapping: Callable = lambda x: x
          ) -> TTNS:
    """
    Applies a TTNO to a TTNS.

    Note, as to fit with the usual matrix product convention, the physical
    index of the state tensors will be contracted with the last half of the
    physical legs of the operator tensors.

    Args:
        ttns (TTNS): The TTNS to which the TTNO is applied.
        ttno (TTNO): The TTNO to be applied.
        id_mapping (Callable): A function that maps the identifiers of the
            TTNS to the identifiers of the TTNO. Defaults to the identity.

    Returns:
        TTNS: The resulting TTNS after applying the TTNO. The identifiers are
            the same as those of the input TTNS.
            Note, this is a new TTNS and the input TTNS is not modified.
    """
    # Order in which a parent will always be before its children
    order = ttns.linearise(LinearisationMode.PARENTS_FIRST)
    new_ttns = TTNS()
    # Root is special
    _contract_root(ttns, ttno, new_ttns, id_mapping)
    for node_id in order[1:]:
        state_node, state_tensor = ttns[node_id]
        op_node, op_tensor = ttno[id_mapping(node_id)]
        parent_id = state_node.parent
        incoming_pipe = None
        if is_lasyten_tensor(state_tensor) and not lb._no_symmetry(lb.as_lasyten(state_tensor)):
            _, parent_tensor = new_ttns[parent_id]
            parent_axis = ttns.nodes[parent_id].neighbour_index(node_id)
            incoming_pipe = parent_tensor.legs[parent_axis].dual
        new_tensor = _node_contraction(state_node,
                                      state_tensor,
                                      op_node,
                                      op_tensor,
                                      id_mapping,
                                      incoming_pipe=incoming_pipe)
        num_neighs = new_ttns.nodes[parent_id].nneighbours()
        new_ttns.add_child_to_parent(Node(identifier=node_id),
                                     new_tensor,
                                     0,
                                     parent_id,
                                     num_neighs)
    return new_ttns

def _contract_root(ttns: TTNS,
                   ttno: TTNO,
                   new_ttns: TTNS,
                   id_mapping: Callable
                   ) -> None:
    """
    Contracts the root node of the TTNS with the root node of the TTNO and
    adds it to the new TTNS.

    Args:
        ttns (TTNS): The TTNS containing the root node.
        ttno (TTNO): The TTNO containing the root node.
        new_ttns (TTNS): The new TTNS to which the contracted root node
            is added.
        id_mapping (Callable): A function that maps the identifiers of the
            TTNS to the identifiers of the TTNO.
    """
    root_node, root_tensor = ttns.root
    op_root_node, op_root_tensor = ttno[id_mapping(root_node.identifier)]
    new_root_tensor = _node_contraction(root_node,
                                       root_tensor,
                                       op_root_node,
                                       op_root_tensor,
                                       id_mapping)
    root_id = root_node.identifier
    new_ttns.add_root(Node(identifier=root_id),
                      new_root_tensor)

def _node_contraction(state_node: Node,
                      state_tensor: npt.NDArray,
                      op_node: Node,
                      op_tensor: npt.NDArray,
                      id_mapping: Callable,
                      *, incoming_pipe=None
                      ) -> npt.NDArray:
    """
    Contracts a state node with an operator node.

    Args:
        state_node (Node): The state node.
        state_tensor (npt.NDArray): The tensor of the state node.
        op_node (Node): The operator node.
        op_tensor (npt.NDArray): The tensor of the operator node.
        id_mapping (Callable): A function that maps the identifiers of the
            state node to the identifiers of the operator node.
    """
    state_open = state_node.open_legs
    op_open = op_node.open_legs
    # Only the second half of the operator open legs are to be contracted
    op_contr_open = op_open[len(op_open) // 2:]
    if len(state_open) != len(op_contr_open):
        errstr = "The number of open legs to be contracted must be the same!"
        errstr += (f" State node {state_node} has {len(state_open)} open legs, "
                   f"operator node {op_node} has {len(op_contr_open)} open legs.")
        raise ValueError(errstr)
    tensor_perm = relative_leg_permutation(op_node,
                                           state_node,
                                           modify_function=id_mapping,
                                           use_open_legs=NodeEnum.OLD)
    native = is_lasyten_tensor(state_tensor) or is_lasyten_tensor(op_tensor)
    if native and not (is_lasyten_tensor(state_tensor) and is_lasyten_tensor(op_tensor)):
        raise TypeError("Direct application requires both state and operator tensors to use the same tensor backend.")
    op_tensor_perm = transpose(op_tensor, axes=tensor_perm)
    new_tensor = tensordot(state_tensor, op_tensor_perm,
                              axes=(state_open, op_contr_open))
    # Now we need to combine the virtual legs of the new tensor.
    num_neighs = state_node.nneighbours()
    perm = []
    shape = []
    for i in range(num_neighs):
        # A dual pipe reverses its component spaces. Reuse the parent's
        # actual pipe, with reversed incoming factors, so both ends share
        # the same fused basis even when sectors have degeneracies.
        pair = (i + num_neighs, i) if i == 0 and incoming_pipe is not None else (i, i + num_neighs)
        perm.extend(pair)
        # The legs coming from the operator are after the state legs
        # The open legs of the operator are at the end of its tensor
        shape.append(new_tensor.shape[i]*new_tensor.shape[i + num_neighs])
    open_legs = list(range(2 * num_neighs, new_tensor.ndim))
    perm.extend(open_legs)
    shape.extend(op_node.open_dimensions()[len(op_node.open_legs) // 2:])
    new_tensor = transpose(new_tensor, axes=perm)
    if native:
        if num_neighs:
            groups = [tuple(range(2*i, 2*i+2)) for i in range(num_neighs)]
            pipes = [incoming_pipe] + [None] * (num_neighs - 1)
            dualities = [incoming_pipe.is_dual if incoming_pipe is not None else False] + [False] * (num_neighs - 1)
            new_tensor = lb.fuse_legs(new_tensor, *groups,
                                      pipe_dualities=dualities, pipes=pipes)
            # PyTreeNet addresses legs by Node axes. LASyTen can assign the
            # same synthetic label to multiple groups of unnamed legs;
            # clear those labels on this fresh result before later daggers.
            new_tensor.native.set_labels([None] * new_tensor.ndim)
    else:
        new_tensor = np.reshape(new_tensor, shape)
    return new_tensor
