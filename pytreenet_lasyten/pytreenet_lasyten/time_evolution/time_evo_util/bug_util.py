
from __future__ import annotations
from copy import deepcopy
from typing import Tuple

from numpy import ndarray
import numpy as np

from ...core.node import Node
from ...core.leg_specification import LegSpecification
from ...util.tensor_splitting import tensor_qr_decomposition, SplitMode
from ...util import lasyten_backend as lb
from ...util.tensor_dispatch import is_lasyten_tensor, transpose, conjugate, tensordot
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...contractions.state_state_contraction import contract_any_nodes


def copy_state_for_update(state, copy_ids):
    """Copy a local working tree without reading every lazy TensorDict entry.

    Copy all node metadata so a lazy leg permutation on the working tree
    cannot invalidate its source. Native coefficient blocks remain shared
    except on the selected nodes. Operations on shared tensors must remain
    out of place, as in TreeTensorNetwork.deepcopy_parts.
    """
    tensors = state._tensors
    raw = getattr(tensors, "data", None)
    if not isinstance(raw, dict) or not hasattr(tensors, "__dict__"):
        return state.deepcopy_parts(copy_ids)
    cls = type(state)
    result = cls.__new__(cls)
    result.__dict__.update(state.__dict__)
    from ...util.bug_performance import copy_nodes
    result._nodes = copy_nodes(state._nodes)
    result._tensors = type(tensors).__new__(type(tensors))
    result._tensors.__dict__.update(tensors.__dict__)
    result._tensors.data = raw.copy()
    result._tensors.nodes = result._nodes
    for key in copy_ids:
        result._tensors.data[key] = deepcopy(raw[key])
    return result


def project_child_basis_changes(node, tensor, basis_change_cache):
    """Project child bonds without inserting a charged centre into a tree.

    Each bridge has axes (old child bond, new child bond). Restore the
    contracted axis after each contraction so parent, child and physical
    axes retain the node's order. The external parent space is untouched.
    """
    for child_id in node.children:
        axis = node.neighbour_index(child_id)
        bridge = basis_change_cache.get_entry(child_id, node.identifier)
        tensor = tensordot(tensor, bridge, axes=([axis], [0]))
        axes = list(range(tensor.ndim - 1))
        axes.insert(axis, tensor.ndim - 1)
        tensor = transpose(tensor, axes)
    return tensor


def make_last_leg_first(tensor):
    """Move the QR bond to the parent position without a NumPy tensor call."""
    axes = (tensor.ndim - 1,) + tuple(range(tensor.ndim - 1))
    return transpose(tensor, axes)


def _native_invariant_basis(node, tensor, mode=SplitMode.REDUCED):
    """Extract the row-space basis; keep a centre's hidden charge on R.

    The parent leg of a charged centre can use a different charge gauge than
    the parent's old invariant child basis. QR over (children, physical) puts
    the fixed charge on the discarded R and produces compatible invariant Q
    factors. Their ranges can then be direct-summed sector by sector.
    """
    array = lb.as_lasyten(tensor)
    ct = lb._lasyten()
    if isinstance(array.native, ct.ChargedTensor):
        if array.native.charge_leg.dim != 1:
            raise NotImplementedError(
                "Native BUG currently requires a one-dimensional fixed charge leg.")
    elif not isinstance(array.native, ct.SymmetricTensor):
        raise NotImplementedError(
            "Native BUG bases require SymmetricTensor or fixed-charge ChargedTensor inputs.")
    q_legs, r_legs = new_basis_tensor_qr_legs(node)
    q, _ = lb.tensor_qr_decomposition(array, q_legs, r_legs, mode=mode.value)
    q = lb.as_lasyten(q)
    if not isinstance(q.native, ct.SymmetricTensor):
        raise RuntimeError(
            "LASyTen QR must leave Q invariant and carry the hidden charge on R.")
    # QR returns a fresh factor. Drop synthetic labels before later conjugates
    # and direct sums, without relabeling either caller-owned input tensor.
    q.native.set_labels([None] * q.ndim)
    return make_last_leg_first(q)


def _native_augmented_basis(node, old_tensor, updated_tensor):
    """An invariant basis containing both old and evolved local ranges."""
    old = _native_invariant_basis(node, old_tensor)
    updated = _native_invariant_basis(node, updated_tensor)
    parent = node.parent_leg
    other_axes = tuple(i for i in range(old.ndim) if i != parent)
    if old.ndim != updated.ndim or any(old.legs[i] != updated.legs[i] for i in other_axes):
        raise ValueError(
            "BUG basis tensors must have identical child/physical spaces. "
            "Project the old child legs into their new bases before augmentation.")
    combined = concat_along_parent_leg(node, old, updated)
    return _native_invariant_basis(node, combined)

def basis_change_tensor_id(node_id: str) -> str:
    """
    The identifier for a basis change tensor.

    """
    return node_id + "_basis_change_tensor"

def reverse_basis_change_tensor_id(node_id: str) -> str:
    """
    Returns the original node identifier from a basis change tensor identifier.

    """
    if node_id.endswith("_basis_change_tensor"):
        return node_id[:-len("_basis_change_tensor")]
    else:
        return node_id

def concat_along_parent_leg(node: Node,
                            old_tensor: ndarray,
                            updated_tensor: ndarray) -> ndarray:
    """
    Concatenates two tensors along the parent leg.

    The old tensor will take the first parent dimension and the updated tensor
    the second parent dimension.

    Args:
        node (GraphNode): The node for which to concatenate the tensors.
        old_tensor (ndarray): The old tensor to concatenate.
        updated_tensor (ndarray): The updated tensor to concatenate.

    Returns:
        ndarray: The concatenated tensor with doubled parent leg dimension,
            but with the same leg order as the input tensors.
    """
    old_native = is_lasyten_tensor(old_tensor)
    new_native = is_lasyten_tensor(updated_tensor)
    if old_native or new_native:
        if not (old_native and new_native):
            raise TypeError("BUG cannot concatenate a NumPy tensor with a native LASyTen tensor.")
        old, new = lb.as_lasyten(old_tensor), lb.as_lasyten(updated_tensor)
        ct = lb._lasyten()
        if not (isinstance(old.native, ct.SymmetricTensor)
                and isinstance(new.native, ct.SymmetricTensor)):
            raise ValueError(
                "Extract invariant bases with compute_new_basis_tensor before concatenating charged centres.")
        return lb.concatenate((old, new), axis=node.parent_leg)
    return np.concatenate((old_tensor, updated_tensor), axis=node.parent_leg)

def new_basis_tensor_qr_legs(node: Node) -> Tuple[Tuple[int, ...],Tuple[int, ...]]:
    """
    Returns the leg indices for the QR decomposition to obtain the new basis
    tensor.

    Args:
        node (GraphNode): The node for which to perform the QR decomposition.

    Returns:
        Tuple([Tuple[int],Tuple[int]]): (q_legs, r_legs) The legs for the QR
            decomposition.

    """
    assert not node.is_root(), "Root node has no parent leg!"
    q_legs = node.children_legs + node.open_legs
    r_legs = [node.parent_leg]
    return tuple(q_legs), tuple(r_legs)

def _compute_new_basis_tensor_qr(node: Node,
                                 combined_tensor: ndarray,
                                 mode: SplitMode = SplitMode.REDUCED) -> ndarray:
    """
    Computes the new basis tensor from a concatenated tensor.

    Args:
        node (GraphNode): The node for which to compute the new basis tensor.
        combined_tensor (ndarray): The concatenated tensor of the old tensor
            and the updated tensor. While the parent leg dimension is doubled,
            the leg order is the same as for the node.
        
    Returns:
        ndarray: The new basis tensor. Note that the leg order is incorrect.

    """
    q_legs, r_legs = new_basis_tensor_qr_legs(node)
    if is_lasyten_tensor(combined_tensor):
        new_basis_tensor, _ = lb.tensor_qr_decomposition(
            combined_tensor, q_legs, r_legs, mode=mode.value)
        return new_basis_tensor
    new_basis_tensor, _ = tensor_qr_decomposition(combined_tensor,
                                            q_legs,
                                            r_legs,
                                            mode=mode)
    return new_basis_tensor

def compute_fixed_size_new_basis_tensor(node: Node,
                                        updated_tensor: ndarray) -> ndarray:
    """
    Computes the updated basis tensor with a fixed size.

    The updated basis tensor is found by performing a QR decomposition of the
    updated tensor.

    Args:
        node (GraphNode): The node which is updated.
        updated_tensor (ndarray): The updated basis tensor obtained by time
            evolving the old tensor. The leg order is the usual.

    Returns:
        ndarray: The new basis tensor. The leg order is the usual and equal to
            the leg order of the updated tensor. Should have the same shape
            as the updated tensor.

    """
    if is_lasyten_tensor(updated_tensor):
        basis = _native_invariant_basis(node, updated_tensor, mode=SplitMode.KEEP)
        if basis.shape != updated_tensor.shape:
            raise NotImplementedError(
                "Fixed-rank native BUG requires a charge-aware QR embedding to retain this shape.")
        return basis
    # We perform a QR decomposition
    new_basis_tensor = _compute_new_basis_tensor_qr(node,
                                                    updated_tensor,
                                                    mode=SplitMode.KEEP)
    # We will have to transpose this anyway later on
    # Currently the parent leg is the last leg
    new_basis_tensor = make_last_leg_first(new_basis_tensor)
    assert new_basis_tensor.shape == updated_tensor.shape, \
        "The new basis tensor has the wrong shape!"
    return new_basis_tensor

def compute_new_basis_tensor(node: Node,
                             old_tensor: ndarray,
                             updated_tensor: ndarray) -> ndarray:
    """
    Compute the updated basis tensor.

    The updated basis tensor is found by concatinating the old and updated
    tensor and finding a basis of their combined range.
    
    Args:
        node (GraphNode): The node which is updated.
        old_tensor (ndarray): The old basis tensor. The tensor/node has to be
            the orthogonality center of the old state. The leg order is the
            usual.
        updated_tensor (ndarray): The updated basis tensor obtained by time
            evolving the old tensor. The leg order is the usual.

    Returns:
        ndarray: The new basis tensor. The leg order is the usual and equal to
            the leg order of the old and updated tensor, but with a new parent
            dimension.
    
    """
    if is_lasyten_tensor(old_tensor) or is_lasyten_tensor(updated_tensor):
        if not (is_lasyten_tensor(old_tensor) and is_lasyten_tensor(updated_tensor)):
            raise TypeError("BUG basis updates require both inputs to use the same tensor backend.")
        return _native_augmented_basis(node, old_tensor, updated_tensor)
    # We know that both tensors have the same leg order
    # We stack them along the parent axis
    combined_tensor = concat_along_parent_leg(node,
                                              old_tensor,
                                              updated_tensor)
    # We perform a QR decomposition
    new_basis_tensor = _compute_new_basis_tensor_qr(node,
                                                    combined_tensor)
    # We will have to transpose this anyway later on
    # Currently the parent leg is the last leg
    new_basis_tensor = make_last_leg_first(new_basis_tensor)
    return new_basis_tensor

def compute_basis_change_tensor(node_old: Node,
                                node_new: Node,
                                tensor_old: ndarray,
                                tensor_new: ndarray,
                                basis_change_tensor_cache: PartialTreeCachDict
                                ) -> ndarray:
    """
    Computes the basis change tensor.

    The basis change tensor is found by contracting the old and new basis
    tensors via their phyical leg and contract the virtual legs with the
    already computed basis change tensors.

    Args:
        node_old (GraphNode): The node of the old basis tensor.
        node_new (GraphNode): The node of the new basis tensor.
        tensor_old (ndarray): The old basis tensor.
        tensor_new (ndarray): The new basis tensor.
        basis_change_tensor_cache (PartialTreeCachDict): The cache for the
            basis change tensors.
    
    Returns:
        ndarray: The basis change tensor, which is a matrix. The first leg will
            have the dimension of the old basis tensor and the second leg the
            dimension of the new basis tensor.

    """
    parent_id = node_old.parent
    assert parent_id is not None, "There is no basis change for the root node!"
    basis_change_tensor = contract_any_nodes(parent_id,
                                             node_old, node_new,
                                             tensor_old, conjugate(tensor_new),
                                             basis_change_tensor_cache)
    return basis_change_tensor

def find_new_basis_replacement_leg_specs(node: Node
                                          ) -> Tuple[LegSpecification,LegSpecification]:
    """
    Find the leg specifications to replace a node by the new basis tensors.

    Args:
        node (GraphNode): The node to replace.
    
    Returns:
        Tuple[LegSpecification,LegSpecification]: (legs_basis_change,
            legs_new_basis) The leg specifications associated to the basis
            change tensor and the new basis tensor respectively.
    """
    legs_basis_change = LegSpecification(node.parent,[],[])
    leg_new_basis = LegSpecification(None,node.children,node.open_legs)
    return legs_basis_change, leg_new_basis
