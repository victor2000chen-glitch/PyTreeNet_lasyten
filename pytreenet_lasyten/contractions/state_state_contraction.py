"""
This module provides the function to contract two TreeTensorNetworkStates.

The main function is contract_two_ttns(ttn1, ttn2), which contracts two
TreeTensorNetworkStates with the same underlying tree structure.
Therefore the contraction is basically the scalar product of the first state
with the complex conjugate of the second state. <TTN2*|TTN1>

Using explicit imports from this module provides utility functions to contract
subtrees and leaf nodes of two states.

Native LASyTen fixed-U(1) overlaps close the one-dimensional charge legs of
a ket/bra pair explicitly. This avoids LASyTen's unimplemented general
ChargedTensor x ChargedTensor state_tensor_product. The charge carriers may
be at the root, a leaf, or different nodes in the two states. No dense tensor
conversion, canonicalization, or state mutation is used. Nonmatching charge
sectors in a partial environment and charge legs of dimension > 1 are outside
this specialization. NumPy contractions preserve the original behavior.

Requires the existing tensor_dispatch.py and lasyten_backend.py ports.
build_full_subtree_cache retains its separate LocalContraction implementation;
the fix below concerns contract_two_ttns and the explicit contraction helpers.
"""
from __future__ import annotations
from typing import Union, Callable, TYPE_CHECKING

import numpy as np

from .tree_cach_dict import PartialTreeCachDict
from .contraction_util import (contract_all_but_one_neighbour_block_to_ket,
                               contract_all_neighbour_blocks_to_ket,
                               get_equivalent_legs)
from .local_contr import LocalContraction
from ..util import lasyten_backend as lb
from ..util.tensor_dispatch import (
    tensordot as _dispatch_tensordot, conjugate, is_lasyten_tensor,
)

if TYPE_CHECKING:
    from ..core.node import Node
    from ..ttns.ttns import TreeTensorNetworkState

__all__ = ['contract_two_ttns']


def tensordot(a, b, axes=2):
    """Bilinear contraction, closing a fixed-charge ket/bra pair natively.

    Conjugation belongs to the caller. In particular, charged_state is not
    conjugated again here: an already conjugated bra includes that factor.
    The invariant parts have an extra charge axis after all public axes when
    wrapped by LasyTenArray. Contract it together with the requested axes.
    """
    if not (is_lasyten_tensor(a) and is_lasyten_tensor(b)):
        return _dispatch_tensordot(a, b, axes=axes)
    a, b = lb.as_lasyten(a), lb.as_lasyten(b)
    ct = lb._lasyten()
    if not (isinstance(a.native, ct.ChargedTensor)
            and isinstance(b.native, ct.ChargedTensor)):
        return _dispatch_tensordot(a, b, axes=axes)
    if isinstance(axes, (int, np.integer)):
        if axes < 0 or axes > min(a.ndim, b.ndim):
            raise ValueError("Invalid contraction rank.")
        ax_a = tuple(range(a.ndim - axes, a.ndim))
        ax_b = tuple(range(axes))
    else:
        if len(axes) != 2:
            raise ValueError("axes must be an integer or a pair of axis sequences.")
        ax_a, ax_b = lb._axes(axes[0], a.ndim), lb._axes(axes[1], b.ndim)
    if len(ax_a) != len(ax_b):
        raise ValueError("Contraction axis counts differ.")
    if len(ax_a) == a.ndim == b.ndim:
        # The existing backend already handles complete charged contractions
        # via native inner(), including orthogonality of different sectors.
        return _dispatch_tensordot(a, b, axes=(ax_a, ax_b))
    if a.native.charge_leg.dim != 1 or b.native.charge_leg.dim != 1:
        raise NotImplementedError(
            "Partial state-state contractions require one-dimensional fixed U1 charge legs.")
    if a.native.charge_leg != b.native.charge_leg.dual:
        raise NotImplementedError(
            "Partial state-state contractions require matching ket/bra charge sectors. "
            "The generic charged tensor product is not implemented by LASyTen.")
    invariant_a = lb.LasyTenArray(a.native.invariant_part)
    invariant_b = lb.LasyTenArray(b.native.invariant_part)
    result = _dispatch_tensordot(
        invariant_a, invariant_b,
        axes=(ax_a + (a.ndim,), ax_b + (b.ndim,)),
    )
    # A one-dimensional charge state contributes one scalar. Only that
    # scalar is transferred; tensor blocks and spaces stay native.
    factor = lb._scalar(a.native.backend.block_backend.inner(
        a.native.charged_state, b.native.charged_state, False,
    ))
    result = result * factor
    if isinstance(result, lb.LasyTenArray):
        result.native.set_labels([None] * result.ndim)
    return result


def _contract_overlap_neighbours(ket_tensor, ket_node, dictionary, ignored=None):
    """Absorb overlap environments through the local charge-aware tensordot.

    Each environment has public axes (ket bond, bra bond). A single unmatched
    charge can be hidden in a ChargedTensor environment until it meets the
    opposite charge from the other state. Calling contraction_util directly
    would bypass this module's fixed-charge closure when those charges meet
    during environment absorption instead of during the final bra contraction.
    Output axis order is identical to the existing contraction_util helpers.
    """
    result = ket_tensor
    ignored_index = None if ignored is None else ket_node.neighbour_index(ignored)
    for neighbour_id in ket_node.neighbouring_nodes():
        if neighbour_id == ignored:
            continue
        if ignored_index is None:
            axis = 0
        else:
            neighbour_index = ket_node.neighbour_index(neighbour_id)
            axis = int(ignored_index < neighbour_index)
        environment = dictionary.get_entry(neighbour_id, ket_node.identifier)
        result = tensordot(result, environment, axes=([axis], [0]))
    return result


def contract_two_ttns(ttn1: TreeTensorNetworkState,
                      ttn2: TreeTensorNetworkState,
                      id_trafo: Union[None,Callable] = None,
                      *, conjugate_second: bool = False
                      ) -> complex:
    """
    Contracts two TreeTensorNetworks.

    Args:
        ttn1 (TreeTensorNetwork): The first TreeTensorNetwork.
        ttn2 (TreeTensorNetwork): The second TreeTensorNetwork.
        id_trafo (Union[None,Callable]): A function to transform the node
            identifiers of ttn1 into the node identifiers of ttn2. If None,
            it is assumed that the identifiers are the same.
            Defaults to None.
        conjugate_second (bool): Conjugate the second tensor at each site
            during contraction, without copying the entire second network.
            Defaults to False for compatibility with preconjugated bras.

    Returns:
        complex: Bilinear contraction of both networks. If
            ``conjugate_second=True``, returns <TTN2|TTN1>. The default
            preserves the original API, where the bra is already conjugated.
    """
    dictionary = PartialTreeCachDict()
    computation_order = ttn1.linearise() # Getting a linear list of all identifiers
    for node_id in computation_order[:-1]: # The last one is the root node
        node = ttn1.nodes[node_id]
        parent_id = node.parent
        # Due to the linearisation the children should already be contracted.
        block = contract_any(node_id, parent_id,
                             ttn1, ttn2,
                             dictionary,
                             id_trafo=id_trafo,
                             state2_conj=conjugate_second)
        dictionary.add_entry(node_id,parent_id,block)
        # The children contraction results are not needed anymore.
        children = node.children
        for child_id in children:
            dictionary.delete_entry(child_id,node_id)
    # Now everything remaining is contracted into the root tensor.
    return complex(contract_node_with_environment(ttn1.root_id,
                                                  ttn1, ttn2,
                                                  dictionary,
                                                  id_trafo=id_trafo,
                                                  conjugate_second=conjugate_second))

def contract_node_with_environment(node_id: str,
                                   state1: TreeTensorNetworkState,
                                   state2: TreeTensorNetworkState,
                                   dictionary: PartialTreeCachDict,
                                   id_trafo: Union[None,Callable] = None,
                                   *, conjugate_second: bool = False
                                   ) -> np.ndarray:
    """
    Contracts a node with its environment.
     
    It is assumed that all subtrees starting from this node are already
    contracted.

    Args:
        node_id (str): The identifier of the node.
        state1 (TreeTensorNetworkState): The first TTN state.
        state2 (TreeTensorNetworkState): The second TTN state.
        dictionary (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
        id_trafo (Union[None,Callable]): A function to transform the node
            identifiers of state1 into the node identifiers of state2. If
            None, it is assumed that the identifiers are the same.
            Defaults to None.
    
    Returns:
        np.ndarray: The resulting tensor. A and B are the tensors in state1 and
            state2, respectively, corresponding to the node with the identifier
            node_id. C aer the tensors in the dictionary corresponding to the
            subtrees going away from the node::

             ______      _____      ______
            |      |____|     |____|      |
            |      |    |  B  |    |      |
            |      |    |_____|    |      |
            |      |       |       |      |
            |  C1  |       |       |  C2  |
            |      |     __|__     |      |
            |      |____|     |____|      |
            |      |    |  A  |    |      |
            |______|    |_____|    |______|
    
    """
    ket_node, ket_tensor = state1[node_id]
    if id_trafo is None:
        bra_node, bra_tensor = state2[node_id]
    else:
        bra_node, bra_tensor = state2[id_trafo(node_id)]
    if conjugate_second:
        bra_tensor = conjugate(bra_tensor)
    return contract_node_with_environment_nodes(ket_node, ket_tensor,
                                                bra_node, bra_tensor,
                                                dictionary,
                                                id_trafo=id_trafo)

def contract_node_with_environment_nodes(ket_node: Node,
                                         ket_tensor: np.ndarray,
                                         bra_node: Node,
                                         bra_tensor: np.ndarray,
                                         dictionary: PartialTreeCachDict,
                                         id_trafo: Union[None,Callable] = None
                                         ) -> np.ndarray:
    """
    Contracts a node with its environment.
     
    It is assumed that all subtrees starting from this node are already
    contracted.

    Args:
        ket_node (Node): The ket node.
        ket_tensor (np.ndarray): The ket tensor.
        bra_node (Node): The bra node.
        bra_tensor (np.ndarray): The bra tensor.
        dictionary (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
        id_trafo (Union[None,Callable]): A function to transform the node
            identifiers of the ket node into the node identifiers of the bra
            node. If None, it is assumed that the identifiers are the same.
            Defaults to None.
    
    Returns:
        np.ndarray: The resulting tensor. A and B are the tensors in state1 and
            state2, respectively, corresponding to the node with the identifier
            node_id. C aer the tensors in the dictionary corresponding to the
            subtrees going away from the node::

             ______      _____      ______
            |      |____|     |____|      |
            |      |    |  B  |    |      |
            |      |    |_____|    |      |
            |      |       |       |      |
            |  C1  |       |       |  C2  |
            |      |     __|__     |      |
            |      |____|     |____|      |
            |      |    |  A  |    |      |
            |______|    |_____|    |______|
    
    """
    ketblock_tensor = _contract_overlap_neighbours(ket_tensor, ket_node, dictionary)
    return contract_bra_to_ket_and_blocks(bra_tensor, ketblock_tensor,
                                          bra_node, ket_node,
                                          id_trafo=id_trafo)

def contract_any(node_id: str, next_node_id: str,
                 state1: TreeTensorNetworkState,
                 state2: TreeTensorNetworkState,
                 dictionary: PartialTreeCachDict,
                 state2_conj: bool = False,
                 id_trafo: Union[Callable,None] = None
                 ) -> np.ndarray:
    """
    Contracts any node.

    More specifically, it contracts the tensors of each state corresponding to
    the specified node with all but one of the subtrees attached to the node.
    The remaining open legs of the resulting tensor point to the uncontracted
    next node.

    Args:
        node_id (str): The identifier of the node.
        next_node_id (str): The identifier of the node to which the remaining
            virtual legs point.
        state1 (TreeTensorNetworkState): The first TTN state.
        state2 (TreeTensorNetworkState): The second TTN state.
        dictionary (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
        state2_conj (bool): Whether to use the complex conjugate of the tensors
            in state2. Defaults to False.
        id_trafo (Union[Callable,None], optional): A function to transform the
            node identifiers of state1 into the node identifiers of stat2. If
            None, it is assumed that the identifiers are the same. Defaults to
            None.

    Returns:
        np.ndarray: The resulting tensor. 

                 _____      ______
            ____|     |____|      |
            1   |  T2 |    |      |
                |_____|    |      |
                   |       |      |
                   |       |  C1  |
                 __|__     |      |
            ____|     |____|      |
            0   |  T1 |    |      |
                |_____|    |______|

    """
    node1, tensor1 = state1[node_id]
    if id_trafo is None:
        node2, tensor2 = state2[node_id]
    else:
        node2, tensor2 = state2[id_trafo(node_id)]
    if state2_conj:
        tensor2 = conjugate(tensor2)
    return contract_any_nodes(next_node_id, node1, node2,
                              tensor1, tensor2, dictionary,
                              id_trafo=id_trafo)

def contract_any_nodes(next_node_id: str,
                       node1: Node, node2: Node,
                       tensor1: np.ndarray, tensor2: np.ndarray,
                       dictionary: PartialTreeCachDict,
                       id_trafo: Union[Callable,None] = None) -> np.ndarray:
    """
    Contracts any two nodes using the given tensors.

    More specifically, it contracts the given tensors with all but one of the
    subtrees attached to the nodes. The remaining open legs of the resulting
    tensor point to the uncontracted next node.

    Note that the two nodes have to have the same connectivity, i.e. the same
    identifiers as neighbours.

    Args:
        next_node_id (str): The identifier of the node to which the remaining
            virtual legs point.
        node1 (Node): The first node.
        node2 (Node): The second node.
        tensor1 (np.ndarray): The tensor corresponding to node1.
        tensor2 (np.ndarray): The tensor corresponding to node2.
        dictionary (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
        id_trafo (Union[Callable,None], optional): A function to transform the
            node identifiers of node1 into the node identifiers of node2. If
            None, it is assumed that the identifiers are the same. Defaults to
            None.
    
    Returns:
        np.ndarray: The resulting tensor. 

                 _____      ______
            ____|     |____|      |
            1   |  T2 |    |      |
                |_____|    |      |
                   |       |      |
                   |       |  C1  |
                 __|__     |      |
            ____|     |____|      |
            0   |  T1 |    |      |
                |_____|    |______|
    """
    if node1.is_leaf():
        return contract_leafs(node1, node2, tensor1, tensor2)
    return contract_subtrees_using_dictionary(next_node_id,
                                              node1, node2,
                                              tensor1, tensor2,
                                              dictionary,
                                              id_trafo=id_trafo)

def contract_leafs(node1: Node, node2: Node,
                   tensor1: np.ndarray, tensor2: np.ndarray
                   ) -> np.ndarray:
    """
    Contracts tensors associated with two leaf nodes.

    If the current subtree starts at a leaf node, only the tensors correspoding
    to this leaf in the two states must be contracted with each other.

    Args:
        node1 (Node): The first node.
        node2 (Node): The second node.
        tensor1 (np.ndarray): The tensor corresponding to node1.
        tensor2 (np.ndarray): The tensor corresponding to node2.

    Returns:
        np.ndarray: The tensor resulting from the contraction::

                     _____
                ____|     |
                    |  T2 |
                    |_____|
                       |
                       |
                     __|__
                ____|     |
                    |  T1 |
                    |_____|
    

    """
    assert node1.is_leaf() and node2.is_leaf()
    errstr = "The leaf nodes must have exactly one open leg."
    assert len(node1.open_legs) == 1 and len(node2.open_legs) == 1, errstr
    axes = (node1.open_legs[0], node2.open_legs[0])
    tensor = tensordot(tensor1, tensor2,
                            axes=axes)
    return tensor

def contract_subtrees_using_dictionary(next_node_id: str,
                                       node1: Node, node2: Node,
                                       tensor1: np.ndarray,
                                       tensor2: np.ndarray,
                                       dictionary: PartialTreeCachDict,
                                       id_trafo: Union[Callable,None] = None
                                       ) -> np.ndarray:
    """
    Contracts a node with all but one of the subtrees attached to it.

    The tensors corresponding to the nodes are contracted with
    each other and all but one subtrees starting from this node. The other
    subtrees are assumed to already be contracted and stored in the provided
    dictionary.

    Args:
        next_node_id (str): The identifier of the node to which the remaining
            virtual legs point.
        node1 (Node): The first node.
        node2 (Node): The second node.
        tensor1 (np.ndarray): The tensor corresponding to node1.
        tensor2 (np.ndarray): The tensor corresponding to node2.
        dictionary (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
        id_trafo (Union[Callable,None], optional): A function to transform the
            node identifiers of node1 into the node identifiers of node2. If
            None, it is assumed that the identifiers are the same. Defaults to
            None.
        
        Returns:
            np.ndarray: The resulting tensor. For example, if the nodes have
                two neighbours::

                         _____      ______
                    ____|     |____|      |
                        |  T2 |    |      |
                        |_____|    |      |
                           |       |      |
                           |       |  C   |
                         __|__     |      |
                    ____|     |____|      |
                        |  T1 |    |      |
                        |_____|    |______|
        
    """
    ket_node = node1
    ket_tensor = tensor1
    ketblock_tensor = _contract_overlap_neighbours(
        ket_tensor, ket_node, dictionary, ignored=next_node_id)
    bra_node = node2
    bra_tensor = tensor2
    return contract_bra_to_ket_and_blocks_ignore_one_leg(bra_tensor,
                                                         ketblock_tensor,
                                                         bra_node,
                                                         ket_node,
                                                         next_node_id,
                                                         id_trafo=id_trafo)

def contract_bra_to_ket_and_blocks(bra_tensor: np.ndarray,
                                   ketblock_tensor: np.ndarray,
                                   bra_node: Node,
                                   ket_node: Node,
                                   id_trafo: Union[Callable,None] = None
                                   ) -> np.ndarray:
    """
    Contracts the bra tensor with the ket and all neighbouring blocks.

    The bra tensor is contracted with the ket tensor and all neighbouring
    subtrees, which are already contracted into the blocks.

    Args:
        bra_tensor (np.ndarray): The tensor of the bra node.
        ketblock_tensor (np.ndarray): The tensor resulting from the contraction
         of the ket node and its neighbouring blocks. Note that the physical leg
         of the ket node is now the first leg.
        bra_node (Node): The bra node.
        ket_node (Node): The ket node.
        id_trafo (Union[Callable,None], optional): A function to transform the
            node identifiers of the ket node into the node identifiers of the
            the bra node. If None, it is assumed that the identifiers are the
            same. Defaults to None.

    Returns:
        np.ndarray: The resulting tensor::
    
             ______      _____      ______
            |      |____|     |____|      |
            |      |    |  B  |    |      |
            |      |    |_____|    |      |
            |      |       |       |      |
            |  C1  |       |       |  C2  |
            |      |     __|__     |      |
            |      |____|     |____|      |
            |      |    |  A  |    |      |
            |______|    |_____|    |______|

    """
    num_neighbours = bra_node.nneighbours()
    legs_block, legs_bra = get_equivalent_legs(ket_node,
                                               bra_node,
                                               id_trafo=id_trafo)
    # All the neighbour block indices are moved up by one
    legs_block = [leg + 1 for leg in legs_block]
    # The kets physical leg is now the first leg
    legs_block.append(0)
    # The bra tensor's physical leg is the last leg
    legs_bra.append(num_neighbours)
    return tensordot(ketblock_tensor, bra_tensor,
                        axes=(legs_block, legs_bra))

def contract_bra_to_ket_and_blocks_ignore_one_leg(bra_tensor: np.ndarray,
                                                  ketblock_tensor: np.ndarray,
                                                  bra_node: Node,
                                                  ket_node: Node,
                                                  next_node_id: str,
                                                  id_trafo: Union[Callable,None] = None
                                                  ) -> np.ndarray:
    """
    Contracts the bra tensor with the ket and all but one neighbouring block.

    Args:
        bra_tensor (np.ndarray): The tensor of the bra node.
        ketblock_tensor (np.ndarray): The tensor resulting from the contraction
         of the ket node and its neighbouring blocks.
        bra_node (Node): The bra node.
        ket_node (Node): The ket node.
        next_node_id (str): The identifier of the node to which the remaining
            virtual legs point.
        id_trafo (Union[Callable,None], optional): A function to transform the
            node identifiers of the ket node into the node identifiers of the
            the bra node. If None, it is assumed that the identifiers are the
            same. Defaults to None.

    Returns:
        np.ndarray: The resulting tensor::

                         _____      ______
                    ____|     |____|      |
                        |  B  |    |      |
                        |_____|    |      |
                           |       |      |
                           |       |  C   |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |    |      |
                        |_____|    |______|
        
    """
    legs_block = []
    legs_bra = []
    next_node_index = ket_node.neighbour_index(next_node_id)
    for neighbour_id in ket_node.neighbouring_nodes():
        if neighbour_id != next_node_id:
            if id_trafo is None:
                bra_neighbour_id = neighbour_id
            else:
                bra_neighbour_id = id_trafo(neighbour_id)
            ket_index = ket_node.neighbour_index(neighbour_id)
            # 1 comes from the physical leg and the other one happens exactly
            # when the current neighbour index is above the ignored index.
            # This is how the block tensor is constructed.
            legs_block.append(ket_index + 1 + int(next_node_index > ket_index))
            bra_index = bra_node.neighbour_index(bra_neighbour_id)
            legs_bra.append(bra_index)
    # The physical leg of the ket is now leg 1
    legs_block.append(1)
    # Physical leg of the bra is the last leg
    num_neighbours = bra_node.nneighbours()
    legs_bra.append(num_neighbours)
    return tensordot(ketblock_tensor, bra_tensor,
                        axes=(legs_block, legs_bra))


def build_full_subtree_cache(ttns: TreeTensorNetworkState
                              ) -> PartialTreeCachDict:
    """
    Contracts all subtrees of the TTNS-TTNS.conj contraction.

    This means all subtrees pointing up and down the tree.

    Args:
        ttns (TTNS): The tree tensor network state.
    
    Returns:
        PartialTreeCachDict: The contracted subtrees.
    """
    cache = PartialTreeCachDict()
    lin_order = ttns.linearise() # Exclude root
    # Get subtrees upwards
    for node_id in lin_order[:-1]:
        node, tensor = ttns[node_id]
        nodes_tensors = [(node, tensor),
                         (node, tensor.conj())]
        ignored_leg = node.parent
        assert ignored_leg is not None
        local_contr = LocalContraction(nodes_tensors,
                                       cache,
                                       ignored_leg=ignored_leg)
        local_contr.contract_into_cache()
    # At this point all upwards envs are in the cache, so everything towards
    # the root.
    # Now we go back down.
    lin_order.reverse()
    for node_id in lin_order:
        node, tensor = ttns[node_id]
        if not node.is_leaf():
            nodes_tensors = [(node, tensor),
                             (node, tensor.conj())]
            for child_id in node.children:
                ignored_leg = child_id
                local_contr = LocalContraction(nodes_tensors,
                                               cache,
                                               ignored_leg=ignored_leg)
                local_contr.contract_into_cache()
    return cache
