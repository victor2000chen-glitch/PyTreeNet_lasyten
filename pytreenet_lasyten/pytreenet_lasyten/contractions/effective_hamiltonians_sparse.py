"""
This module contains functions to find effective Hamiltonians.

The functions use cached tensors to be more efficient.

Install as pytreenet_lasyten/contractions/effective_hamiltonians_sparse.py.
Relative imports also support the original pytreenet package name.
Requires the previously adapted tensor_dispatch.py, lasyten_backend.py, and
contraction_util.py.

NumPy inputs retain the SciPy LinearOperator/vector interface. LASyTen inputs
with matrix_free=True return LasyTenEffectiveHamiltonian: its matvec and @
accept a LASyTen site tensor and return a LASyTen tensor in the state's leg
order. No implicit dense conversion or flattening is performed. A Davidson
solver using NumPy vectors needs a separate symmetry-block packing adapter.
With matrix_free=False, native inputs return a native two-leg matrix.

This module consumes environments; it does not build a SandwichCache or port
DMRG orthogonalization, environment construction, or the eigensolver.
"""

from __future__ import annotations

from math import prod
from typing import Tuple, Union

from numpy import ndarray
import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..core.node import Node
from ..ttns.ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator
from .tree_cach_dict import PartialTreeCachDict
from ..util.tensor_util import tensor_matricisation_half as _numpy_matricisation_half
from .contraction_util import contract_all_but_one_neighbour_block_to_hamiltonian
from ..util.ttn_exceptions import NotCompatibleException
from ..util import lasyten_backend as lb
from ..util.tensor_dispatch import is_lasyten_tensor, tensordot, transpose

Tensor = Union[ndarray, lb.LasyTenArray]

def tensor_matricisation_half(tensor):
    """Fuse output/input halves, retaining native symmetry and LegPipe metadata."""
    if is_lasyten_tensor(tensor):
        tensor = lb.as_lasyten(tensor)
        if tensor.ndim % 2:
            raise ValueError("An effective Hamiltonian must have an even tensor rank.")
        half = tensor.ndim // 2
        return lb.tensor_matricization(
            tensor, tuple(range(half)), tuple(range(half, tensor.ndim)),
            correctly_ordered=True,
        )
    return _numpy_matricisation_half(tensor)


class LasyTenEffectiveHamiltonian:
    """Cached matrix-free Hamiltonian action on a native site tensor.

    Unlike scipy.sparse.linalg.LinearOperator, matvec takes a tensor with its
    original leg spaces, including a ChargedTensor's hidden charge. The
    shape attribute describes the full public product spaces, not a packed
    fixed-charge vector length. input_shape and output_shape are tensor shapes.
    NumPy is used only for scalar dtype and contraction-planning metadata.

    Rebuild after replacing Hamiltonian/cache entries or changing bond spaces.
    No Hermiticity assumption is made. Only the existing bosonic adapter
    symmetries (NoSymmetry and U1) are supported.
    """

    def __init__(self, fixed_operands, path, input_legs, output_legs,
                 input_shape, output_shape, dtype):
        self._fixed = tuple(
            (fixed_operands[i], tuple(fixed_operands[i + 1]))
            for i in range(0, len(fixed_operands), 2)
        )
        self._path = tuple(tuple(step) for step in path[1:])
        self._input_labels = tuple(input_legs)
        self._output_labels = tuple(output_legs)
        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.shape = (prod(self.output_shape), prod(self.input_shape))
        self.dtype = np.dtype(dtype)

        # Every input/output label has one opposite leg on the fixed network.
        legs = {}
        for tensor, labels in self._fixed:
            legs.update(zip(labels, tensor.legs))
        self.input_spaces = tuple(legs[label].dual for label in self._input_labels)
        self.output_spaces = tuple(legs[label] for label in self._output_labels)

    def matvec(self, tensor):
        """Return H_eff(tensor), preserving symmetry and tensor axis order."""
        if not is_lasyten_tensor(tensor):
            raise TypeError(
                "LASyTen matvec expects a LASyTen site tensor, not a flattened "
                "NumPy vector. A NumPy Davidson solver needs symmetry-block packing."
            )
        tensor = lb.as_lasyten(tensor)
        if tensor.shape != self.input_shape:
            raise ValueError(f"Expected trial tensor shape {self.input_shape}, got {tensor.shape}.")
        for axis, (actual, expected) in enumerate(zip(tensor.legs, self.input_spaces)):
            if not lb.spaces_contractible(actual, expected.dual):
                raise ValueError(f"Trial tensor leg {axis} has an incompatible symmetry space.")

        operands = [(tensor, self._input_labels), *self._fixed]
        for selected in self._path:
            value, labels = operands[selected[0]]
            for index in selected[1:]:
                other, other_labels = operands[index]
                shared = tuple(label for label in labels if label in other_labels)
                axes_a = tuple(labels.index(label) for label in shared)
                axes_b = tuple(other_labels.index(label) for label in shared)
                value = tensordot(value, other, axes=(axes_a, axes_b))
                labels = tuple(label for label in labels if label not in shared) + tuple(
                    label for label in other_labels if label not in shared
                )
            operands = [operand for index, operand in enumerate(operands)
                        if index not in selected]
            operands.append((value, labels))
        if len(operands) != 1:
            raise RuntimeError("Incomplete native Hamiltonian contraction path.")
        result, labels = operands[0]
        result = transpose(result, tuple(labels.index(label) for label in self._output_labels))
        # Contraction/fusion may otherwise assign repeated synthetic labels.
        result.native.set_labels([None] * result.ndim)
        return result

    __call__ = matvec

    def __matmul__(self, tensor):
        return self.matvec(tensor)


def find_tensor_leg_permutation(state_node: Node,
                                hamiltonian_node: Node
                                ) -> Tuple[int,...]:
    """
    Find the correct permutation to permute the effective hamiltonian
    tensor to fit with the state tensor legs.
    
    After contracting all the cached tensors to the site Hamiltonian, the
    legs of the resulting tensor are in the order of the Hamiltonian TTNO.
    However, they need to be permuted to match the legs of the site's
    state tensor. Such that the two can be easily contracted.

    Args:
        state_node (Node): The node of the state tensor.
        hamiltonian_node (Node): The node of the Hamiltonian tensor.
    
    Returns:
        Tuple[int,...]: The permutation to apply to the effective
            Hamiltonian tensor to match the legs of the state tensor.

    """
    permutation = [hamiltonian_node.neighbour_index(neighbour_id)
                   for neighbour_id in state_node.neighbouring_nodes()]
    n_open = state_node.nopen_legs()
    output_legs = []
    input_legs = []
    for hamiltonian_index in permutation:
        output_legs.append(2*n_open + 2*hamiltonian_index + 1)
        input_legs.append(2*n_open + 2*hamiltonian_index)
    output_legs.extend(list(range(n_open)))
    input_legs.extend(list(range(n_open, 2*n_open)))
    output_legs.extend(input_legs)
    return tuple(output_legs)

def contract_all_except_node(state_node: Node,
                             hamiltonian_node: Node,
                             hamiltonian_tensor: Tensor,
                             tensor_cache: PartialTreeCachDict) -> Tensor:
    """
    Contract bra, ket and hamiltonian for all but one node into that
    node's Hamiltonian tensor.

    Uses the cached trees to contract the bra, ket, and hamiltonian
    tensors for all nodes in the trees apart from the given target node.
    All the resulting tensors are contracted to the hamiltonian tensor
    corresponding to the target node.

    Args:
        state_node (GraphNode): The node of the state tensor.
        hamiltonian_node (GraphNode): The node of the Hamiltonian tensor.
        hamiltonian_tensor (ndarray): The Hamiltonian tensor that will be
            contracted with the cached tensors.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.
    
    Returns:
        ndarray: The tensor resulting from the contraction::
        
             _____        out        _____
            |     |____n-1    0_____|     |
            |     |                 |     |
            |     |        |n       |     |
            |     |     ___|__      |     |
            |     |    |      |     |     |
            |     |____|      |_____|     |
            |     |    |   H  |     |     |
            |     |    |______|     |     |
            |     |        |        |     |
            |     |        |2n+1    |     |
            |     |                 |     |
            |     |_____       _____|     |
            |_____|  2n         n+1 |_____|
                            in

            where n is the number of neighbours of the node.

    """
    target_node_id = state_node.identifier
    neighbours = hamiltonian_node.neighbouring_nodes()
    for neighbour_id in neighbours:
        cached_tensor = tensor_cache.get_entry(neighbour_id,
                                                target_node_id)
        hamiltonian_tensor = tensordot(hamiltonian_tensor,
                                       cached_tensor,
                                       axes=([0], [1]))
    # Transposing to have correct leg order
    axes = find_tensor_leg_permutation(state_node, hamiltonian_node)
    output_tensor = transpose(hamiltonian_tensor, axes=axes)
    return output_tensor

def get_effective_single_site_hamiltonian_nodes_sparse(state_node: Node,
                                                hamiltonian_node: Node,
                                                hamiltonian_tensor: Tensor,
                                                tensor_cache: PartialTreeCachDict,
                                                *,
                                                matrix_free: bool = True
                                                ) -> Union[ndarray, lb.LasyTenArray, LinearOperator, LasyTenEffectiveHamiltonian]:
    """
    Obtain the effective site Hamiltonian as a matrix or linear operator.

    Args:
        state_node (Node): The node of the state tensor.
        hamiltonian_node (Node): The node of the Hamiltonian tensor.
        hamiltonian_tensor (ndarray): The Hamiltonian tensor that will be
            contracted with the cached tensors.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.
        matrix_free (bool): Return a LinearOperator that contracts the local
            Hamiltonian action without constructing its matrix. The contraction
            path is planned once. Defaults to True.

    Returns:
        ndarray | LasyTenArray | LinearOperator | LasyTenEffectiveHamiltonian:
            The effective site Hamiltonian. Native matrix-free matvecs take tensors.

    """
    if matrix_free:
        return _single_site_hamiltonian_linear_operator(state_node,
                                                        hamiltonian_node,
                                                        hamiltonian_tensor,
                                                        tensor_cache)
    tensor = contract_all_except_node(state_node,
                                      hamiltonian_node,
                                      hamiltonian_tensor,
                                      tensor_cache)
    return tensor_matricisation_half(tensor)

def get_effective_single_site_hamiltonian_sparse(node_id: str,
                                          state: TreeTensorNetworkState,
                                          hamiltonian: TreeTensorNetworkOperator,
                                          tensor_cache: PartialTreeCachDict,
                                          *,
                                          matrix_free: bool = True
                                          ) -> Union[ndarray, lb.LasyTenArray, LinearOperator, LasyTenEffectiveHamiltonian]:
    """
    Obtain the effective site Hamiltonian as a matrix or linear operator.

    Args:
        node_id (str): The identifier of the node for which the effective
            Hamiltonian is to be found.
        state (TreeTensorNetworkState): The state of the system.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian of the
            system.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.
        matrix_free (bool): Return a LinearOperator instead of constructing
            the dense matrix. Defaults to True. Rebuild the operator after
            changing the local Hamiltonian, environments, or bond dimensions.

    Returns:
        ndarray | LasyTenArray | LinearOperator | LasyTenEffectiveHamiltonian:
            The effective site Hamiltonian. Native matrix-free matvecs take tensors. The matrix
            free NumPy form supports ``operator @ vector``. The LASyTen form
            supports ``operator @ tensor`` and ``operator.matvec(tensor)``;
            neither matrix-free form provides a dense ``diagonal()`` method.

    """
    state_node = state.nodes[node_id]
    hamiltonian_node, hamiltonian_tensor = hamiltonian[node_id]
    return get_effective_single_site_hamiltonian_nodes_sparse(state_node,
                                                       hamiltonian_node,
                                                       hamiltonian_tensor,
                                                       tensor_cache,
                                                       matrix_free=matrix_free)


def _single_site_hamiltonian_linear_operator(
        state_node: Node,
        hamiltonian_node: Node,
        hamiltonian_tensor: Tensor,
        tensor_cache: PartialTreeCachDict) -> Union[LinearOperator, LasyTenEffectiveHamiltonian]:
    """Plan a local Hamiltonian action in the state's tensor-leg order.

    Environment legs are ordered (ket, operator, bra). Keeping the trial
    tensor in the contraction network lets the planner interleave environment
    and operator contractions, rather than building the dense effective
    Hamiltonian or attaching every environment to the ket first.

    The returned operator holds references to the current environment tensors.
    Its contraction path controls operation order, not a hard memory limit.
    """
    neighbours = state_node.neighbouring_nodes()
    nvirtual = len(neighbours)
    nphysical = state_node.nopen_legs()
    if set(neighbours) != set(hamiltonian_node.neighbouring_nodes()):
        raise ValueError("State and Hamiltonian neighbours must match.")
    if hamiltonian_node.nopen_legs() != 2 * nphysical:
        raise ValueError("Hamiltonian must have output and input physical legs.")

    # Use separate labels for the ket, bra, and operator virtual indices.
    ket_legs = list(range(nvirtual))
    bra_legs = list(range(nvirtual, 2 * nvirtual))
    operator_legs = list(range(2 * nvirtual, 3 * nvirtual))
    physical_in = list(range(3 * nvirtual, 3 * nvirtual + nphysical))
    physical_out = list(range(3 * nvirtual + nphysical,
                              3 * nvirtual + 2 * nphysical))
    input_legs = ket_legs + physical_in
    output_legs = bra_legs + physical_out
    operator_leg_by_neighbour = dict(zip(neighbours, operator_legs))
    hamiltonian_legs = [operator_leg_by_neighbour[neighbour]
                        for neighbour in hamiltonian_node.neighbouring_nodes()]
    hamiltonian_legs += physical_out + physical_in

    fixed_operands = [hamiltonian_tensor, hamiltonian_legs]
    output_shape = []
    input_shape = tuple(state_node.shape)
    for i, neighbour in enumerate(neighbours):
        environment = tensor_cache.get_entry(neighbour, state_node.identifier)
        fixed_operands.extend([environment,
                               [ket_legs[i], operator_legs[i], bra_legs[i]]])
        output_shape.append(environment.shape[2])
    output_shape.extend(hamiltonian_tensor.shape[nvirtual:nvirtual + nphysical])
    native_flags = [is_lasyten_tensor(value) for value in fixed_operands[::2]]
    native = any(native_flags)
    if native and not all(native_flags):
        raise TypeError("Hamiltonian and environments must all use the same tensor backend.")
    if native:
        fixed_operands[::2] = [lb.as_lasyten(value) for value in fixed_operands[::2]]
    dtype = np.result_type(*(value.dtype for value in fixed_operands[::2]))
    # einsum_path coerces operands to NumPy. Supply only broadcast shape
    # placeholders, including for every fixed native operand.
    planning_operands = fixed_operands.copy()
    planning_operands[::2] = [
        np.broadcast_to(np.zeros((), dtype=dtype), value.shape)
        for value in fixed_operands[::2]
    ]

    # A broadcast scalar supplies shape information without allocating a trial
    # vector. The default einsum_path budget can reject all useful pairwise
    # contractions and fall back to an expensive all-at-once contraction.
    # Allow the greedy planner to consider larger binary intermediates.
    trial = np.broadcast_to(np.zeros((), dtype=dtype), input_shape)
    if neighbours:
        # Include the trial tensor in the first contraction. Otherwise a
        # planner may assemble the full effective matrix before applying it.
        # Choose the environment giving the smallest first intermediate.
        first = min(range(nvirtual), key=lambda i:
                    trial.size // input_shape[i]
                    * fixed_operands[2 + 2 * i].shape[1]
                    * fixed_operands[2 + 2 * i].shape[2])
        intermediate_legs = [leg for leg in input_legs if leg != ket_legs[first]]
        intermediate_legs += [operator_legs[first], bra_legs[first]]
        environment = fixed_operands[2 + 2 * first]
        intermediate_shape = [dim for i, dim in enumerate(input_shape)
                               if i != first]
        intermediate_shape += [environment.shape[1], environment.shape[2]]
        intermediate = np.broadcast_to(np.zeros((), dtype=dtype),
                                        intermediate_shape)
        remaining = planning_operands.copy()
        del remaining[2 + 2 * first:4 + 2 * first]
        tail, _ = np.einsum_path(*remaining, intermediate, intermediate_legs,
                                 output_legs,
                                 optimize=("greedy", 2**63 - 1))
        path = ["einsum_path", (0, 2 + first), *tail[1:]]
    else:
        path, _ = np.einsum_path(trial, input_legs, *planning_operands, output_legs,
                                 optimize=("greedy", 2**63 - 1))

    if native:
        return LasyTenEffectiveHamiltonian(
            fixed_operands, path, input_legs, output_legs,
            input_shape, tuple(output_shape), dtype,
        )

    def matvec(vector: ndarray) -> ndarray:
        tensor = np.asarray(vector).reshape(input_shape)
        result = np.einsum(tensor, input_legs, *fixed_operands, output_legs,
                            optimize=path)
        return result.reshape(-1)

    shape = (int(np.prod(output_shape)), int(np.prod(input_shape)))
    return LinearOperator(shape=shape, matvec=matvec, dtype=dtype)

def get_effective_bond_hamiltonian_tensor(
                                         state_node: Node,
                                         tensor_cache: PartialTreeCachDict
                                         ) -> Tensor:
    """
    Obtains the effective bond Hamiltonian as a matrix.

    Args:
        state_node (GraphNode): The node of the state tensor, representing the
            bond for which no equivalent node exists in the TTNO.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.

    Returns:
        ndarray: The effective bond Hamiltonian

             _____        out        _____
            |     |____0      1_____|     |
            |     |                 |     |
            |     |                 |     |
            |     |                 |     |
            |     |                 |     |
            |     |_________________|     |
            |     |                 |     |
            |     |                 |     |
            |     |                 |     |
            |     |                 |     |
            |     |                 |     |
            |     |_____       _____|     |
            |_____|  2         3    |_____|
                            in    
    
    """
    neighbours = state_node.neighbouring_nodes()
    if len(neighbours) != 2:
        raise ValueError("The effective bond Hamiltonian can only be "
                         "computed for a bond with two neighbours.")
    nghbrp = state_node.parent
    nghbrc = state_node.children[0]
    tensorp = tensor_cache.get_entry(nghbrp,
                                     nghbrc)
    tensorc = tensor_cache.get_entry(nghbrc,
                                     nghbrp)
    # Contract the Hamiltonian legs
    tensor = tensordot(tensorp,
                       tensorc,
                       axes=(1, 1))
    tensor = transpose(tensor, axes=[1, 3, 0, 2])
    return tensor

def get_effective_bond_hamiltonian_nodes(
                                         state_node: Node,
                                         tensor_cache: PartialTreeCachDict
                                         ) -> Tensor:
    """
    Obtains the effective bond Hamiltonian as a matrix.

    Args:
        state_node (GraphNode): The node of the state tensor, representing the
            bond for which no equivalent node exists in the TTNO.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.

    Returns:
        ndarray: The effective bond Hamiltonian as a matrix.
    
    """
    tensor = get_effective_bond_hamiltonian_tensor(state_node,
                                                    tensor_cache)
    return tensor_matricisation_half(tensor)

def get_effective_bond_hamiltonian(bond_node_id: str,
                                   state: TreeTensorNetworkState,
                                   tensor_cache: PartialTreeCachDict
                                   ) -> Tensor:
    """
    Obtains the effective bond Hamiltonian as a matrix.

    Args:
        bond_node_id (str): The identifier of the node representing the bond
            for which the effective Hamiltonian is to be found.
        state (TreeTensorNetworkState): The state of the system.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.

    Returns:
        ndarray: The effective bond Hamiltonian as a matrix.
    """
    state_node = state.nodes[bond_node_id]
    return get_effective_bond_hamiltonian_nodes(state_node, tensor_cache)

def contract_all_except_two_nodes(state_node: Node,
                                  target_node: Node,
                                  target_tensor: Tensor,
                                  next_node: Node,
                                  next_tensor: Tensor,
                                  tensor_cache: PartialTreeCachDict
                                  ) -> Tensor:
    """
    Contracts the nodes for all but two sites.

    Uses all cached tensors to contract bra, ket, and hamiltonian tensors
    of all nodes except for the two given nodes. Of these nodes only the
    Hamiltonian nodes are contracted.
    IMPORTANT: This function assumes that the given nodes are already
    contracted in the TTNS.

    Args:
        state_node (Node): The node of the state tensor.
        target_node (Node): The target node.
        target_tensor (ndarray): The tensor of the target node.
        next_node (Node): The next node.
        next_tensor (ndarray): The tensor of the next node.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.

    Returns:
        np.ndarray: The resulting effective two-site Hamiltonian tensor::

             _____                out              _____
            |     |____n-1                  0_____|     |
            |     |                               |     |
            |     |        |n           |         |     |
            |     |     ___|__        __|___      |     |
            |     |    |      |      |      |     |     |
            |     |____|      |______|      |_____|     |
            |     |    |  H_1 |      | H_2  |     |     |
            |     |    |______|      |______|     |     |
            |     |        |            |         |     |
            |     |        |2n+1        |         |     |
            |     |                               |     |
            |     |_____                     _____|     |
            |_____|  2n                       n+1 |_____|
                                    in
    
    """
    # Contract all but one neighbouring block of each node
    target_node_id = target_node.identifier
    next_node_id = next_node.identifier
    target_block = contract_all_but_one_neighbour_block_to_hamiltonian(target_tensor,
                                                                        target_node,
                                                                        next_node_id,
                                                                        tensor_cache)

    next_block = contract_all_but_one_neighbour_block_to_hamiltonian(next_tensor,
                                                                        next_node,
                                                                        target_node_id,
                                                                        tensor_cache)
    # Contract the two blocks
    h_eff = tensordot(target_block, next_block, axes=(0,0))
    # Now we need to sort the legs to fit with the underlying TTNS.
    # Note that we assume, the two tensors have already been contracted.
    leg_permutation = _determine_two_site_leg_permutation(state_node, target_node,
                                                                next_node)
    return transpose(h_eff, leg_permutation)

def create_two_site_id(node_id: str, next_node_id: str) -> str:
    """
    Create the identifier of a two site node obtained from contracting
    the two note with the input identifiers.
    """
    return "TwoSite_" + node_id + "_contr_" + next_node_id

def _determine_two_site_leg_permutation(state_node: Node,
                                        target_node: Node,
                                        next_node: Node) -> Tuple[int]:
    """
    Determine the permutation of the effective Hamiltonian tensor.
    
    This is the leg permutation required on the two-site effective
    Hamiltonian tensor to fit with the underlying TTNS, assuming
    the two sites have already been contracted in the TTNS.

    Args:
        state (TreeTensorNetworkState): The state of the system.
        target_node (Node): The target node.
        next_node (Node): The next node.
    
    Returns:
        Tuple[int]: The permutation of the legs of the two-site effective
            Hamiltonian tensor.
    """
    neighbours_target = target_node.neighbouring_nodes()
    neighbours_next = next_node.neighbouring_nodes()
    neighbours_two_site = state_node.neighbouring_nodes()
    next_node_id = next_node.identifier
    
    # Determine the permutation of the legs
    input_legs = []
    for neighbour_id in neighbours_two_site:
        if neighbour_id in neighbours_target:
            block_leg = _find_block_leg_target_node(target_node,
                                                            next_node_id,
                                                            neighbour_id)
        elif neighbour_id in neighbours_next:
            block_leg = _find_block_leg_next_node(target_node,
                                                        next_node,
                                                        neighbour_id)
        else:
            errstr = "The two-site Hamiltonian has a neighbour that is not a neighbour of the two sites."
            raise NotCompatibleException(errstr)
        input_legs.append(block_leg)
    output_legs = [leg + 1 for leg in input_legs]
    target_num_neighbours = target_node.nneighbours()
    output_legs = output_legs + [0,2*target_num_neighbours] # physical legs
    input_legs = input_legs + [1,2*target_num_neighbours + 1] # physical legs
    # As in matrices, the output legs are first
    return tuple(output_legs + input_legs)

def _find_block_leg_target_node(target_node: Node,
                                next_node_id: str,
                                neighbour_id: str) -> int:
    """
    Determines the leg index of the input leg of the contracted subtree
    block on the effective hamiltonian tensor corresponding to a given
    neighbour of the target node.

    Args:
        target_node (Node): The target node.
        next_node_id (str): The id of the next node.
        neighbour_id (str): The id of the neighbour of the target node.
    
    Returns:
        int: The leg index of the input leg of the contracted subtree
            block on the effective hamiltonian tensor.
    """
    index_next_node = target_node.neighbour_index(next_node_id)
    ham_neighbour_index = target_node.neighbour_index(neighbour_id)
    constant = int(ham_neighbour_index < index_next_node)
    return 2 * (ham_neighbour_index + constant)

def _find_block_leg_next_node(target_node: Node,
                              next_node: Node,
                              neighbour_id: str) -> int:
    """
    Determines the leg index of the input leg of the contracted subtree
    bnlock on the effective hamiltonian tensor corresponding to a given
    eighbour of the next node.

    Args:
        target_node (Node): The target node.
        next_node (Node): The next node.
        neighbour_id (str): The id of the neighbour of the next node.
    
    Returns:
        int: The leg index of the input leg of the contracted subtree
            block on the effective hamiltonian tensor.
    """
    # Luckily the situation is pretty much the same so we can reuse most
    # of the code.
    target_node_id = target_node.identifier
    leg_index_temp = _find_block_leg_target_node(next_node,
                                                        target_node_id,
                                                        neighbour_id)
    target_node_numn = target_node.nneighbours()
    return 2 * target_node_numn + leg_index_temp

def get_effective_two_site_hamiltonian_nodes(state_node: Node,
                                             target_node: Node,
                                             target_tensor: Tensor,
                                             next_node: Node,
                                             next_tensor: Tensor,
                                             tensor_cache: PartialTreeCachDict) -> Tensor:
    """
    Obtains the effective two-site Hamiltonian as a matrix.

    Args:
        state_node (Node): The node of the state tensor.
        target_node (Node): The node of the target Hamiltonian tensor.
        target_tensor (ndarray): The tensor of the target node.
        next_node (Node): The node of the next Hamiltonian tensor.
        next_tensor (ndarray): The tensor of the next node.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.

    Returns:
        ndarray: The effective two-site Hamiltonian

    """
    tensor = contract_all_except_two_nodes(state_node,
                                          target_node,
                                          target_tensor,
                                          next_node,
                                          next_tensor,
                                          tensor_cache)
    return tensor_matricisation_half(tensor)

def get_effective_two_site_hamiltonian(target_node_id: str,
                                       next_node_id: str,
                                       state: TreeTensorNetworkState,
                                       hamiltonian: TreeTensorNetworkOperator,
                                       tensor_cache: PartialTreeCachDict) -> Tensor:
    """
    Obtains the effective site Hamiltonian as a matrix.

    Args:
        node_id (str): The identifier of the node for which the effective
            Hamiltonian is to be found.
        state (TreeTensorNetworkState): The state of the system.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian of the
            system.
        tensor_cache (PartialTreeCachDict): The cache of environment tensors.

    Returns:
        ndarray: The effective site Hamiltonian

    """
    target_node, target_tensor = hamiltonian[target_node_id]
    next_node, next_tensor = hamiltonian[next_node_id]
    two_site_id = create_two_site_id(target_node_id, next_node_id)
    state_node = state.nodes[two_site_id]
    return get_effective_two_site_hamiltonian_nodes(state_node,
                                                    target_node,
                                                    target_tensor,
                                                    next_node,
                                                    next_tensor,
                                                    tensor_cache)
