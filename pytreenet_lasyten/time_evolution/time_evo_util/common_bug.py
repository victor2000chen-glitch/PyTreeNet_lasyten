"""
This module contains the shared BUG updates for NumPy and LASyTen TTNS.

Native basis construction is delegated to bug_util: the centre may carry a
fixed U(1) charge, but every retained nonroot basis must remain invariant.

"""
from typing import Tuple
from copy import deepcopy, copy
from dataclasses import dataclass

from numpy import ndarray

from ...util.tensor_dispatch import tensordot, conjugate, transpose

from ...ttns import TreeTensorNetworkState
from ...ttno import TreeTensorNetworkOperator
from ...contractions.sandwich_caching import SandwichCache
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...core.node import relative_leg_permutation
from .effective_time_evolution import (single_site_time_evolution,
                                      effective_single_site_evolution)
from ...util.tensor_splitting import SplitMode
from ...contractions.state_operator_contraction import contract_leaf
from ...core.leg_specification import LegSpecification
from ...contractions.state_operator_contraction import contract_any_node_environment_but_one
from ..ttn_time_evolution import TTNTimeEvolutionConfig
from ..time_evolution import TimeEvoMode

from .bug_util import (basis_change_tensor_id,
                       copy_state_for_update,
                       project_child_basis_changes,
                       compute_basis_change_tensor,
                       compute_new_basis_tensor,
                       compute_fixed_size_new_basis_tensor)

@dataclass
class CommonBUGConfig(TTNTimeEvolutionConfig):
    """
    The configuration class for the BUG methods.

    Attributes:
        deep (bool): Whether to use deepcopies of the TTNS during the update.
            If False, only the relevant nodes are copied at each point.
        fixed_rank (bool): Whether to use the fixed rank BUG or the standard
            BUG.

    """
    deep: bool = False
    fixed_rank: bool = False
    time_evo_mode: TimeEvoMode = TimeEvoMode.FASTEST
    # MPI uses one serial subtree kernel per worker process.
    subtree_workers: int = 1

def update_leaf_node(node_id: str,
                current_state: TreeTensorNetworkState,
                new_state: TreeTensorNetworkState,
                parent_state: TreeTensorNetworkState,
                current_cache: SandwichCache,
                hamiltonian: TreeTensorNetworkOperator,
                time_step_size: float,
                bug_config: CommonBUGConfig,
                solver_options: dict = None
                ) -> Tuple[TreeTensorNetworkState, ndarray, ndarray]:
    """
    Updates a leaf node according to the fixed rank BUG.

    Args:
        node_id (str): The id of the node to be updated.
        current_state (TreeTensorNetworkState): The state, where the current
            node is the orthogonality center.
        new_state (TreeTensorNetworkState): The state into which all the
            updated nodes are stored.
        parent_state (TreeTensorNetworkState): The state where the parent of
            the current node is the orthogonality center.
        current_cache (SandwichCache): The cache for the neighbour blocks.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian operator of
            model as a TTNO.
        time_step_size (float): The time step size.
        fixed_rank (bool): Whether to use the fixed rank BUG or the standard
            BUG.
        solver_options (Union[Dict[str, Any], None], optional): Most time
            evolutions algorithms use some kind of solver to resolve a
            partial differential equation. This dictionary can be used to
            pass additional options to the solver. Refer to the
            documentation of `ptn.time_evolution.TimeEvoMode` for further
            information. Defaults to None.

    Returns:
        TreeTensorNetworkState: The updated new state.
        ndarray: The neighbour block towards the parent, with the updated
            tensors.
        ndarray: The basis change tensor.

    """
    updated_tensor = single_site_time_evolution(node_id,
                                                current_state,
                                                hamiltonian,
                                                time_step_size,
                                                current_cache,
                                                mode=bug_config.time_evo_mode,
                                                solver_options=solver_options)
    old_basis_tensor = parent_state.tensors[node_id]
    basis_node = parent_state.nodes[node_id]
    if bug_config.fixed_rank:
        new_basis_tensor = compute_fixed_size_new_basis_tensor(basis_node,
                                                               updated_tensor)
    else:
        new_basis_tensor = compute_new_basis_tensor(basis_node,
                                                    old_basis_tensor,
                                                    updated_tensor)
    basis_change_tensor = tensordot(old_basis_tensor,
                                    conjugate(new_basis_tensor),
                                    axes=([1],[1]))
    assert not current_state.nodes[node_id].is_root()
    parent_id = current_state.nodes[node_id].parent
    new_state.split_node_replace(node_id,
                                 basis_change_tensor,
                                 new_basis_tensor,
                                 basis_change_tensor_id(node_id),
                                 node_id,
                                 LegSpecification(parent_id,[],[]),
                                 LegSpecification(None, [], [1]))
    state_node, state_tensor = new_state[node_id]
    op_node, op_tensor = hamiltonian[node_id]
    block_tensor = contract_leaf(state_node, state_tensor,
                                 op_node, op_tensor)
    return new_state, block_tensor, basis_change_tensor

def update_non_leaf_node(node_id: str,
                         current_state: TreeTensorNetworkState,
                         new_state: TreeTensorNetworkState,
                         parent_state: TreeTensorNetworkState,
                         current_cache: SandwichCache,
                         hamiltonian: TreeTensorNetworkOperator,
                         time_step_size: float,
                         bug_config: CommonBUGConfig,
                         solver_options: dict = None
                         ) -> Tuple[TreeTensorNetworkState, ndarray, ndarray]:
    """
    Updates a non-leaf node according to the fixed rank BUG.

    Args:
        node_id (str): The id of the node to be updated.
        current_state (TreeTensorNetworkState): The state, where the current
            node is the orthogonality center.
        new_state (TreeTensorNetworkState): The state into which all the
            updated nodes are stored.
        parent_state (TreeTensorNetworkState): The state where the parent of
            the current node is the orthogonality center.
        current_cache (SandwichCache): The cache for the neighbour blocks.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian operator of
            model as a TTNO.
        time_step_size (float): The time step size.
        bug_config (CommonBUGConfig): The configuration for the BUG method.
        solver_options (Union[Dict[str, Any], None], optional): Most time
            evolutions algorithms use some kind of solver to resolve a
            partial differential equation. This dictionary can be used to
            pass additional options to the solver. Refer to the
            documentation of `ptn.time_evolution.TimeEvoMode` for further
            information. Defaults to None.

    Returns:
        TreeTensorNetworkState: The updated new state.
        ndarray: The neighbour block towards the parent, with the updated
            tensors.
        ndarray: The basis change tensor.

    """
    child_environment_cache = PartialTreeCachDict() # We only need to add entries, so no Sandwich needed
    child_bc_tensors = PartialTreeCachDict() # Needed to save the basis change tensors of the children
    for child_id in tuple(current_state.nodes[node_id].children):
        new_state, child_block, child_bc_tensor = update_node(child_id,
                                                                new_state,
                                                                current_state,
                                                                current_cache,
                                                                hamiltonian,
                                                                time_step_size,
                                                                bug_config=bug_config,
                                                                solver_options=solver_options)
        child_bc_tensors.add_entry(child_id, node_id, child_bc_tensor)
        child_environment_cache.add_entry(child_id, node_id, child_block)
    # If we update it inside the loop, the cache changed after the first child, but
    # we need the old cache for the other children updates.
    return _finish_non_leaf_update(
        node_id, current_state, new_state, parent_state, current_cache,
        hamiltonian, time_step_size, bug_config,
        child_environment_cache, child_bc_tensors,
        solver_options=solver_options,
    )


def _finish_non_leaf_update(node_id, current_state, new_state, parent_state,
                            current_cache, hamiltonian, time_step_size,
                            bug_config, child_environment_cache,
                            child_bc_tensors, solver_options=None):
    """Complete one ancestor after its child subtrees have been updated."""
    current_cache.update(child_environment_cache)
    # Assemble the invariant subtree in its original parent gauge. A charged
    # centre from current_state can have a different external parent space;
    # inserting it here would temporarily break that bond in new_state.
    new_state.contract_all_children(node_id)
    new_state_node, _ = new_state[node_id]
    centre_node, centre_tensor = current_state[node_id]
    permutation = relative_leg_permutation(centre_node, new_state_node)
    centre_tensor = transpose(centre_tensor, permutation)
    old_tensor = project_child_basis_changes(new_state_node, centre_tensor,
                                             child_bc_tensors)
    # This metadata belongs only to the effective local problem, not to the
    # partially rebuilt TTN. Its parent matches the old parent environment;
    # its child spaces match the newly computed child environments.
    local_node = deepcopy(new_state_node)
    local_node.link_tensor(old_tensor)
    op_node, op_tensor = hamiltonian[node_id]
    updated_tensor = effective_single_site_evolution(
        old_tensor, local_node, op_tensor, op_node,
        time_step_size, current_cache,
        mode=bug_config.time_evo_mode,
        **({} if solver_options is None else solver_options),
    )
    if bug_config.fixed_rank:
        new_basis_tensor = compute_fixed_size_new_basis_tensor(local_node,
                                                                updated_tensor)
    else:
        new_basis_tensor = compute_new_basis_tensor(local_node,
                                                    old_tensor,
                                                    updated_tensor)
    old_basis_node, old_basis_tensor = parent_state[node_id]
    basis_change_tensor = compute_basis_change_tensor(old_basis_node,
                                                        new_state_node,
                                                        old_basis_tensor,
                                                        new_basis_tensor,
                                                        child_bc_tensors)
    # Now we need to insert the new tensors into the new state
    new_state.split_node_replace(node_id,
                                 basis_change_tensor,
                                 new_basis_tensor,
                                 basis_change_tensor_id(node_id),
                                 node_id,
                                 LegSpecification(new_state_node.parent,[],[]),
                                 LegSpecification(None, new_state_node.children, new_state_node.open_legs))
    state_node, state_tensor = new_state[node_id]
    op_node, op_tensor = hamiltonian[node_id]
    # The new basis has a temporary basis-change node as its parent. Give
    # only this local operator metadata the same parent name. This lets both
    # contraction orders ignore the same leg; the Hamiltonian is untouched.
    local_op_node = copy(op_node)
    local_op_node.parent = state_node.parent
    block_tensor = contract_any_node_environment_but_one(
        state_node.parent,
        state_node, state_tensor,
        local_op_node, op_tensor,
        current_cache,
    )
    return new_state, block_tensor, basis_change_tensor

def update_node(node_id: str,
                new_state: TreeTensorNetworkState,
                parent_state: TreeTensorNetworkState,
                parent_tensor_cache: SandwichCache,
                hamiltonian: TreeTensorNetworkOperator,
                time_step_size: float,
                bug_config: CommonBUGConfig,
                solver_options: dict = None
                ) -> Tuple[TreeTensorNetworkState, ndarray, ndarray]:
    """
    Updates a node according to the fixed rank BUG.

    Args:
        node_id (str): The id of the node to be updated.
        new_state (TreeTensorNetworkState): The state into which all the
            updated nodes are stored.
        parent_state (TreeTensorNetworkState): The state where the parent is
            the orthogonality center.
        parent_tensor_cache (SandwichCache): The cache for the parents
            neighbour blocks.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian operator of
            model as a TTNO.
        time_step_size (float): The time step size.
        bug_config (CommonBUGConfig): The configuration for the BUG method.
        solver_options (Union[Dict[str, Any], None], optional): Most time
            evolutions algorithms use some kind of solver to resolve a
            partial differential equation. This dictionary can be used to
            pass additional options to the solver. Refer to the
            documentation of `ptn.time_evolution.TimeEvoMode` for further
            information. Defaults to None.

    Returns:
        TreeTensorNetworkState: The updated new state.
        ndarray: The neighbour block towards the parent, with the updated
            tensors.
        ndarray: The basis change tensor.

    """
    current_state, current_cache = _prepare_node_update(
        node_id, parent_state, parent_tensor_cache, bug_config)
    if current_state.nodes[node_id].is_leaf():
        return update_leaf_node(node_id,
                                current_state,
                                new_state,
                                parent_state,
                                current_cache,
                                hamiltonian,
                                time_step_size,
                                bug_config=bug_config,
                                solver_options=solver_options)
    else:
        return update_non_leaf_node(node_id,
                                    current_state,
                                    new_state,
                                    parent_state,
                                    current_cache,
                                    hamiltonian,
                                    time_step_size,
                                    bug_config=bug_config,
                                    solver_options=solver_options)


def _prepare_node_update(node_id, parent_state, parent_tensor_cache, bug_config):
    """Prepare the original gauge/environment for one recursive subtree."""
    parent_id = parent_state.nodes[node_id].parent
    assert parent_id is not None, "There is no basis change for the root node!"
    assert parent_id == parent_state.orthogonality_center_id, "The parent is not the orth center!"
    if bug_config.deep:
        current_state = deepcopy(parent_state)
    else:
        copy_nodes = [node_id, parent_id]
        current_state = copy_state_for_update(parent_state, copy_nodes)
    if bug_config.fixed_rank:
        mode = SplitMode.KEEP
    else:
        mode = SplitMode.REDUCED
    current_state.move_orthogonalization_center(node_id, mode=mode)
    current_cache = copy(parent_tensor_cache)
    current_cache.state = current_state
    # Since the orth center changed, we need to update that specific neighbour block
    current_cache.update_tree_cache(parent_id, node_id)
    # We can also delete the obsolete neighbour block from the child to the parent
    current_cache.delete_entry(node_id, parent_id)
    return current_state, current_cache

def root_update(current_state: TreeTensorNetworkState,
                hamiltonian: TreeTensorNetworkOperator,
                time_step_size: float,
                bug_config: CommonBUGConfig,
                solver_options: dict = None
                ) -> TreeTensorNetworkState:
    """
    Updates the root node of the state according to the fixed rank BUG.

    Args:
        current_state (TreeTensorNetworkState): The current state of the system
            with the root node as the orthogonality center.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian operator of
            model as a TTNO.
        time_step_size (float): The time step size.
        bug_config (CommonBUGConfig): The configuration for the BUG method.
        solver_options (Union[Dict[str, Any], None], optional): Most time
            evolutions algorithms use some kind of solver to resolve a
            partial differential equation. This dictionary can be used to
            pass additional options to the solver. Refer to the
            documentation of `ptn.time_evolution.TimeEvoMode` for further
            information. Defaults to None.
    
    Returns:
        TreeTensorNetworkState: The updated state.

    """
    if bug_config.subtree_workers != 1:
        raise ValueError("Serial BUG uses subtree_workers=1; use MPIBUG for process workers.")
    root_id = current_state.root_id
    assert root_id is not None, "The state has no root node!"
    assert current_state.orthogonality_center_id == root_id, \
        "The orthogonality center is not the root node!"
    new_state = deepcopy(current_state)
    current_cache = SandwichCache.init_cache_but_one(current_state,
                                                     hamiltonian,
                                                     root_id)
    child_environment_cache = PartialTreeCachDict() # We only need to add entries, so no Sandwich needed
    for child_id in tuple(current_state.nodes[root_id].children):
        new_state, child_block, _ = update_node(child_id,
                                                new_state,
                                                current_state,
                                                current_cache,
                                                hamiltonian,
                                                time_step_size,
                                                bug_config=bug_config,
                                                solver_options=solver_options)
        child_environment_cache.add_entry(child_id, root_id, child_block)
    # If we update it inside the loop, the cache changed after the first child, but
    # we need the old cache for the other children updates.
    return _finish_root_update(
        current_state, new_state, current_cache, hamiltonian,
        time_step_size, bug_config, child_environment_cache,
        solver_options=solver_options,
    )


def _finish_root_update(current_state, new_state, current_cache, hamiltonian,
                        time_step_size, bug_config, child_environment_cache,
                        solver_options=None):
    """Finish the root using the same native action as serial BUG."""
    root_id = current_state.root_id
    current_cache.update(child_environment_cache)
    # The copied root already contains its old centre tensor. Absorb the
    # child bridges directly; no cross-tree centre replacement is needed.
    new_state.contract_all_children(root_id)
    # Now we can update the tensor
    updated_tensor = single_site_time_evolution(root_id,
                                                new_state,
                                                hamiltonian,
                                                time_step_size,
                                                current_cache,
                                                mode=bug_config.time_evo_mode,
                                                solver_options=solver_options)
    new_state.replace_tensor(root_id, updated_tensor)
    new_state.orthogonality_center_id = root_id
    return new_state
