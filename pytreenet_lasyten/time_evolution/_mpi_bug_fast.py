"""Compact subtree MPIBUG tasks with process-local immutable-H caches.

The existing serial update_node and finishing kernels are unchanged. Worker
views intentionally retain boundary/ancestor metadata and are not checkpoints.
Only trusted, same-user process/MPI executors are supported.
"""
from collections import OrderedDict
from concurrent.futures import wait, FIRST_COMPLETED
from copy import deepcopy
from numbers import Integral
from time import perf_counter
import os
import pickle
import resource
import uuid
import weakref

from .bug import BUG
from .fixed_bug import FixedBUG
from .time_evo_util.common_bug import (_prepare_node_update, update_node,
    _finish_non_leaf_update, _finish_root_update)
from .time_evo_util.bug_util import basis_change_tensor_id
from ..contractions.sandwich_caching import SandwichCache
from ..contractions.tree_cach_dict import PartialTreeCachDict
from ..ttns import TreeTensorNetworkState
from ..ttno import TreeTensorNetworkOperator
from ..core.ttn import TensorDict
from ._mpi_block_wire import (pack_view, restore_view, pack_array, unpack_array,
                              node_record, restore_node)

_H_CACHE=OrderedDict()
_H_CACHE_LIMIT=8
_H_TENSOR_CACHE=weakref.WeakValueDictionary()
WORKER_PROTOCOL=1


class _TrackedHamiltonianDict(TensorDict):
    """Track ordinary writes, not TensorDict's internal lazy-transpose reads."""
    def __setitem__(self,key,value):
        super().__setitem__(key,value)
        self._worker_h_revision=getattr(self,'_worker_h_revision',0)+1

    def __delitem__(self,key):
        super().__delitem__(key)
        self._worker_h_revision=getattr(self,'_worker_h_revision',0)+1


def _frontier(state,target):
    if isinstance(target,bool) or not isinstance(target,Integral) or target<1:
        raise ValueError('subtree_tasks must be a positive integer')
    frontier=list(state.nodes[state.root_id].children)
    while len(frontier)<target:
        candidates=[]
        for j,start in enumerate(frontier):
            branch=start
            while len(state.nodes[branch].children)==1:
                branch=state.nodes[branch].children[0]
            if len(state.nodes[branch].children)>1:
                candidates.append((_cost(state,start),-j,branch))
        if not candidates:break
        _,minus_j,branch=max(candidates)
        j=-minus_j
        frontier[j:j+1]=state.nodes[branch].children
    return frontier


def _cost(state,node_id):
    # Cheap deterministic initial heuristic; measured subtree times replace it.
    return sum(state.tensors.data[i].size for i in state.find_subtree_of_node(node_id))


def _scope(state,node_id):
    inside=set(state.find_subtree_of_node(node_id))
    parent=state.nodes[node_id].parent
    tensors=inside|{parent}
    metadata=set(tensors)
    current=parent
    while current is not None:
        metadata.add(current)
        current=state.nodes[current].parent
    # QR of the boundary parent changes neighbour names on these metadata
    # stubs. Their coefficient tensors are never read by this subtree update.
    metadata.update(state.nodes[parent].neighbouring_nodes())
    return tensors,metadata


def _freeze_task(node_id,new_state,parent_state,parent_cache,dt,config,options):
    tensors,metadata=_scope(parent_state,node_id)
    job=dict(node_id=node_id,new=pack_view(new_state,tensors,metadata),
        parent=pack_view(parent_state,tensors,metadata),
        environments=[(a,b,pack_array(v)) for (a,b),v in parent_cache.items() if b in tensors],
        dt=dt,config=config,options=options)
    # Synchronous freezing prevents asynchronous executor pickling from seeing
    # later coordinator gauge mutations. No native objects enter this pickle.
    return pickle.dumps(job,protocol=5),tensors,metadata


def _worker(job_blob,token,h_blob=None,cache_h=True):
    started=perf_counter()
    if cache_h and token not in _H_CACHE and h_blob is None:
        return dict(need_h=True,pid=os.getpid(),token=token)
    tick=perf_counter()
    if h_blob is not None:
        # Different frontier fragments can include the same boundary operator.
        # Share its immutable coefficient array, but never mutable Node metadata.
        h=restore_view(TreeTensorNetworkOperator,pickle.loads(h_blob),
            tensor_cache=_H_TENSOR_CACHE if cache_h else None,cache_token=token[0])
        if cache_h:
            _H_CACHE[token]=h
            _H_CACHE.move_to_end(token)
            while len(_H_CACHE)>_H_CACHE_LIMIT:_H_CACHE.popitem(last=False)
    elif cache_h:
        h=_H_CACHE[token]
        _H_CACHE.move_to_end(token)
    else:
        raise ValueError('Uncached tasks must contain a Hamiltonian fragment.')
    job=pickle.loads(job_blob)
    new=restore_view(TreeTensorNetworkState,job['new'])
    parent=restore_view(TreeTensorNetworkState,job['parent'])
    environments={(a,b):unpack_array(v) for a,b,v in job['environments']}
    cache=SandwichCache(parent,h,environments)
    decode_seconds=perf_counter()-tick
    tick=perf_counter()
    state,env,bc=update_node(job['node_id'],new,parent,cache,h,job['dt'],
                           job['config'],solver_options=job['options'])
    compute_seconds=perf_counter()-tick
    tick=perf_counter()
    boundary=basis_change_tensor_id(job['node_id'])
    # Use raw tensors plus lazy permutations; no reads/canonicalization while
    # packing and no native representation spaces in Python pickle.
    records=[node_record(state.nodes[i],state.tensors.data[i])
             for i in state.find_subtree_of_node(boundary)]
    body=dict(child=job['node_id'],parent=parent.nodes[job['node_id']].parent,
              nodes=records,environment=pack_array(env),basis_change=pack_array(bc))
    result=pickle.dumps(body,protocol=5)
    return dict(need_h=False,payload=result,pid=os.getpid(),node_id=job['node_id'],
        decode_seconds=decode_seconds,compute_seconds=compute_seconds,
        encode_seconds=perf_counter()-tick,total_seconds=perf_counter()-started,
        process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)


def _merge(state,blob):
    body=pickle.loads(blob)
    for record in body['nodes']:
        node,tensor=restore_node(record)
        state.nodes[node.identifier]=node
        state.tensors.data[node.identifier]=tensor
    state.nodes[body['parent']].replace_child(body['child'],basis_change_tensor_id(body['child']))
    return body['child'],unpack_array(body['environment']),unpack_array(body['basis_change'])


class _FastMixin:
    worker_protocol_version=WORKER_PROTOCOL

    def __init__(self,*args,executor,subtree_tasks=None,cache_hamiltonian=True,**kwargs):
        if executor is None or not callable(getattr(executor,'submit',None)):
            raise ValueError('An active persistent executor is required on the coordinator')
        if subtree_tasks is None:subtree_tasks=getattr(executor,'num_workers',1)
        if isinstance(subtree_tasks,bool) or not isinstance(subtree_tasks,Integral) or subtree_tasks<1:
            raise ValueError('subtree_tasks must be a positive integer')
        self.subtree_executor=executor
        self.subtree_tasks=subtree_tasks
        self.cache_worker_hamiltonian=bool(cache_hamiltonian)
        self._worker_generation=uuid.uuid4().hex
        self._worker_h_owner=None
        self._worker_h_revision=None
        self._worker_h_fragments={}
        self._worker_costs={}
        self._worker_partition_key=None
        self.last_worker_stats={}
        super().__init__(*args,**kwargs)
        if self.config.subtree_workers!=1:
            raise ValueError('Use subtree_workers=1 with MPI; select MPI ranks instead')

    def invalidate_worker_hamiltonian(self):
        """Call after an in-place change to a nominally static Hamiltonian."""
        self._worker_generation=uuid.uuid4().hex
        self._worker_h_fragments.clear()
        self._worker_h_owner=self.hamiltonian
        self._worker_h_revision=self._hamiltonian_revision()

    def _hamiltonian_revision(self):
        tensors=self.hamiltonian.tensors
        if type(tensors) is TensorDict:
            # UserDict/TensorDict are pure Python classes with the same layout.
            # The inherited numerical storage and read behavior are unchanged.
            tensors.__class__=_TrackedHamiltonianDict
            tensors._worker_h_revision=0
        return getattr(tensors,'_worker_h_revision',None)

    def run_one_time_step(self,**kwargs):
        started=perf_counter()
        result=super().run_one_time_step(**kwargs)
        elapsed=perf_counter()-started
        self.last_worker_stats.update(step_seconds=elapsed,
            outside_recursive_seconds=max(0.,elapsed-self.last_worker_stats['recursive_seconds']))
        return result

    def recursive_update(self):
        if self.config.subtree_workers!=1:
            raise ValueError('Use subtree_workers=1 with MPI; select MPI ranks instead')
        if (self._worker_h_owner is not self.hamiltonian or
                self._worker_h_revision!=self._hamiltonian_revision()):
            self.invalidate_worker_hamiltonian()
        cache_h=(self.cache_worker_hamiltonian and self._worker_h_revision is not None
                 and not getattr(self.config,'time_dep',False))
        if not cache_h:self.invalidate_worker_hamiltonian()
        old=self.state
        root=old.root_id
        if root is None or old.orthogonality_center_id!=root:
            raise ValueError('The root must be the orthogonality centre.')
        started=perf_counter()
        # QR/contract operations can reorder the node dictionary without
        # changing the tree. Do not flush worker caches for that bookkeeping.
        partition_key=(root,self.subtree_tasks,tuple(sorted((i,n.parent,tuple(n.children))
                           for i,n in old.nodes.items())))
        if partition_key!=self._worker_partition_key:
            self.invalidate_worker_hamiltonian()
            self._worker_frontier=_frontier(old,self.subtree_tasks)
            self._worker_partition_key=partition_key
        frontier=list(self._worker_frontier)
        frontier_set=set(frontier)
        estimates={f:self._worker_costs[f] if f in self._worker_costs else _cost(old,f)
                   for f in frontier}
        subtree_cost={}
        def cost(i):
            if i not in subtree_cost:
                subtree_cost[i]=sum(estimates[f]
                    for f in frontier if i in old.find_path_to_root(f))
            return subtree_cost[i]
        # Copy once, as the serial integrator does; never commit this copy if
        # a worker fails. Old state and gauges remain unmodified.
        tick=perf_counter()
        new=deepcopy(old)
        root_cache=SandwichCache.init_cache_but_one(old,self.hamiltonian,root)
        initial_seconds=perf_counter()-tick
        ancestors=[]
        pending={}
        stats=dict(frontier=frontier,task_bytes=0,hamiltonian_bytes=0,result_bytes=0,
                   cache_misses=0,freeze_seconds=0.,prepare_seconds=0.,merge_seconds=0.,tasks=[])

        def prepare(i,parent,parent_cache):
            if i in frontier_set:
                tick=perf_counter()
                blob,tensor_ids,metadata_ids=_freeze_task(i,new,parent,parent_cache,
                    self.time_step_size,self.config,self.solver_options)
                token=(self._worker_generation,i)
                if i not in self._worker_h_fragments:
                    self._worker_h_fragments[i]=pickle.dumps(
                        pack_view(self.hamiltonian,tensor_ids,metadata_ids),protocol=5)
                hblob=self._worker_h_fragments[i]
                stats['freeze_seconds']+=perf_counter()-tick
                stats['task_bytes']+=len(blob)
                initial_h=None if cache_h else hblob
                if initial_h is not None:stats['hamiltonian_bytes']+=len(initial_h)
                future=self.subtree_executor.submit(_worker,blob,token,initial_h,cache_h)
                pending[future]=(blob,token,hblob)
                return
            tick=perf_counter()
            current,cache=_prepare_node_update(i,parent,parent_cache,self.config)
            stats['prepare_seconds']+=perf_counter()-tick
            ancestors.append((i,current,cache,parent))
            for child in sorted(current.nodes[i].children,key=cost,reverse=True):
                prepare(child,current,cache)

        boundaries={}
        try:
            for child in sorted(old.nodes[root].children,key=cost,reverse=True):
                prepare(child,old,root_cache)
            while pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    blob,token,hblob=pending.pop(future)
                    result=future.result()
                    if result['need_h']:
                        # Executor APIs do not guarantee process affinity.
                        # A cache miss requests a fragment; the retry carries it
                        # regardless of which worker receives the retry.
                        stats['cache_misses']+=1
                        stats['task_bytes']+=len(blob)
                        stats['hamiltonian_bytes']+=len(hblob)
                        retry=self.subtree_executor.submit(_worker,blob,token,hblob,True)
                        pending[retry]=(blob,token,hblob)
                        continue
                    stats['result_bytes']+=len(result['payload'])
                    tick=perf_counter()
                    node,env,bc=_merge(new,result.pop('payload'))
                    stats['merge_seconds']+=perf_counter()-tick
                    boundaries[node]=(env,bc)
                    stats['tasks'].append(result)
                    elapsed=result['compute_seconds']
                    previous=self._worker_costs.get(node,elapsed)
                    self._worker_costs[node]=.5*previous+.5*elapsed
        except BaseException:
            for future in pending:future.cancel()
            raise

        def child_caches(i):
            environments=PartialTreeCachDict()
            changes=PartialTreeCachDict()
            for child in old.nodes[i].children:
                env,bc=boundaries[child]
                environments.add_entry(child,i,env)
                changes.add_entry(child,i,bc)
            return environments,changes
        tick=perf_counter()
        for i,current,cache,parent in reversed(ancestors):
            env,bc=child_caches(i)
            _,block,change=_finish_non_leaf_update(i,current,new,parent,cache,self.hamiltonian,
                self.time_step_size,self.config,env,bc,solver_options=self.solver_options)
            boundaries[i]=(block,change)
        env,_=child_caches(root)
        self.state=_finish_root_update(old,new,root_cache,self.hamiltonian,self.time_step_size,
            self.config,env,solver_options=self.solver_options)
        stats.update(initial_cache_seconds=initial_seconds,
            coordinator_finish_seconds=perf_counter()-tick,
            recursive_seconds=perf_counter()-started,cache_hamiltonian=cache_h)
        self.last_worker_stats=stats
        self._worker_h_revision=self._hamiltonian_revision()


class MPIBUG(_FastMixin,BUG):
    """Drop-in adaptive MPIBUG using compact CPU block messages."""

    def truncation(self):
        tick=perf_counter()
        result=super().truncation()
        self.last_worker_stats['truncation_seconds']=perf_counter()-tick
        return result


class MPIFixedBUG(_FastMixin,FixedBUG):
    """Drop-in fixed-rank MPIBUG using compact CPU block messages."""
