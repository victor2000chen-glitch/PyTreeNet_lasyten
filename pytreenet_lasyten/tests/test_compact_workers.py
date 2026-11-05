"""Actual NumPy BUG kernels versus serial; native coverage is a separate pilot."""
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,Future
from copy import deepcopy
import multiprocessing
import pickle
import unittest

import numpy as np
from examples.benchmark_workers import problem,compare
from pytreenet_lasyten.time_evolution.bug import BUG,BUGConfig
from pytreenet_lasyten.time_evolution.fixed_bug import FixedBUG,FixedBUGConfig
from pytreenet_lasyten.time_evolution._mpi_bug_fast import MPIBUG,MPIFixedBUG,_scope,_freeze_task
from pytreenet_lasyten.time_evolution._mpi_serialization import encode_task
from pytreenet_lasyten.time_evolution._mpi_block_wire import pack_view,restore_view
from pytreenet_lasyten.time_evolution.time_evolution import TimeEvoMode
from pytreenet_lasyten.contractions.sandwich_caching import SandwichCache


class WorkerTests(unittest.TestCase):
    def test_h_fragments_share_arrays_but_not_metadata(self):
        import weakref
        from pytreenet_lasyten.ttno.ttno_class import TreeTensorNetworkOperator
        state,h=problem('numpy',1,4,'unused')
        cache=weakref.WeakValueDictionary()
        a,am=_scope(state,'b');b,bm=_scope(state,'c')
        left=restore_view(TreeTensorNetworkOperator,pack_view(h,a,am),cache,'epoch')
        right=restore_view(TreeTensorNetworkOperator,pack_view(h,b,bm),cache,'epoch')
        self.assertIs(left.tensors.data['s'],right.tensors.data['s'])
        self.assertIsNot(left.nodes['s'],right.nodes['s'])

    def test_long_three_arm_tree_and_payload_reduction(self):
        from pytreenet_lasyten.core.node import Node
        from pytreenet_lasyten.ttns.ttns import TreeTensorNetworkState as TTNS
        from pytreenet_lasyten.ttno.ttno_class import TreeTensorNetworkOperator as TTNO
        rng=np.random.default_rng(23)
        rand=lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
        state,operator=TTNS(),TTNO()
        state.add_root(Node(identifier='root'),rand((4,4,4,1)))
        operator.add_root(Node(identifier='root'),np.ones((1,1,1,1,1),complex))
        for arm in range(3):
            for j in range(12):
                i=f'arm{arm}_{j}'
                parent='root' if j==0 else f'arm{arm}_{j-1}'
                axis=arm if j==0 else 1
                state.add_child_to_parent(Node(identifier=i),rand((4,4,2) if j<11 else (4,2)),0,parent,axis)
                operator.add_child_to_parent(Node(identifier=i),np.eye(2).reshape((1,1,2,2) if j<11 else (1,2,2)),0,parent,axis)
        state.canonical_form('root');state.normalise()
        cfg=BUGConfig(max_bond_dim=4,time_evo_mode=TimeEvoMode.RK45)
        cache=SandwichCache.init_cache_but_one(state,operator,'root')
        old=encode_task('arm0_0',state,state,cache,operator,.01,cfg,{})
        new,ids,meta=_freeze_task('arm0_0',state,state,cache,.01,cfg,{})
        # Cold H transfer is separate; these bytes compare per-step state/cache
        # messages, which are the recurring cost after caches warm up.
        self.assertLess(len(new),.6*len(old['payload']))
        print(f'Long-arm task state/cache payload: old={len(old["payload"])} bytes, compact={len(new)} bytes')
        with ThreadPoolExecutor(3) as ex:
            parallel=MPIBUG(state,operator,.01,.01,{},config=deepcopy(cfg),executor=ex,subtree_tasks=3)
            serial=BUG(state,operator,.01,.01,{},config=deepcopy(cfg))
            parallel.run_one_time_step();serial.run_one_time_step()
            compare(parallel.state,serial.state)

    def test_multiple_frontiers_deep_modes_and_fixed_rank(self):
        for fixed in (False,True):
            for deep in (False,True):
                for tasks in (1,2,3,8):
                    with self.subTest(fixed=fixed,deep=deep,tasks=tasks):
                        state,h=problem('numpy',1,4,'unused')
                        original=deepcopy(state)
                        config=(FixedBUGConfig if fixed else BUGConfig)(
                            deep=deep,time_evo_mode=TimeEvoMode.RK45,
                            **({} if fixed else dict(max_bond_dim=4,rel_tol=0,total_tol=0)))
                        options=dict(rtol=1e-11,atol=1e-13)
                        reference=(FixedBUG if fixed else BUG)(state,h,.01,.02,{},config=deepcopy(config),solver_options=options)
                        with ThreadPoolExecutor(2) as ex:
                            parallel=(MPIFixedBUG if fixed else MPIBUG)(state,h,.01,.02,{},
                                config=deepcopy(config),solver_options=options,executor=ex,subtree_tasks=tasks)
                            for step in range(2):
                                parallel.run_one_time_step();reference.run_one_time_step()
                                compare(parallel.state,reference.state)
                                if step:
                                    self.assertEqual(parallel.last_worker_stats['cache_misses'],0)
                                    self.assertEqual(parallel.last_worker_stats['hamiltonian_bytes'],0)
                                self.assertEqual(parallel.state.orthogonality_center_id,state.root_id)
                                self.assertFalse(any(i.endswith('_basis_change_tensor') for i in parallel.state.nodes))
                        compare(state,original)

    def test_spawn_and_explicit_invalidation(self):
        state,h=problem('numpy',1,4,'unused')
        config=BUGConfig(max_bond_dim=4,time_evo_mode=TimeEvoMode.RK45)
        with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn')) as ex:
            engine=MPIBUG(state,h,.01,.03,{},config=deepcopy(config),executor=ex,subtree_tasks=3)
            oracle=BUG(state,h,.01,.03,{},config=deepcopy(config))
            for step in range(3):
                if step==1:
                    # In-place operator edit; the public invalidation API is mandatory.
                    h.tensors.data[h.root_id][...]*=1.17
                    engine.invalidate_worker_hamiltonian()
                elif step==2:
                    # Ordinary tensor replacement is detected automatically.
                    h.tensors[h.root_id]=h.tensors[h.root_id]*.91
                engine.run_one_time_step();oracle.run_one_time_step()
                compare(engine.state,oracle.state)
            self.assertTrue(engine.last_worker_stats['tasks'])

    def test_dynamic_h_and_cache_disabled(self):
        for dynamic in (False,True):
            state,h=problem('numpy',1,4,'unused')
            cfg=BUGConfig(max_bond_dim=4,time_evo_mode=TimeEvoMode.RK45)
            with ThreadPoolExecutor(2) as ex:
                p=MPIBUG(state,h,.01,.02,{},config=deepcopy(cfg),executor=ex,
                    subtree_tasks=3,cache_hamiltonian=dynamic)
                b=BUG(state,h,.01,.02,{},config=deepcopy(cfg))
                # Exercise cache policy independently of the optional abstract
                # time-dependent-TTNO wrapper, using the same fixed H oracle.
                p.config.time_dep=b.config.time_dep=dynamic
                for _ in range(2):p.run_one_time_step();b.run_one_time_step();compare(p.state,b.state)
                self.assertFalse(p.last_worker_stats['cache_hamiltonian'])
                self.assertEqual(p.last_worker_stats['cache_misses'],0)
                self.assertGreater(p.last_worker_stats['hamiltonian_bytes'],0)

    def test_scope_and_frozen_message(self):
        state,h=problem('numpy',1,4,'unused')
        cache=SandwichCache.init_cache_but_one(state,h,state.root_id)
        cfg=BUGConfig()
        message,ids,metadata=_freeze_task('b',state,state,cache,.01,cfg,{})
        job=pickle.loads(message)
        data_ids={r['identifier'] for r in job['parent']['records'] if 'tensor' in r}
        self.assertEqual(data_ids,{'s','b'})
        self.assertIn('r',metadata)
        self.assertIn('c',metadata)
        self.assertNotIn('a',metadata)
        before=next(r['tensor']['data'].copy() for r in job['parent']['records'] if r['identifier']=='b')
        state.tensors['b']*=2
        restored=pickle.loads(message)
        after=next(r['tensor']['data'] for r in restored['parent']['records'] if r['identifier']=='b')
        np.testing.assert_array_equal(before,after)
        self.assertTrue(all(b in ids for a,b,v in job['environments']))

    def test_failure_keeps_original_state(self):
        class Fails:
            def submit(self,*a,**kw):
                future=Future();future.set_exception(RuntimeError('worker failure'));return future
        state,h=problem('numpy',1,4,'unused')
        engine=MPIBUG(state,h,.01,.01,{},executor=Fails(),subtree_tasks=3)
        before=engine.state
        with self.assertRaisesRegex(RuntimeError,'worker failure'):engine.run_one_time_step()
        self.assertIs(before,engine.state)
        compare(before,state)


if __name__=='__main__':unittest.main(verbosity=2)
