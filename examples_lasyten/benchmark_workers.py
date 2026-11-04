#!/usr/bin/env python3
"""Compare identical serial/compact-worker BUG trajectories and report phase/bytes.

NumPy tests use a tiny branching tree. Native tests use the existing HK helper,
the same physical model and the installed U1/U1xU1 constructors. CPU only.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
import multiprocessing
import os
from pathlib import Path
import sys
from time import perf_counter


def problem(backend,nb,chi,helper_dir,seed=11):
    import numpy as np
    from pytreenet_lasyten.core.node import Node
    from pytreenet_lasyten.ttns.ttns import TreeTensorNetworkState as TTNS
    from pytreenet_lasyten.ttno.ttno_class import TreeTensorNetworkOperator as TTNO
    from pytreenet_lasyten.operators import Hamiltonian,TensorProduct
    if backend!='numpy':
        sys.path.insert(0,str(Path(helper_dir).expanduser().resolve()))
        import Complex_bug_impurity_solver_lasyten as model
        from utils import Discretisation_function
        from pytreenet_lasyten.util.hk_u1_model import build_u1_t3ns
        from pytreenet_lasyten.util.u1_ttno import u1_ttno_from_hamiltonian
        state=build_u1_t3ns(norb=3,nb=nb,virtual_bond_dimension=chi,
            bath_bond_dimension=chi,symmetry=backend,seed=seed)
        v,e=Discretisation_function(1.,nb)
        symbolic=model.construct_Hubbard_Kanamori_hamiltonian(3,2.,1.6,.2,4.,0.)
        model.add_bath_interaction(symbolic,n_bath=nb,v=np.asarray(v,complex),e=np.asarray(e,float))
        return state,u1_ttno_from_hamiltonian(symbolic,state)
    rng=np.random.default_rng(seed)
    random=lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
    state=TTNS()
    state.add_root(Node(identifier='r'),random((2,2,1)))
    state.add_child_to_parent(Node(identifier='a'),random((2,2)),0,'r',0)
    state.add_child_to_parent(Node(identifier='s'),random((2,2,2,1)),0,'r',1)
    state.add_child_to_parent(Node(identifier='b'),random((2,2)),0,'s',1)
    state.add_child_to_parent(Node(identifier='c'),random((2,2)),0,'s',2)
    state.canonical_form('r');state.normalise()
    down=np.array([[0,1],[0,0]],complex)
    terms=[(.31,{'a':'up','b':'down'}),(.31,{'a':'down','b':'up'}),
           (-.17,{'b':'up','c':'down'}),(-.17,{'b':'down','c':'up'}),
           (.23,{'a':'n'}),(-.11,{'b':'n'}),(.07,{'c':'n'}),(.19,{'a':'n','c':'n'})]
    h=Hamiltonian([(Fraction(1),f't{i}',TensorProduct(term)) for i,(_,term) in enumerate(terms)],
        conversion_dictionary=dict(up=down.T,down=down,n=np.diag([0.,1.]),I1=np.eye(1),I2=np.eye(2)),
        coeffs_mapping={f't{i}':c for i,(c,_) in enumerate(terms)})
    return state,TTNO.from_hamiltonian(h,state)


def compare(left,right):
    import numpy as np
    nl=left.scalar_product(use_orthogonal_center=False).real
    nr=right.scalar_product(use_orthogonal_center=False).real
    overlap=left.scalar_product(right,use_orthogonal_center=False)
    phase_sensitive=overlap/np.sqrt(nl*nr)
    fidelity=abs(phase_sensitive)**2
    np.testing.assert_allclose(nl,nr,rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(phase_sensitive,1.,rtol=1e-8,atol=1e-9)
    return dict(fidelity=float(fidelity),overlap_phase=float(np.angle(overlap)),norm_squared=float(nl))


def run(executor,args):
    from pytreenet_lasyten.time_evolution.bug import BUG,BUGConfig
    from pytreenet_lasyten.time_evolution._mpi_bug_fast import MPIBUG
    from pytreenet_lasyten.time_evolution.time_evolution import TimeEvoMode
    state,h=problem(args.backend,args.nb,args.chi,args.helper_dir,args.seed)
    config=BUGConfig(max_bond_dim=args.chi,rel_tol=1e-10,total_tol=1e-12,
                     time_evo_mode=TimeEvoMode.RK45)
    options=dict(rtol=1e-8,atol=1e-10)
    classes=[('serial',BUG),('compact_workers',MPIBUG)]
    if args.include_legacy:
        from pytreenet_lasyten.time_evolution.mpi_bug import LegacyMPIBUG
        classes.insert(1,('legacy_workers',LegacyMPIBUG))
    final={};report={}
    for name,cls in classes:
        kwargs={} if name=='serial' else dict(executor=executor,subtree_tasks=args.subtree_tasks)
        engine=cls(deepcopy(state),h,args.dt,args.dt*(args.steps+args.warmup),{},
                   config=deepcopy(config),solver_options=options,**kwargs)
        for _ in range(args.warmup):engine.run_one_time_step()
        details=[]
        started=perf_counter()
        for _ in range(args.steps):
            engine.run_one_time_step()
            if name=='compact_workers':details.append(engine.last_worker_stats)
        elapsed=perf_counter()-started
        final[name]=engine.state
        report[name]=dict(seconds_per_step=elapsed/args.steps,total_seconds=elapsed,
                          worker_steps=details)
        print(f'{name}: {elapsed/args.steps:.6f} seconds/step',flush=True)
    for name in final:
        if name!='serial':report[name]['agreement']=compare(final[name],final['serial'])
    report['settings']=vars(args).copy()
    report['settings']['output']=str(args.output)
    report['compact_workers']['speedup_over_serial']=(report['serial']['seconds_per_step']/report['compact_workers']['seconds_per_step'])
    if 'legacy_workers' in report:
        report['compact_workers']['speedup_over_legacy']=report['legacy_workers']['seconds_per_step']/report['compact_workers']['seconds_per_step']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Serial/worker norm, phase and fidelity checks passed.',flush=True)
    print('Saved:',args.output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executor',choices=['process','mpi'],default='process')
    p.add_argument('--workers',type=int,default=3,help='Used only for process executor; MPI uses its allocated worker ranks.')
    p.add_argument('--backend',choices=['numpy','u1','u1xu1'],default='numpy')
    p.add_argument('--nb',type=int,default=3)
    p.add_argument('--chi',type=int,default=8)
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--subtree-tasks',type=int,default=3)
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--dt',type=float,default=.01)
    p.add_argument('--helper-dir',default=os.environ.get('HK_HELPER_DIR',str(Path.home()/'DMFT')))
    p.add_argument('--include-legacy',action='store_true',help='Requires the old native HDF5 codec to work, if using native tensors.')
    p.add_argument('--output',type=Path,default=Path('worker_benchmark.json'))
    args=p.parse_args()
    if min(args.workers,args.nb,args.chi,args.subtree_tasks,args.steps,args.threads)<=0 or args.warmup<0 or args.dt<=0:
        p.error('Counts/time step must be positive; warmup may be zero.')
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name]=str(args.threads)
    if args.executor=='mpi':
        from mpi4py import MPI
        from mpi4py.futures import MPICommExecutor
        if MPI.COMM_WORLD.Get_size()<2:raise ValueError('MPI mode needs at least two ranks.')
        with MPICommExecutor(MPI.COMM_WORLD,root=0) as executor:
            if executor is not None:run(executor,args)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as executor:
            run(executor,args)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        traceback.print_exc()
        if '--executor' in sys.argv and 'mpi' in sys.argv:
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
        raise
