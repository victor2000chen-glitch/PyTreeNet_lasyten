"""Native checks for LRZ: no fake LASyTen objects and no HDF5 requirement."""
from copy import deepcopy
import time

import numpy as np
from scipy.linalg import expm


def merge_to_single_charge(network,cls):
    from pytreenet_lasyten.util import lasyten_backend as lb
    from pytreenet_lasyten.util.u1_ttno import _insert
    from pytreenet_lasyten.util.abelian_charges import fixed_charge
    tensors={}
    ct=lb._lasyten()
    for i in network.nodes:
        a=network.tensors[i];spaces=[]
        for leg in a.legs:
            positive=leg.dual if leg.is_dual else leg
            numbers=np.asarray(positive.sectors_of_basis).sum(axis=1).astype(np.int64)
            spaces.append(lb.u1_space(numbers,dual=leg.is_dual))
        charge=None
        if isinstance(a.native,ct.ChargedTensor):
            charge=lb.u1_space([sum(fixed_charge(network))])
        tensors[i]=lb.from_numpy(a.to_numpy(),spaces=spaces,charge=charge,tol=1e-11)
    result=_insert(network,tensors,cls)
    result.orthogonality_center_id=network.orthogonality_center_id
    if hasattr(network,'u1_info'):
        result.u1_info=dict(total_particles=sum(fixed_charge(network)),physical_nodes=network.u1_info['physical_nodes'])
    if hasattr(network,'u1_operator_info'):
        result.u1_operator_info=dict(total_charge=0)
    return result


def main():
    from examples.native_symmetry_problem import ORDER,make_template,make_hamiltonian,native_vector,fermion_matrix
    import lasyten as ct
    from pytreenet_lasyten.ttns.ttns import TreeTensorNetworkState as TTNS
    from pytreenet_lasyten.ttno.ttno_class import TreeTensorNetworkOperator as TTNO
    from pytreenet_lasyten.util.u1u1_t3ns import random_u1u1_state_like,check_u1u1_state
    from pytreenet_lasyten.util.u1_ttno import u1_ttno_from_hamiltonian
    from pytreenet_lasyten.ttno.state_diagram import TTNOFinder
    from pytreenet_lasyten.util.abelian_charges import fixed_charge,check_native_charges
    from pytreenet_lasyten.util.u1_excitations import create_impurity_excitations
    from pytreenet_lasyten.ttns.ttns_ttno.application import apply_ttno_to_ttns,ApplicationMethod
    from pytreenet_lasyten.util.tensor_splitting import SplitMode,SVDParameters
    from pytreenet_lasyten.time_evolution.bug import BUG,BUGConfig
    from pytreenet_lasyten.time_evolution.time_evolution import TimeEvoMode
    from pytreenet_lasyten.dmrg.dmrg_sparse import DMRGAlgorithm_sparse
    start=time.perf_counter()
    state=random_u1u1_state_like(make_template(),total_particles_by_spin=(1,1),max_bond_dim=4,seed=11)
    assert fixed_charge(state)==(1,1)
    check_u1u1_state(state)
    before=native_vector(state)
    for node in state.nodes:
        state.move_orthogonalization_center(node,mode=SplitMode.REDUCED)
    state.ensure_root_orth_center(mode=SplitMode.REDUCED)
    np.testing.assert_allclose(native_vector(state),before,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(state.scalar_product(),np.vdot(before,before),atol=1e-10)
    h,dense=make_hamiltonian()
    operators=[]
    for method in (TTNOFinder.BASE,TTNOFinder.SGE):
        H=u1_ttno_from_hamiltonian(h,state,method=method)
        check_native_charges(state,H)
        result=apply_ttno_to_ttns(state,H,ApplicationMethod.DIRECT)
        np.testing.assert_allclose(native_vector(result),dense@before,atol=2e-10,rtol=2e-10)
        operators.append(H)
    H=operators[-1]
    for index in (0,1):
        removal,addition,zr,za=create_impurity_excitations(state,impurity_node=ORDER[index],parity_nodes=ORDER[:index])
        expected=list((1,1));expected[index]-=1
        assert fixed_charge(removal)==tuple(expected)
        expected[index]+=2
        assert fixed_charge(addition)==tuple(expected)
        np.testing.assert_allclose(native_vector(removal)*np.sqrt(zr),fermion_matrix(index)@before,atol=2e-10)
        np.testing.assert_allclose(native_vector(addition)*np.sqrt(za),fermion_matrix(index,True)@before,atol=2e-10)
        np.testing.assert_allclose(zr+za,1,atol=2e-10)
    legacy=merge_to_single_charge(state,TTNS)
    legacy_H=merge_to_single_charge(H,TTNO)
    np.testing.assert_allclose(native_vector(legacy),before,atol=1e-11)
    dt=.001
    config=BUGConfig(max_bond_dim=16,rel_tol=0,total_tol=0,time_evo_mode=TimeEvoMode.RK45)
    options=dict(rtol=1e-10,atol=1e-12)
    two=BUG(deepcopy(state),H,dt,dt,{},config=deepcopy(config),solver_options=options)
    one=BUG(deepcopy(legacy),legacy_H,dt,dt,{},config=deepcopy(config),solver_options=options)
    two.run_one_time_step();one.run_one_time_step()
    two.state.ensure_root_orth_center(mode=SplitMode.REDUCED)
    assert check_native_charges(two.state)==(1,1)
    np.testing.assert_allclose(native_vector(two.state),native_vector(one.state),atol=1e-7,rtol=1e-7)
    np.testing.assert_allclose(native_vector(two.state),expm(-1j*dt*dense)@before,atol=1e-6,rtol=1e-6)
    dmrg=DMRGAlgorithm_sparse(initial_state=deepcopy(state),hamiltonian=H,num_sweeps=2,max_iter=100,
        svd_params=SVDParameters(max_bond_dim=16,rel_tol=1e-12,total_tol=1e-14),site='one-site')
    dmrg.run()
    dmrg.state.ensure_root_orth_center(mode=SplitMode.REDUCED)
    assert check_native_charges(dmrg.state)==(1,1)
    v=native_vector(dmrg.state);energy=np.vdot(v,dense@v)/np.vdot(v,v)
    indices=[i for i in range(16) if sum((i>>(3-k))&1 for k in (0,2))==1 and sum((i>>(3-k))&1 for k in (1,3))==1]
    exact=np.linalg.eigvalsh(dense[np.ix_(indices,indices)])[0]
    np.testing.assert_allclose(energy,exact,atol=1e-8)
    print(f'PASS native U1xU1: QR, overlaps, BASE/SGE Hpsi, both-spin JW excitations, legacy agreement, BUG/expm and DMRG/ED ({time.perf_counter()-start:.2f}s).')


if __name__=='__main__':
    main()
