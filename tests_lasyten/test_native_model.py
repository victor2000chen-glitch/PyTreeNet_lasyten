"""Compare the new native kernels to independently contracted dense U1 models."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from .fake_native import backend, fixture, baseline, data, charge_sum, Symmetric, Charged, NONE, Space

HERE = Path(__file__).resolve().parent


@pytest.fixture
def native(monkeypatch):
    spec = importlib.util.spec_from_file_location('tested_native_bug_speedups', HERE.parent / 'pytreenet_lasyten/util/bug_native_speedups.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    lb, ct = backend()
    monkeypatch.setattr(helper, '_backend', lambda: lb)
    return helper, lb, ct


def fingerprint(array):
    n = array.native
    inv = n.invariant_part if isinstance(n, Charged) else n
    return (inv.data.copy(), tuple(inv.legs), list(n.labels), n.num_codomain_legs,
            n.charged_state.copy() if isinstance(n, Charged) else None, dict(array.__dict__))


def unchanged(array, before):
    now = fingerprint(array)
    np.testing.assert_array_equal(now[0], before[0])
    assert now[1:4] == before[1:4]
    if before[4] is not None:
        np.testing.assert_array_equal(now[4], before[4])
    assert now[5] == before[5]


def dense_operator(ham, hnode, envs, neighbours):
    v = len(neighbours)
    wlabels = list(range(2*v+2,3*v+2))
    hlabels = [None]*v
    for i,key in enumerate(neighbours):
        hlabels[hnode.neighbour_index(key)] = wlabels[i]
    args = [ham.native.data, hlabels+[2*v,2*v+1]]
    for i,key in enumerate(neighbours):
        args.extend([envs[key].native.data, [v+i,wlabels[i],i]])
    args.append(list(range(v))+[2*v]+list(range(v,2*v))+[2*v+1])
    result = np.einsum(*args, optimize='greedy')
    size = int(np.sqrt(result.size))
    return result.reshape(size,size)


@pytest.mark.parametrize('v', [0,1,2,3])
@pytest.mark.parametrize('p', [1,2])
@pytest.mark.parametrize('charged', [False,True])
@pytest.mark.parametrize('small_operator', [False,True])
def test_prepared_actions_match_dense_charge_conserving_operator(v,p,charged,small_operator,native):
    helper, lb, ct = native
    helper.configure(max_operator_elements=4096 if small_operator else 0)
    order = list(reversed([f'v{i}' for i in range(v)]))
    state,node,ham,hnode,cache,envs = fixture(lb,v,p,charge=1 if p==2 else 0,
                                          ham_order=order,charged=charged)
    inputs = [state,ham,*envs.values()]
    originals = [fingerprint(x) for x in inputs]
    expected = dense_operator(ham,hnode,envs,node.neighbouring_nodes()) @ data(state).reshape(-1)
    old = baseline(lb,state,node,ham,hnode,cache)
    np.testing.assert_allclose(data(old).reshape(-1), expected, rtol=2e-12,atol=2e-12)
    action = helper.prepare_action(state,node,ham,hnode,cache)
    assert action is not None
    if small_operator:
        assert action.local_operator is not None
    for key in ct.calls:
        ct.calls[key]=0
    result = action(0,state)
    np.testing.assert_allclose(data(result).reshape(-1), expected, rtol=2e-12,atol=2e-12)
    assert result.legs == state.legs and result.shape == state.shape
    assert result.native.labels == state.native.labels
    assert isinstance(result.native,Charged) == charged
    assert ct.calls['tdot'] == (1 if small_operator else v+1)
    assert ct.calls['wrappers'] == 0
    for x,before in zip(inputs,originals):
        unchanged(x,before)


@pytest.mark.parametrize('guard,limit', [('max_operator_elements',35),('max_intermediate_elements',35)])
def test_size_guard_precedes_operator_allocations_and_keeps_matrix_free_action(guard,limit,native):
    helper,lb,ct=native
    helper.configure(**{guard:limit})
    state,node,ham,hnode,cache,_=fixture(lb,1,2,charge=1)
    ct.calls['tdot']=0
    action=helper.prepare_action(state,node,ham,hnode,cache)
    assert action.local_operator is None and ct.calls['tdot']==0
    result=action(0,state)
    expected=baseline(lb,state,node,ham,hnode,cache)
    np.testing.assert_allclose(data(result),data(expected),atol=1e-12)


def test_exact_size_bound_is_allowed_and_reported(native):
    helper,lb,ct=native
    helper.configure(max_operator_elements=36,max_intermediate_elements=36)
    args=fixture(lb,1,2,charge=1)
    action=helper.prepare_action(*args[:5])
    assert action.local_operator is not None
    assert helper.stats()['largest_precontract_intermediate_elements']==36


def test_each_local_solve_gets_new_environment_coefficients_without_mutating_old_action(native):
    helper,lb,ct=native
    args=fixture(lb,2,2,charge=1)
    state,node,ham,hnode,cache,envs=args
    old=helper.prepare_action(*args[:5])
    expected=data(old(0,state)).copy()
    envs['v0']=lb.LasyTenArray(envs['v0'].native*2)
    new=helper.prepare_action(*args[:5])
    np.testing.assert_allclose(data(new(0,state)),2*expected,atol=2e-12)
    np.testing.assert_allclose(data(old(0,state)),expected,atol=2e-12)


def test_repeated_matvecs_reduce_operations_and_preserve_cache_labels(native):
    helper,lb,ct=native
    args=fixture(lb,3,2,charge=1)
    state,node,ham,hnode,cache,envs=args
    original_labels=[list(x.native.labels) for x in [ham,*envs.values()]]
    ct.calls.update({key:0 for key in ct.calls})
    for _ in range(7):baseline(lb,state,node,ham,hnode,cache)
    before=dict(ct.calls)
    ct.calls.update({key:0 for key in ct.calls})
    action=helper.prepare_action(*args[:5])
    for _ in range(7):action(0,state)
    assert before['tdot']==28 and ct.calls['tdot']==10
    assert ct.calls['wrappers']==0 and ct.calls['permute']<before['permute']
    assert [x.native.labels for x in [ham,*envs.values()]]==original_labels


@pytest.mark.parametrize('charged',[False,True])
@pytest.mark.parametrize('fast',[False,True])
def test_fused_conjugate_preserves_values_spaces_labels_and_inputs(charged,fast,native):
    helper,lb,ct=native
    state,*_=fixture(lb,2,2,charge=1,charged=charged)
    before=fingerprint(state)
    baseline_conjugate=lb.transpose(lb.LasyTenArray(ct.dagger(state.native)),tuple(reversed(range(state.ndim))))
    helper.configure(fast_conjugate=fast)
    ct.calls['permute']=0
    actual=helper.conjugate(state)
    np.testing.assert_array_equal(data(actual),data(baseline_conjugate))
    assert actual.legs==baseline_conjugate.legs and actual.native.labels==baseline_conjugate.native.labels
    assert actual.native.num_codomain_legs==actual.ndim
    if fast:
        assert ct.calls['permute']==1
    unchanged(state,before)


def test_no_symmetry_and_numpy_paths_retain_constructor_and_fallback(native):
    helper,lb,ct=native
    legs=[Space((0,0),'a'),Space((0,0,0),'b')]
    array=lb.LasyTenArray(Symmetric(np.arange(6).reshape(2,3)+1j,legs,symmetry=NONE))
    ct.calls['wrappers']=0
    result=helper.conjugate(array)
    np.testing.assert_array_equal(data(result),data(array).conj())
    assert ct.calls['wrappers']==1
    assert helper.prepare_action(np.eye(2),None,np.eye(2),None,None) is None
    assert helper.prepare_action(array,None,array,None,None) is None


@pytest.mark.parametrize('small_operator',[False,True])
def test_multiple_physical_legs_keep_output_input_order(small_operator,native):
    from types import SimpleNamespace
    from .fake_native import random_invariant
    helper,lb,ct=native
    helper.configure(max_operator_elements=4096 if small_operator else 0)
    rng=np.random.default_rng(83)
    v=Space((-1,0,1),'v')
    w=Space((-2,-1,0,1,2),'w')
    p,q=Space((0,1),'p'),Space((0,1),'q')
    state=lb.LasyTenArray(Charged(random_invariant(rng,[v,p,q,Space((1,),'N',True)]),[1.+0j]))
    environment=lb.LasyTenArray(random_invariant(rng,[v.dual,w,v]))
    ham=lb.LasyTenArray(random_invariant(rng,[w.dual,p,q,p.dual,q.dual]))
    node=SimpleNamespace(identifier='r',neighbouring_nodes=lambda:['v'],nopen_legs=lambda:2)
    hnode=SimpleNamespace(neighbour_index=lambda key:0,nneighbours=lambda:1,nopen_legs=lambda:4)
    cache=SimpleNamespace(get_entry=lambda *args:environment)
    H=np.einsum('awb,wpqPQ->bpqaPQ',environment.native.data,ham.native.data).reshape(12,12)
    expected=H@data(state).reshape(-1)
    result=helper.prepare_action(state,node,ham,hnode,cache)(0,state)
    np.testing.assert_allclose(data(result).reshape(-1),expected,atol=2e-12)


def test_zero_hidden_charge_amplitude_never_requires_charged_addition(native):
    helper,lb,ct=native
    state,*_=fixture(lb,1,2,charge=1)
    state.native=Charged(state.native.invariant_part,[0j])
    result=helper.native_sum(state,[.3,-.2],[state,state],include_base=False)
    np.testing.assert_array_equal(data(result),np.zeros_like(data(state)))


@pytest.mark.parametrize('charged',[False,True])
@pytest.mark.parametrize('include_base',[False,True])
def test_rk_sums_scaling_zero_and_nonunit_hidden_amplitude(charged,include_base,native):
    helper,lb,ct=native
    state,*_=fixture(lb,1,2,charge=1,charged=charged)
    x=helper.native_scale(state,.3+.2j)
    y=helper.native_scale(state,-.7j)
    before=fingerprint(state)
    result=helper.native_sum(state,[.2,-.4,0],[x,y,state],step=.7,include_base=include_base)
    expected=(data(state) if include_base else 0)+.7*(.2*data(x)-.4*data(y))
    np.testing.assert_allclose(data(result),expected,atol=1e-13)
    zero=helper.native_sum(state,[0],[state],include_base=False)
    np.testing.assert_array_equal(data(zero),np.zeros_like(data(state)))
    assert result.legs==state.legs and zero.legs==state.legs
    unchanged(state,before)


@pytest.mark.parametrize('method',['RK23','RK45'])
@pytest.mark.parametrize('duration',[.01,-.01,.01-.002j])
@pytest.mark.parametrize('small_operator',[False,True])
def test_prepared_native_propagation_matches_independent_matrix_exponential(method,duration,small_operator,native,monkeypatch):
    import importlib,sys
    te=importlib.import_module('pytreenet_lasyten.time_evolution.time_evolution')
    helper,lb,ct=native
    helper.configure(max_operator_elements=4096 if small_operator else 0)
    state,node,ham,hnode,cache,envs=fixture(lb,2,2,charge=1,ham_order=['v1','v0'])
    # Keep the time scale mild while testing a general complex, non-Hermitian map.
    ham.native=ham.native*.03
    H=dense_operator(ham,hnode,envs,node.neighbouring_nodes())
    initial=data(state).copy()
    action=helper.prepare_action(state,node,ham,hnode,cache)
    monkeypatch.setattr(te,'lb',lb)
    monkeypatch.setattr(te,'is_lasyten_tensor',lambda a:isinstance(a,lb.LasyTenArray))
    monkeypatch.setattr(te,'_native_scale',helper.native_scale)
    monkeypatch.setattr(te,'_native_sum',helper.native_sum)
    stats={}
    result=te._native_time_evolve_action(state,action,duration,True,method,
        first_step=abs(duration),rtol=1e-10,atol=1e-12,stats=stats)
    expected=(expm(-1j*duration*H)@initial.reshape(-1)).reshape(initial.shape)
    np.testing.assert_allclose(data(result),expected,atol=2e-9,rtol=2e-9)
    np.testing.assert_array_equal(data(state),initial)
    assert stats['success'] and result.legs==state.legs


def test_configuration_and_first_step_override(native):
    helper,_,_=native
    assert helper.propose_first_step(.1,None)==.1
    assert helper.propose_first_step(.1,.02)==.02
    assert helper.propose_first_step(0,None) is None
    helper.configure(full_first_step=False)
    assert helper.propose_first_step(.1,None) is None
    with pytest.raises(ValueError):helper.configure(max_operator_elements=-1)
    with pytest.raises(TypeError):helper.configure(fast_wrapping=1)
    with pytest.raises(ValueError):helper.configure(unknown=True)
    helper.reset_stats()
    assert helper.stats()=={}


def test_installed_effective_hook_and_default_first_step_execute_native_prepared_path(native,monkeypatch):
    import ast,importlib,sys
    te=importlib.import_module('pytreenet_lasyten.time_evolution.time_evolution')
    effective=importlib.import_module('pytreenet_lasyten.time_evolution.time_evo_util.effective_time_evolution')
    helper,lb,ct=native
    state,node,ham,hnode,cache,envs=fixture(lb,2,2,charge=1)
    ham.native=ham.native*.01
    initial=data(state).copy()
    H=dense_operator(ham,hnode,envs,node.neighbouring_nodes())
    monkeypatch.setattr(te,'lb',lb)
    monkeypatch.setattr(te,'is_lasyten_tensor',lambda a:isinstance(a,lb.LasyTenArray))
    monkeypatch.setattr(te,'_native_scale',helper.native_scale)
    monkeypatch.setattr(te,'_native_sum',helper.native_sum)
    def forbidden(*args,**kwargs):
        raise AssertionError('Prepared native hook did not replace the generic action.')
    monkeypatch.setattr(effective,'contract_ket_ham_with_envs',forbidden)
    monkeypatch.setattr(te,'_bug_native_speedups',helper)
    monkeypatch.setattr(effective,'_bug_native_speedups',helper)
    local=effective.effective_single_site_evolution
    stats={}
    result=local(state,node,ham,hnode,.01,cache,mode=te.TimeEvoMode.RK45,
                 rtol=1e-10,atol=1e-12,stats=stats)
    np.testing.assert_allclose(data(result).reshape(-1),expm(-1j*.01*H)@initial.reshape(-1),atol=2e-10)
    assert helper.stats()['precontracted_actions']==1
    assert stats['nfev']==7 and stats['n_rejected']==0
