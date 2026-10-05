import numpy as np
import pytest
from scripts.tau_bias_sampling import bin_indices, sampling_scan


def example():
    u = np.tile(np.linspace(-.99,.99,80)[:,None], (1,9))
    return u, {'full':u, 'matched':u}, np.ones(80)


def test_edges():
    assert bin_indices(np.array([-1., -.5, 0., .5, 1.])).tolist() == [0,1,3,4,5]
    with pytest.raises(ValueError): bin_indices(np.array([1.1]))


def test_perfect_models_close_and_pair_exactly():
    u,t,w = example()
    r = sampling_scan(u,t,{'pretrain':u,'dgpo':u},w,repeats=3)
    assert len(r['cases']) == 19
    for case in r['cases']:
        assert all(m['mean'] == 0 for m in case['metrics'].values())
    assert np.allclose(r['cases'][0]['delta_C']['dgpo'],0)


def test_nonresponsive_model_and_improvement():
    u,t,w = example()
    r = sampling_scan(u,t,{'pretrain':np.zeros_like(u),'dgpo':u},w,repeats=5)
    case = r['cases'][2]
    assert case['metrics']['full/pretrain/response']['mean'] > .1
    assert case['metrics']['full/dgpo_minus_pretrain/response']['mean'] < -.1


def test_original_mc_weights_retained():
    u,t,w = example(); w[:40] = 3
    r = sampling_scan(u,t,{'dgpo':u},w,repeats=2)
    assert np.allclose(np.array(r['nominal_full_panel_C']['dgpo']).ravel(),np.average(u,axis=0,weights=w))


def test_raw_not_ema():
    import torch
    from scripts.run_tau_cij_sampling_bias import load_raw
    model = torch.nn.Linear(2,1)
    raw = {k:torch.ones_like(v) for k,v in model.state_dict().items()}
    load_raw(model,{'state_dict':{'model.'+k:v for k,v in raw.items()},'ema_state_dict':{k:v*7 for k,v in raw.items()}})
    assert all(torch.equal(v,raw[k]) for k,v in model.state_dict().items())
    with pytest.raises(ValueError): load_raw(model,{'state_dict':{}})


def test_target_solver_and_zero_mass():
    from scripts.tau_bias_sampling import target_probabilities
    z=np.linspace(-2,2,101); w=np.linspace(1,3,101); w[0]=0
    for target in [-.5,0,.5]:
        p,lam=target_probabilities(z,w,target)
        assert p[0]==0 and np.isclose(p.sum(),1)
        assert abs(p@z-target)<1e-8
    p,_=target_probabilities(z,w,np.average(z,weights=w))
    assert np.allclose(p,w/w.sum())
    with pytest.raises(ValueError): target_probabilities(z,w,3)


def test_target_scan_unit_weights_and_shared_B():
    u,t,w=example(); w[:40]=3
    moments={k:np.tile(u[:,:1],(1,6)) for k in ['truth/full','truth/matched','pretrain','dgpo']}
    r=sampling_scan(u,t,{'pretrain':u,'dgpo':u},w,repeats=30,targets=[-.5,0,.5],moments=moments)
    assert len(r['cases'])==28
    for c in r['cases']:
        assert all(m['mean']==0 for m in c['metrics'].values())
        assert c['B']['dgpo']==c['B']['truth/full']
        if c['requested_target'] is not None:
            assert abs(c['expected_target']-c['requested_target'])<1e-8
            # Would fail if original MC weights were multiplied a second time.
            assert abs(np.asarray(c['C']['truth/full']).flat[0]-c['requested_target'])<.06
    assert np.allclose(r['cases'][0]['delta_C']['dgpo'],0)


def test_target_unsupported_not_silently_clipped():
    u,t,w=example()
    r=sampling_scan(u,t,{'dgpo':u},w,repeats=2,targets=[2.])
    assert sum(c.get('status')=='unsupported' for c in r['cases'])==9


def test_moments_signed_kappa():
    from scripts.tau_bias_sampling import polarization_moments
    a=np.array([[1.,0,0]]); b=np.array([[0,1.,0]])
    assert np.allclose(polarization_moments(a,b,np.array([[1.,-.5]])),[[3,0,0,0,-6,0]])


def test_plots_targeted(tmp_path):
    from scripts.tau_bias_plots import direct_plots
    u,t,w=example()
    moments={k:np.zeros((80,6)) for k in ['truth/full','truth/matched','pretrain','dgpo']}
    r=sampling_scan(u,t,{'pretrain':u*.5,'dgpo':u},w,repeats=3,targets=[-.5,0,.5],moments=moments)
    paths=direct_plots(r,tmp_path)
    assert len(paths)==10
    assert 'bias/direct/C_matrix_values/full' in paths
    assert 'bias/direct/C_matrix_values/matched' in paths
    assert all(p.exists() and p.stat().st_size>1000 for p in paths.values())
