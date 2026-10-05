"""No remote data, inference or training needed for the fixed-score diagnostic."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from scripts.tau_cap_ess_ablation import match_ess, weight_arms, analyze
from scripts.tau_tail_attribution import candidate_weights
from scripts.diagnose_tau_cap_ess_ablation import read_settings, ROOT, plots, execute


def fixture():
    rng = np.random.default_rng(4); n,k = 48,8
    inputs = dict(source_ids=np.arange(n).astype(str), weight=rng.uniform(.5,1.5,n),
        truth_cij=rng.normal(size=(n,9)), category=np.tile([11,12,21,22],12), visible_pt_sum=np.arange(n))
    s = np.minimum(rng.normal(1,2,(n,k)),np.log(30)-.001)
    g = rng.normal(size=(n,k,9))
    cfg = dict(caps=[30,20,10,5], condition_pt_edges=[12,24,36], bootstrap=80, bootstrap_seed=42)
    return inputs,g,s,cfg


def test_caps_normalization_and_exact_ess_matching():
    base=np.array([1.,2.,0.,3.]);s=np.log([[30,2,4],[1,12,24],[10,30,30],[6,12,28]])
    rows={name:(w,m) for name,w,m in weight_arms(base,s,[30,20,10,5])}
    original,_=candidate_weights(base,s)
    np.testing.assert_allclose(rows['cap30'][0],original,rtol=0,atol=0)
    for cap in [20,10,5]:
        w,meta=rows[f'cap{cap}'];mix,m=rows[f'mix{cap}']
        direct=base[:,None]*np.minimum(np.exp(s),cap);direct/=direct.sum()
        np.testing.assert_allclose(w,direct)
        assert w[2].sum()==0 and mix[2].sum()==0
        assert np.isclose(np.square(w).sum(),np.square(mix).sum(),rtol=1e-10)
        assert 0<=m['mix_lambda']<=1
        assert np.isclose(meta['removed_unnormalized_ratio_mass_fraction'],
            1-(base[:,None]*np.minimum(np.exp(s),cap)).sum()/(base[:,None]*np.exp(s)).sum())


def test_nonmonotonic_and_unreachable_ess():
    w=np.array([.8,.2]);u=np.array([.1,.9])
    assert match_ess(w,u,1.01) is None
    lam=match_ess(w,u,2.)
    assert np.isclose(lam,3/7)
    assert match_ess(w,u,1/np.square(w).sum())==0
    assert match_ess(np.array([.5,.5]),np.array([.5,.5]),1.5) is None
    with pytest.raises(ValueError): match_ess(w,u,3)


def test_no_truth_enters_matching_and_identity_caps_have_zero_contrast():
    inputs,g,s,cfg=fixture()
    s=np.minimum(s,np.log(2))
    a,arrays=analyze(inputs,g,s,cfg)
    b,_=analyze(dict(inputs,truth_cij=inputs['truth_cij']*17),g,s,cfg)
    for cap in [20,10,5]:
        assert a['arms'][f'mix{cap}']['mix_lambda']==b['arms'][f'mix{cap}']['mix_lambda']==0
    assert a['contrast_family']==6 and a['component_family']==54
    for r in a['contrasts'].values():
        assert r['error_change']==0
        np.testing.assert_allclose(r['pointwise_ci95'],0,atol=1e-14)
    np.testing.assert_allclose(arrays['event_masses'].sum(0),np.ones(8),rtol=0,atol=1e-14)


def test_paired_replay_group_sums_and_plots(tmp_path):
    inputs,g,s,cfg=fixture()
    report,arrays=analyze(inputs,g,s,cfg)
    json.dumps(report,allow_nan=False)
    for i,name in enumerate(arrays['arm_names']):
        row=report['arms'][name]
        np.testing.assert_allclose(np.array(row['C']).reshape(9),arrays['event_numerators'][:,i].sum(0))
        for grouping in ['decay','decay_x_pt']:
            gr=[r for r in report['groups'] if r['arm']==name and r['grouping']==grouping]
            assert np.isclose(sum(r['weighted_mass'] for r in gr),1)
            assert np.isclose(sum(r['base_mass'] for r in gr),1)
    assert arrays['bootstrap_C'].shape==(80,8,9)
    plots(report,tmp_path)
    assert (tmp_path/'cap_ess.png').stat().st_size>1000
    assert (tmp_path/'components.png').stat().st_size>1000


def test_settings_and_endpoint_pipeline(tmp_path):
    settings=read_settings(ROOT/'config/conditional_tau_cap_ess_ablation.yaml')
    assert settings['A']['run']=='zrv2yfgt' and 'B' not in settings
    assert settings['source_workers']==16 and settings['caps']==[30,20,10,5]
    inputs,g,s,cfg=fixture()
    before,_=analyze(inputs,g,s,cfg)
    saved=dict(truth_C=before['truth_C'],arms={'bounded':before['arms']['cap30']})
    np.savez(tmp_path/'inputs.npz',visible_pt_sum=inputs['visible_pt_sum'])
    cfg.update(source=str(tmp_path),train_events='filtered',A={})
    with patch('scripts.diagnose_tau_cap_ess_ablation.load_source',return_value=({},inputs,{'cij':g},{},{})), \
         patch('scripts.diagnose_tau_cap_ess_ablation.read_model',return_value=({'condition_pt_edges':[12,24,36]},s,saved)), \
         patch('evenet_dgpo.evenet.dataset.filtered_data.validate_filtered_dataset',return_value={'rows':416701}):
        report=execute(cfg,tmp_path)
    assert report['source_endpoints_verified'] and report['classifier_fits']==0
    assert (tmp_path/'COMPLETE').is_file() and (tmp_path/'cap_ess_report.json').is_file()


def test_bad_scores_and_duplicate_ids_rejected():
    inputs,g,s,cfg=fixture()
    with pytest.raises(ValueError): list(weight_arms(inputs['weight'],s+100,cfg['caps']))
    with pytest.raises(ValueError): analyze(dict(inputs,source_ids=np.zeros(48)),g,s,cfg)
