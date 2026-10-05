import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from scripts.tau_conditional_normalization import analyze, normalized_scores, logmeanexp
from scripts.tau_tail_attribution import candidate_weights
from scripts.tau_cij_components import analyze as baseline
from scripts.diagnose_tau_conditional_normalization import execute, read_settings, log_report


def fixture():
    rng=np.random.default_rng(294)
    n=36
    inputs=dict(source_ids=np.arange(n).astype(str),weight=rng.uniform(.5,2,n),
                category=np.arange(n)%3,truth_cij=rng.normal(size=(n,9)))
    data=dict(logits=rng.normal(size=(n,64)),cij=rng.normal(size=(n,64,9)))
    cfg=dict(prefixes=[32,64],cap=30,bootstrap=30,bootstrap_seed=42,top_count=3)
    return inputs,data,cfg


def test_condition_shift_invariance_and_mass():
    inputs,data,cfg=fixture(); s=data['logits']; base=inputs['weight']
    a,b,za,zb=normalized_scores(s)
    aa,bb,_,_=normalized_scores(s+np.arange(len(s))[:,None]*20)
    np.testing.assert_allclose(a,aa,atol=1e-12)
    np.testing.assert_allclose(b,bb,atol=1e-12)
    w,_=candidate_weights(base,a)
    np.testing.assert_allclose(w.sum(1),base/base.sum())
    cw,_=candidate_weights(base,b)
    expected=base*np.cosh(za-zb); expected/=expected.sum()
    np.testing.assert_allclose(cw.sum(1),expected)
    assert not np.allclose(cw.sum(1),base/base.sum())


def test_cross_denominator_formula_and_own_half_exclusion():
    s=np.array([[0.,1.,2.,3.],[4.,5.,6.,7.]])
    _,b,za,zb=normalized_scores(s)
    np.testing.assert_allclose(np.exp(b[:,:2]),np.exp(s[:,:2])/np.exp(s[:,2:]).mean(1)[:,None])
    changed=s.copy(); changed[:,0]+=3
    _,_,za2,zb2=normalized_scores(changed)
    np.testing.assert_array_equal(zb,zb2)
    assert not np.allclose(za,za2)


def test_extreme_logits_and_zero_base():
    s=np.array([[10000.,-10000.]*32,[-10000.,10000.]*32])
    a,b,_,_=normalized_scores(s)
    for score in (a,b):
        w,_=candidate_weights(np.array([1.,0.]),score)
        assert np.isfinite(w).all() and w.sum()==pytest.approx(1)
        assert np.all(w[1]==0)
    with pytest.raises(ValueError): normalized_scores(s[:,:3])
    with pytest.raises(ValueError): logmeanexp(np.array([[np.nan,0.]]))


def test_baseline_replay_bootstrap_components_and_no_mutation():
    inputs,data,cfg=fixture(); before=data['logits'].copy()
    report,_=analyze(inputs,data,cfg)
    old,_=baseline(inputs,data,cfg)
    for new,prior in zip(report['prefixes']['64']['arms'][:3],old['arms']):
        for key in ('cij','error','event_ess'): np.testing.assert_allclose(new[key],prior[key])
    assert report['component_family']['entries']==36
    assert len(report['prefixes']['64']['components_vs_raw'])==18
    for panel in report['prefixes'].values():
        for row in panel['arms']:
            if row['arm'] in ('raw','unweighted','cap30'):
                np.testing.assert_allclose(row['comparisons'][row['arm']]['error_change_ci95'],0)
        assert panel['arms'][3]['event_mass_tv'] < 1e-12
        assert panel['arms'][4]['event_mass_tv'] > 0
    np.testing.assert_array_equal(before,data['logits'])
    json.dumps(report,allow_nan=False)


def test_float32_panel_replay():
    inputs,data,cfg=fixture()
    for key in ('weight','truth_cij'): inputs[key]=inputs[key].astype(np.float32)
    for key in data: data[key]=data[key].astype(np.float32)
    report,_=analyze(inputs,data,cfg); prior,_=baseline(inputs,data,cfg)
    from scripts.diagnose_tau_cij_components import verify_replay
    verify_replay(dict(truth=report['truth'],arms=report['prefixes']['64']['arms'][:3]),prior)


def test_execute_writes_and_rejects_bad_replay(tmp_path):
    inputs,data,cfg=fixture(); prior,_=baseline(inputs,data,cfg)
    with patch('scripts.diagnose_tau_conditional_normalization.load_source',return_value=({},inputs,data,prior,{})):
        report=execute(cfg,tmp_path)
    assert report['source_endpoints_verified']
    assert (tmp_path/'COMPLETE').exists()
    assert (tmp_path/'normalization_comparison.png').stat().st_size > 1000
    bad=tmp_path/'bad'; bad.mkdir(); prior['truth'][0]+=1
    with patch('scripts.diagnose_tau_conditional_normalization.load_source',return_value=({},inputs,data,prior,{})):
        with pytest.raises(ValueError,match='Truth endpoint'): execute(cfg,bad)
    assert not (bad/'COMPLETE').exists()


def test_config_pins_completed_panel():
    cfg=read_settings('config/conditional_tau_normalization_diagnostic.yaml')
    assert cfg['source_run']=='fzekzrmr' and cfg['source_workers']==16
    assert cfg['prefixes']==[32,64]


def test_constant_scores_leave_all_normalization_arms_unweighted():
    inputs,data,cfg=fixture(); data['logits'][:]=7
    report,_=analyze(inputs,data,cfg)
    for panel in report['prefixes'].values():
        for row in panel['arms'][:5]:
            np.testing.assert_allclose(row['cij'],panel['arms'][0]['cij'],atol=1e-14)
        for left,right in ((5,7),(6,8)):
            np.testing.assert_allclose(panel['arms'][left]['cij'],panel['arms'][right]['cij'],atol=1e-14)
        assert panel['denominator']['log_z_half_correlation'] is None


def test_known_condition_offset_removed_but_within_scores_not_repaired():
    inputs,data,cfg=fixture()
    data['cij'][:]=inputs['truth_cij'][:,None]
    data['logits'][:]=np.arange(len(inputs['weight']))[:,None]*.2
    report,_=analyze(inputs,data,cfg)
    rows=report['prefixes']['64']['arms']
    assert rows[1]['error'] > .01
    assert rows[3]['error'] < 1e-12 and rows[4]['error'] < 1e-12
    s=np.array([[0.,2.,0.,2.]])
    same,cross,_,_=normalized_scores(s)
    assert same[0,1]-same[0,0]==pytest.approx(2)
    assert cross[0,1]-cross[0,0]==pytest.approx(2)


def test_directional_arms_match_direct_formula_and_wandb(monkeypatch,tmp_path):
    inputs,data,cfg=fixture(); report,_=analyze(inputs,data,cfg)
    rows=report['prefixes']['64']['arms']
    scores=data['logits'][:,:32]-logmeanexp(data['logits'][:,32:])[:,None]
    w,_=candidate_weights(inputs['weight'],scores)
    np.testing.assert_allclose(rows[7]['cij'],np.einsum('nk,nkd->d',w,data['cij'][:,:32]))
    assert 'raw_A' in rows[7]['comparisons'] and 'raw_B' in rows[8]['comparisons']
    logged=[]; saved=[]
    mock=SimpleNamespace(Table=lambda **kw:kw,Image=lambda p:p)
    monkeypatch.setitem(sys.modules,'wandb',mock)
    run=SimpleNamespace(summary={},log=logged.append,save=lambda *a,**kw:saved.append(a))
    log_report(run,report,tmp_path)
    assert run.summary['phase']=='complete'
    assert 'K64/A_using_Z_B/minus_raw_A/error_change' in run.summary
    assert len(logged[0]['Cij/components_vs_raw']['data'])==36
    assert len(logged[0]['Cij/matrices']['data'])==18
    assert len(saved)==2
