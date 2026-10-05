import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from scripts.tau_cij_components import analyze, event_accounting, simultaneous_intervals
from scripts.diagnose_tau_cij_components import read_settings, load_source, verify_replay, execute
from scripts.run_tau_cap_confirmation import analyze as confirmation
from scripts.test_tau_tail_attribution import fixture
from scripts.tau_tail_attribution import candidate_weights


def panel():
    inputs,data,cfg=fixture()
    cfg.update(cap=30,top_count=3)
    return inputs,data,cfg


def test_config_pins_saved_full_panel_and_no_training(tmp_path):
    cfg=read_settings(Path(__file__).resolve().parents[1]/'config/conditional_tau_cij_components.yaml')
    assert cfg['source_run']=='fzekzrmr' and cfg['source_workers']==16 and cfg['bootstrap']==2000
    cfg['cap']=10
    p=tmp_path/'bad.yaml'; p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError): read_settings(p)


def test_reproduces_prior_and_exact_component_changes():
    inputs,data,cfg=panel()
    r,arrays=analyze(inputs,data,cfg)
    prior=confirmation(inputs,data,cfg)
    verify_replay(r,prior)
    assert len(r['components'])==18 and r['family']['entries']==18
    for name in ('raw','cap30'):
        rows=[row for row in r['components'] if row['arm']==name]
        change=np.array([row['absolute_error_change'] for row in rows])
        np.testing.assert_allclose(arrays[name+'_absolute_error_contribution'].sum(0),change,atol=1e-12)
        np.testing.assert_allclose(arrays[name+'_within']+arrays[name+'_between'],arrays[name+'_delta'],atol=1e-12)
        categories=sum(np.array(g['absolute_error_contribution']) for g in r['categories'][name])
        np.testing.assert_allclose(categories,change,atol=1e-12)
        for row in rows:
            assert row['simultaneous_ci95'][0] <= row['absolute_error_change'] <= row['simultaneous_ci95'][1]
    prior['arms'][1]['error']+=1
    with pytest.raises(ValueError,match='endpoint'): verify_replay(r,prior)


def test_identity_has_no_change_no_tradeoff():
    inputs,data,cfg=panel(); data['logits'][:]=0
    r,arrays=analyze(inputs,data,cfg)
    for row in r['components']:
        assert row['absolute_error_change']==pytest.approx(0,abs=1e-12)
        np.testing.assert_allclose(row['simultaneous_ci95'],0,atol=1e-12)
        assert row['status']=='unresolved'
    np.testing.assert_allclose(arrays['cap30_delta'],0,atol=1e-12)


def test_between_only_and_within_only_accounting_and_overshoot():
    # Equal candidates in each condition: all motion is between conditions.
    g=np.array([[[0.],[0.]],[[2.],[2.]]])
    u=np.ones((2,2))/4; w=np.array([[.1,.1],[.4,.4]])
    nu=np.einsum('nk,nkd->nd',u,g); na=np.einsum('nk,nkd->nd',w,g)
    a=event_accounting(na,w.sum(1),nu,u.sum(1),np.array([1.2]))
    np.testing.assert_allclose(a['within'],0)
    # Crosses truth: U error .2 -> A error .4, net +.2 (not naive direction).
    assert a['absolute_error_contribution'].sum()==pytest.approx(.2)
    # Equal event masses with different candidate preferences: within only.
    g=np.array([[[0.],[2.]],[[0.],[2.]]]); w=np.array([[.1,.4],[.1,.4]])
    nu=np.einsum('nk,nkd->nd',u,g); na=np.einsum('nk,nkd->nd',w,g)
    a=event_accounting(na,w.sum(1),nu,u.sum(1),np.array([2.]))
    np.testing.assert_allclose(a['between'],0)
    assert a['absolute_error_contribution'].sum()==pytest.approx(-.6)


def test_accounting_invariant_to_observable_origin_and_zero_weight():
    inputs,data,cfg=panel(); inputs['weight'][0]=0
    _,first=analyze(inputs,data,cfg)
    inputs['truth_cij']+=13; data['cij']+=13
    _,second=analyze(inputs,data,cfg)
    for key in first:
        if key!='source_ids': np.testing.assert_allclose(first[key],second[key],atol=1e-12)
    np.testing.assert_allclose(first['raw_delta'][0],0)


def test_max_deviation_joint_family_band():
    point=np.zeros((2,9)); draws=np.zeros((100,2,9))
    draws[:,1,8]=np.arange(100)
    bands,radius=simultaneous_intervals(point,draws)
    assert radius==pytest.approx(np.quantile(np.arange(100),.95))
    np.testing.assert_allclose(bands[0],-radius)
    np.testing.assert_allclose(bands[1],radius)


def test_concentration_and_tradeoffs_match_direct_accounting():
    inputs,data,cfg=panel(); data['logits'][3,5]=14
    report,saved=analyze(inputs,data,cfg)
    for name in ('raw','cap30'):
        c=saved[name+'_absolute_error_contribution']
        for j,r in enumerate(report['attribution'][name]):
            harm=np.maximum(c[:,j],0)
            assert r['harmful_sum']-r['helpful_sum']==pytest.approx(r['net'])
            assert r['concentration'][0]['harmful_sum']==pytest.approx(harm.max())
            help_=np.maximum(-c[:,j],0)
            for k in range(9):
                expected=None if help_.sum()==0 else help_[c[:,k]>0].sum()/help_.sum()
                value=report['tradeoffs'][name][j][k]
                if expected is None: assert value is None
                else: assert value==pytest.approx(expected)


def test_reject_bad_id_and_nonfinite():
    inputs,data,cfg=panel(); inputs['source_ids'][0]=inputs['source_ids'][1]
    with pytest.raises(ValueError): analyze(inputs,data,cfg)
    inputs,data,cfg=panel(); data['cij'][0,0,0]=np.nan
    with pytest.raises(ValueError): analyze(inputs,data,cfg)


def test_source_alignment_and_protocol_guards(tmp_path):
    inputs,data,cfg=panel(); n,k=data['logits'].shape
    cfg.update(source=str(tmp_path),source_run='fzekzrmr',events='/filtered/val',
               expected_events=n,expected_candidates=k,source_workers=16)
    m=dict(conditions=n,prefixes=[64],workers=16,cap=30,events='/filtered/val',weights='raw_state_dict_only',
           classifier_run='pzq0nl1i',inherited_candidates=0,classifier_fits=0,policy_updates=0)
    (tmp_path/'COMPLETE').write_text('ok')
    (tmp_path/'wandb.json').write_text('{"id":"fzekzrmr"}')
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    (tmp_path/'cap_confirmation_report.json').write_text(json.dumps(confirmation(inputs,data,cfg)))
    np.savez(tmp_path/'inputs.npz',**inputs)
    np.savez(tmp_path/'samples_and_scores.npz',source_ids=inputs['source_ids'],**data)
    with patch('evenet_dgpo.evenet.dataset.filtered_data.validate_filtered_dataset',return_value={'rows':n}):
        load_source(cfg)
        np.savez(tmp_path/'samples_and_scores.npz',source_ids=inputs['source_ids'][::-1],**data)
        with pytest.raises(ValueError,match='identities'): load_source(cfg)
        m['weights']='ema'; (tmp_path/'manifest.json').write_text(json.dumps(m))
        with pytest.raises(ValueError,match='protocol'): load_source(cfg)


def test_execute_report_and_wandb_summary_api(tmp_path):
    inputs,data,cfg=panel(); prior=confirmation(inputs,data,cfg)
    from wandb.sdk.wandb_summary import Summary
    values={}; summary=Summary(lambda:values)
    summary._set_update_callback(lambda record:values.update({x.key[0]:x.value for x in record.update}))
    logs=[]; files=[]
    run=SimpleNamespace(summary=summary,log=logs.append,save=lambda *a,**k:files.append(a))
    fake=SimpleNamespace(Table=lambda **kw:kw,Image=lambda path:path)
    with patch('scripts.diagnose_tau_cij_components.load_source',return_value=({},inputs,data,prior,{'rows':len(inputs['weight'])})), \
            patch.dict('sys.modules',{'wandb':fake}):
        report=execute(cfg,tmp_path,run)
    assert values['phase']=='complete' and values['generated_samples']==0
    assert report['source_endpoints_verified'] and (tmp_path/'COMPLETE').is_file()
    assert (tmp_path/'event_contributions.npz').is_file() and len(files)==3
    json.loads((tmp_path/'component_report.json').read_text())
    assert len(logs)>5
