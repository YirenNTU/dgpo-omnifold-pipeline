import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evenet_dgpo')]
import pytest
import numpy as np
import torch
from RL.DGPO_neutrino.tau_oracle_mixture import mix_reward,training_reward
from RL.DGPO_neutrino.dgpo_utils import compute_per_event_advantage
from scripts.tau_oracle_dgpo_monitor import summarize


def sample():
    torch.manual_seed(42)
    return torch.randn(8,12,dtype=torch.float64,requires_grad=True),torch.randn(8,12,2,2)*.1,torch.randn(12,2,2)*.1,torch.randn(12,2,3)


def test_continuation_requires_full_pre_refit_state_and_baseline(tmp_path):
    import pytest
    from scripts.train_tau_dgpo_general_oracle import continuation_state
    root=tmp_path/'parent';ck=root/'checkpoints'/'step290.ckpt';ck.parent.mkdir(parents=True)
    p=dict(global_step=290,dgpo_next_epoch=29,dgpo_optimizer_state_dict={},dgpo_ref_state_dict={},
           dgpo_omnifold_reward_stack=dict(round_id=0,last_refit_epoch=-1,denominator_step=0,
             head=dict(candidate_dim=533,source_policy_step=0)))
    torch.save(p,ck)
    c=dict(output=str(tmp_path/'child'),continuation_checkpoint=str(ck))
    with pytest.raises(ValueError,match='baseline'):continuation_state(c)
    baseline=root/'tau_diagnostics/eval-step-00000000';baseline.mkdir(parents=True)
    for f in ('DIRECT_COMPLETE','direct_metrics.json','direct_samples.npz'):(baseline/f).touch()
    (baseline.parent/'direct_monitor_protocol.json').write_text('{}')
    assert continuation_state(c)==(ck,baseline)
    p['dgpo_omnifold_reward_stack']['round_id']=1;torch.save(p,ck)
    with pytest.raises(ValueError,match='pre-refit'):continuation_state(c)


def test_child_monitor_inherits_protocol_and_exports_nonzero_step(tmp_path):
    import json,pytest
    from scripts.tau_oracle_dgpo_monitor import export_monitor,inherit_monitor_protocol,TARGETS
    parent=tmp_path/'parent';base=parent/'eval-step-00000000';base.mkdir(parents=True)
    n=119002
    rng=np.random.default_rng(42)
    truth=rng.normal(0,.03,(n,2,2)).astype('float32')
    draws=np.repeat(truth[:,None],8,axis=1)
    draws[:,::2]+=.01
    panel=dict(source_ids=np.arange(n),truth_deltas=truth,category=np.zeros(n),event_weight=np.ones(n))
    export_monitor(panel,draws,base,parent,0)
    child=tmp_path/'child';child.mkdir();folder=child/'eval-step-00000300';folder.mkdir()
    with pytest.raises(ValueError,match='Missing step-zero monitor protocol'):
        export_monitor(panel,draws,folder,child,300)
    inherit_monitor_protocol(parent,child);inherit_monitor_protocol(parent,child)
    result=export_monitor(panel,draws,folder,child,300)
    assert (folder/'DIRECT_COMPLETE').exists()
    for key in TARGETS:
        assert result['tau/direct/'+key+'/spread_rms_rad_over_initial']==pytest.approx(1.)
    p=child/'direct_monitor_protocol.json';spec=json.loads(p.read_text());spec['kreg']=4;p.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='protocol differs'):inherit_monitor_protocol(parent,child)


def test_exact_mixture_and_alpha_zero():
    s,d,t,v=sample();r,a=mix_reward(s,d,t,v,.1)
    expected=.9*s.softmax(0).detach();expected[a['best'],torch.arange(12)]+=.1
    torch.testing.assert_close(r.softmax(0),expected)
    assert not r.requires_grad
    r0,_=mix_reward(s,d,t,v,0)
    torch.testing.assert_close(compute_per_event_advantage(r0,estimator='leave_one_out_unscaled')[0],
        compute_per_event_advantage(s,estimator='leave_one_out_unscaled')[0])


def test_candidate_and_event_permutations_and_periodicity():
    s,d,t,v=sample();r,_=mix_reward(s,d,t,v)
    k=torch.randperm(8);b=torch.randperm(12)
    rp,_=mix_reward(s[k][:,b],d[k][:,b],t[b],v[b])
    torch.testing.assert_close(rp,r[k][:,b])
    dd=d.double();dd[...,1]+=2*np.pi
    rr,_=mix_reward(s,dd,t,v)
    torch.testing.assert_close(rr,r)


def test_truth_best_candidate_receives_mass_and_invalid_rejected():
    s,d,t,v=sample();d[3]=t
    r,a=mix_reward(s,d,t,v)
    assert (a['best']==3).all()
    torch.testing.assert_close(r.exp().sum(0),torch.ones(12,dtype=r.dtype))
    import pytest
    with pytest.raises(ValueError):mix_reward(s,d,t,v*0)
    with pytest.raises(ValueError):mix_reward(s,d,t,v,1)


def test_monitor_spread_and_wrapped_phi():
    d=np.zeros((10,8,2,2));d[:,::2,:,1]=np.pi-.01;d[:,1::2,:,1]=-np.pi+.01
    m,_=summarize(d,np.zeros((10,2,2)),[np.linspace(-.2,.2,41)]*4)
    assert abs(m['leg_a_delta_phi/spread_rms_rad']-.01)<1e-10
    assert m['leg_a_delta_theta/spread_rms_rad']==0


def test_real_cached_oracle_matches_previous_selection():
    src=ROOT/'artifacts/tau_pet_bidirectional_classifier/epoch287-10pct-uncapped-regularized-inputs-run-01'
    old=ROOT/'artifacts/tau_pet_bidirectional_classifier/two-leg-nearest-truth-oracle-svd-run-01/oracle_selection.npz'
    vis=Path('/Users/yirenwu/Ztautau/ml_pipeline/artifacts/tau_general_step500_layers/saved_samples.npz')
    if not old.exists() or not vis.exists():
        import pytest;pytest.skip('Local research caches unavailable')
    with np.load(vis) as z:v=np.stack([z['visible_'+leg][:128,1:] for leg in 'ab'],1)
    d=np.load(src/'validation_deltas.npy',mmap_mode='r')[:128].copy();t=np.load(src/'validation_truth_deltas.npy')[:128]
    s=torch.zeros(8,128,dtype=torch.float64)
    _,a=mix_reward(s,torch.tensor(d).permute(1,0,2,3),torch.tensor(t),torch.tensor(v))
    with np.load(old) as z:np.testing.assert_array_equal(a['best'].numpy(),z['candidate_index'][:128])


def test_runtime_preserves_source_architecture_and_training_contract():
    import yaml
    from scripts.train_tau_dgpo_general_oracle import assemble,settings
    source=Path('/Users/yirenwu/Ztautau/ml_pipeline/artifacts/tau_step500_eval_config/source_runtime_step500.yaml')
    if not source.exists():
        import pytest;pytest.skip('Original runtime unavailable locally')
    s=yaml.safe_load(source.read_text());c=settings(ROOT/'config/tau_dgpo_general_oracle10_epoch287.yaml')
    r=assemble(s,c)
    assert r['network']==s['network']
    assert r['platform']==s['platform']
    assert r['dgpo']['reference_trust']==s['dgpo']['reference_trust']
    assert r['dgpo']['K']==8 and r['platform']['number_of_workers']==16
    assert not r['dgpo']['tau_ratio']['combined_reward']
    assert r['dgpo']['tau_ratio']['refit_every_epochs']==30
    assert r['dgpo']['tau_ratio']['refit_relative_to_install']
    assert r['dgpo']['tau_ratio']['validation_every_epochs']==5
    assert r['dgpo']['tau_ratio']['direct_oracle_monitor']
    assert r['nersc']['submit'] is False


@pytest.mark.parametrize('alpha',[.1,.2])
def test_refit_and_resume_mix_the_new_head_and_new_candidates(alpha):
    """Exercise the real General install/serialization path, not just softmax."""
    import copy
    from unittest.mock import patch
    from scripts.test_tau_process_label_conditioning import bundle,batch
    from scripts.test_dgpo_tau_ratio import fake_hidden
    from RL.DGPO_neutrino.conditional_tau_reward import ConditionalTauReward
    from scripts.train_conditional_spin_ratio import build_classifier
    torch.manual_seed(42)
    source=bundle();general=ConditionalTauReward(source,torch.nn.Linear(1,1),'cpu',2)
    b=batch();b['x_invisible']=torch.randn(5,2,2)*.03;b['x_invisible_mask']=torch.ones(5,2,1)
    d=torch.randn(8,5,2,2)*.04;d[2]=b['x_invisible']

    def check(head,draws,winner):
        logits=head.compute(draws,b)
        r,metrics=training_reward(logits,draws,b,alpha)
        expected=(1-alpha)*logits.double().softmax(0);expected[winner,:]+=alpha
        torch.testing.assert_close(r.double().softmax(0),expected,rtol=1e-6,atol=1e-7)
        return logits,r

    with patch('scripts.tau_backbone_alignment.candidate_hidden',side_effect=fake_hidden):
        old,_=check(general,d,2)
        fresh=copy.deepcopy(source['head']);fresh.update(source_policy_step=100,optimizer_steps=2000)
        fitted=build_classifier(fresh)
        with torch.no_grad():
            for p in fitted.parameters():p.add_(torch.randn_like(p)*.03)
        fresh['state_dict']=fitted.state_dict()
        general.install(fresh,policy_step=100,epoch=9)
        new,_=check(general,d,2)
        assert not torch.allclose(old.softmax(0),new.softmax(0))
        assert (general.round_id,general.denominator_step)==(1,100)
        # A new bank moves the closest candidate. Never retain the old pick.
        d2=d.clone();d2[[2,6]]=d2[[6,2]]
        _,expected=check(general,d2,6)
        restored=ConditionalTauReward(source,torch.nn.Linear(1,1),'cpu',2)
        restored.load_stack_payload(general.stack_payload())
        _,actual=check(restored,d2,6)
        torch.testing.assert_close(actual,expected)
        assert (restored.round_id,restored.denominator_step)==(1,100)


@pytest.mark.parametrize("interval", [30, 50])
def test_cadence_migration_only_changes_refit_and_is_retry_safe(tmp_path, interval):
    import json,yaml
    from scripts.train_tau_dgpo_general_oracle import settings,assemble,update_cadence,validate_cadence_change
    from unittest.mock import patch
    import pytest
    c=settings(ROOT/'config/tau_dgpo_general_oracle10_epoch287.yaml')
    c['refit_every_epochs']=interval
    if interval==30:c.pop('classifier_fit')  # Historical migration predates fit extension.
    c['output']=str(tmp_path/'out');c['source_directory']=str(tmp_path/'source')
    old=dict(c,refit_every_epochs=10 if interval==30 else 30)
    if interval==30:old.pop('refit_relative_to_install')
    validate_cadence_change(old,c)
    bad=dict(c,oracle_fraction=.2)
    with pytest.raises(ValueError):validate_cadence_change(old,bad)
    sourcefile=Path('/Users/yirenwu/Ztautau/ml_pipeline/artifacts/tau_step500_eval_config/source_runtime_step500.yaml')
    if not sourcefile.exists():pytest.skip('Source runtime unavailable')
    source=yaml.safe_load(sourcefile.read_text())
    src=Path(c['source_directory']);src.mkdir();(src/'runtime.yaml').write_text(yaml.safe_dump(source))
    out=Path(c['output']);out.mkdir();(out/'PREPARED').touch()
    (out/'settings.json').write_text(json.dumps(old));(out/'runtime.yaml').write_text(yaml.safe_dump(assemble(source,old)))
    ck=out/'checkpoints';ck.mkdir();(ck/'last.ckpt').write_bytes(b'checkpoint must not change')
    update_cadence(c);update_cadence(c)
    assert yaml.safe_load((out/'runtime.yaml').read_text())==assemble(source,c)
    assert (ck/'last.ckpt').read_bytes()==b'checkpoint must not change'
    # Simulate interrupted metadata commit and retry without losing history.
    (out/'settings.json').write_text(json.dumps(old));update_cadence(c)
    assert json.loads((out/'settings.json').read_text())==c


def test_fit_migration_preserves_state_and_recovers_partial_write(tmp_path):
    import json,yaml,pytest
    from scripts.train_tau_dgpo_general_oracle import settings,assemble,update_fit
    c=settings(ROOT/'config/tau_dgpo_general_oracle10_epoch287.yaml')
    c.update(output=str(tmp_path/'out'),source_directory=str(tmp_path/'source'))
    old=dict(c);old.pop('classifier_fit')
    sourcefile=Path('/Users/yirenwu/Ztautau/ml_pipeline/artifacts/tau_step500_eval_config/source_runtime_step500.yaml')
    if not sourcefile.exists():pytest.skip('Source runtime unavailable')
    source=yaml.safe_load(sourcefile.read_text())
    src=Path(c['source_directory']);src.mkdir();(src/'runtime.yaml').write_text(yaml.safe_dump(source))
    out=Path(c['output']);out.mkdir();(out/'PREPARED').touch()
    (out/'settings.json').write_text(json.dumps(old));(out/'runtime.yaml').write_text(yaml.safe_dump(assemble(source,old)))
    ck=out/'checkpoints';ck.mkdir();(ck/'last.ckpt').write_bytes(b'preserve optimizer and weights')
    update_fit(c);update_fit(c)
    fit=yaml.safe_load((out/'runtime.yaml').read_text())['dgpo']['tau_ratio']['fit']
    assert fit['epochs']==1000 and fit['lr_decay_epochs']==250 and fit['require_plateau']
    assert fit['weight_decay']==.005 and fit['dropout']==.1
    assert (ck/'last.ckpt').read_bytes()==b'preserve optimizer and weights'
    (out/'settings.json').write_text(json.dumps(old));update_fit(c)
    bad=dict(c,oracle_fraction=.2)
    with pytest.raises(ValueError):update_fit(bad)


def test_extended_fit_requires_plateau_and_holds_minimum_lr(tmp_path):
    import pytest
    from unittest.mock import patch
    from scripts.test_dgpo_tau_ratio import fit_fixture
    from RL.DGPO_neutrino.conditional_tau_cycle import fit_head
    from scripts.train_conditional_spin_ratio import pair_metrics
    cfg,data=fit_fixture(16)
    cfg.update(epochs=8,lr_decay_epochs=2,require_plateau=True,patience=3,min_delta=.001,
               dropout=.1,weight_decay=.005)
    reports=[];values=iter([.6,.5,.5,.5,.5])
    def controlled(p,q,w):
        m=pair_metrics(p,q,w);m['bce']=next(values);return m
    with patch('scripts.train_conditional_spin_ratio.pair_metrics',side_effect=controlled):
        _,saved,status=fit_head(cfg,data,tmp_path/'best.pt',torch.device('cpu'),0,1,reports.append)
    assert status['early_stopped'] and status['epochs']==5 and saved['epoch']==2
    assert all(abs(r['lr']-cfg['min_lr'])<1e-12 for r in reports[2:])
    cfg['epochs']=2;values=iter([.6,.5])
    with patch('scripts.train_conditional_spin_ratio.pair_metrics',side_effect=controlled):
        with pytest.raises(RuntimeError,match='without validation plateau'):
            fit_head(cfg,data,tmp_path/'capped.pt',torch.device('cpu'),0,1,lambda _:None)
    assert (tmp_path/'capped.pt').exists()


def test_refit30_is_relative_to_last_install_and_validation_stays5():
    from types import SimpleNamespace
    from RL.DGPO_neutrino.conditional_tau_cycle import ConditionalTauCycle
    for installed,next_epoch in [(-1,29),(9,39),(19,49)]:
        calls=[];cycle=object.__new__(ConditionalTauCycle)
        cycle.cfg=dict(refit_every_epochs=30,refit_relative_to_install=True,validation_every_epochs=5)
        cycle.reward=SimpleNamespace(last_refit_epoch=installed)
        cycle.evaluate=lambda epoch,step:calls.append(('val',epoch))
        cycle.refit=lambda epoch,step:calls.append(('refit',epoch))
        for epoch in range(installed+1,next_epoch+1):cycle.epoch_end(epoch,(epoch+1)*10)
        assert [e for kind,e in calls if kind=='refit']==[next_epoch]
        assert all((e+1)%5==0 for kind,e in calls if kind=='val')


def test_oracle20_config_routes_fraction_without_changing_policy_contract():
    import copy,yaml
    from scripts.train_tau_dgpo_general_oracle import assemble,settings
    c=settings(ROOT/'config/tau_dgpo_general_oracle20_epoch287.yaml')
    assert c['oracle_fraction']==.2 and not c.get('continuation_checkpoint')
    assert c['refit_every_epochs']==50
    source=Path('/Users/yirenwu/Ztautau/ml_pipeline/artifacts/tau_step500_eval_config/source_runtime_step500.yaml')
    if not source.exists():pytest.skip('Original runtime unavailable locally')
    source=yaml.safe_load(source.read_text())
    r=assemble(source,c)
    control=copy.deepcopy(c);control['oracle_fraction']=.1
    old=assemble(source,control)
    assert r['dgpo']['oracle_mixture']['alpha']==.2
    assert r['experiment']['oracle_fraction']==.2
    assert '0.8 softmax' in r['experiment']['reward']
    assert 'oracle20_epoch287.yaml' in r['nersc']['execution']['command']
    old['dgpo']['oracle_mixture']['alpha']=.2
    assert r['dgpo']==old['dgpo']
    assert r['network']==old['network'] and r['platform']==old['platform']
    assert r['dgpo']['tau_ratio']['fit']['lr_decay_epochs']==1500
    assert r['dgpo']['tau_ratio']['fit']['batch_size']==1024
