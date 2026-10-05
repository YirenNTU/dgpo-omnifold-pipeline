"""Local CPU fixtures; production remains a user-launched 16-GPU experiment."""
import copy
import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from scripts.train_dgpo_tau_ratio import ROOT, assemble_runtime, validate_settings, read_overlay_yaml, read_yaml
from scripts.train_conditional_spin_ratio import build_classifier, score_pair
from RL.DGPO_neutrino.conditional_tau_reward import (
    KIND, ConditionalTauReward, candidate_coordinates, feature_precision, canonical_policy_batch,
    POLICY_CONDITIONING_CONTRACT,
)
from RL.DGPO_neutrino.conditional_tau_cycle import fit_head, summarize_cij, ConditionalTauCycle
from RL.DGPO_neutrino.rewards import RewardAggregator
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec
from RL.DGPO_neutrino.omnifold_ztautau.dgpo_reward import validate_omnifold_reward_startup
from RL.DGPO_neutrino.model_utils import select_dgpo_training_state, resolve_dgpo_auto_resume_checkpoint


def example_bundle():
    spec = EventPackingSpec({'x': (2, 3), 'x_mask': (2,), 'conditions': (2,), 'conditions_mask': (2,)})
    cfg = dict(condition_dim=spec.width+16, hidden=128, candidate_dim=25,
        relative_dim=6, head_kind='film', head_depth=3, condition_hidden=256,
        condition_width=256, dropout=0., ratio_bound=30, ratio_objective='bce',
        relative_preprocessing={'mean': [0.]*6, 'scale': [1.]*6}, packing_spec=spec.to_dict())
    cfg['state_dict'] = build_classifier(cfg).state_dict()
    return dict(kind=KIND, schema_version=1, source_run='zrv2yfgt', source_checkpoint='/tmp/raw1110.ckpt',
        normalization_file='/tmp/normalization.pt', head=cfg,
        condition_mean=torch.zeros(spec.width+16), condition_scale=torch.ones(spec.width+16))


def fake_hidden(policy, batch, delta):
    return delta.flatten(1).cpu().numpy()


def live_batch(n=5):
    b = dict(x=torch.ones(n, 2, 3), x_mask=torch.ones(n, 2, dtype=torch.bool),
        conditions=torch.ones(n, 2), conditions_mask=torch.ones(n, 2, dtype=torch.bool),
        event_category=torch.full((n,), 12))
    for leg, p in [('a', [10., 3., 1., 7.]), ('b', [12., -4., 2., -8.])]:
        for key, value in zip(('E','px','py','pz'), p):
            b[f'lead_{leg}_visible_{key}'] = torch.full((n,), value)
    return b


def test_reward_reproduces_saved_head_and_excludes_answers():
    torch.manual_seed(4)
    reward = ConditionalTauReward(example_bundle(), torch.nn.Linear(1, 1), 'cpu', microbatch=2)
    batch = live_batch()
    delta = torch.randn(3, 5, 2, 2)*.03
    with patch('scripts.tau_backbone_alignment.candidate_hidden', side_effect=fake_hidden):
        result = reward.compute(delta, batch)
        expected = []
        for d in delta:
            c, f = reward.features(batch, d)
            expected.append(reward.head(torch.tensor(c), torch.tensor(f)))
        torch.testing.assert_close(result, torch.stack(expected))
        leaked = dict(batch, x_invisible=torch.randn(5,2,2), classification=torch.zeros(5),
            analyzing_power_a=torch.randn(5), source_event_key=torch.arange(5), truth_cij=torch.randn(5,9))
        torch.testing.assert_close(result, reward.compute(delta, leaked))
    assert result.shape == (3,5) and result.max() <= np.log(30)
    assert not any(p.requires_grad for p in reward.head.parameters())
    assert not any(p.requires_grad for p in reward.backbone.parameters())
    agg=RewardAggregator(); agg.add(reward,1)
    assert agg.conditional_tau_source is reward and agg.omnifold_source is None


def test_reward_install_replaces_not_multiplies_and_resume_preserves_source():
    b=example_bundle()
    r=ConditionalTauReward(b,torch.nn.Linear(1,1),'cpu')
    new=copy.deepcopy(b['head'])
    new['state_dict']['readout.bias'] += 1
    r.install(new,policy_step=100,epoch=9)
    saved=r.stack_payload()
    clone=ConditionalTauReward(b,torch.nn.Linear(1,1),'cpu')
    clone.load_stack_payload(saved)
    assert clone.round_id==1 and clone.denominator_step==100 and clone.last_refit_epoch==9
    for k,v in new['state_dict'].items():
        torch.testing.assert_close(clone.head.state_dict()[k],v)
    torch.testing.assert_close(clone.bundle['head']['state_dict']['readout.bias'], b['head']['state_dict']['readout.bias'])
    bad=copy.deepcopy(saved); bad['condition_mean'][0]+=1
    with pytest.raises(ValueError,match='normalization'):
        clone.load_stack_payload(bad)
    with pytest.raises(ValueError,match='H4/cascade'):
        clone.load_stack_payload(dict(kind='old_H4',schema_version=1))
    old=copy.deepcopy(saved); old.pop('policy_conditioning_contract')
    with pytest.raises(ValueError,match='pre-fix'):
        clone.load_stack_payload(old)


@pytest.mark.parametrize('saved_device', ['cpu', 'cuda'])
def test_resume_normalization_compares_on_cpu(saved_device):
    if saved_device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA resume regression requires GPU')
    reward = ConditionalTauReward(example_bundle(), torch.nn.Linear(1, 1), 'cpu')
    saved = copy.deepcopy(reward.stack_payload())
    for key in ('condition_mean', 'condition_scale'):
        saved[key] = saved[key].to(saved_device)
    original_equal = torch.equal
    def cpu_equal(a, b):
        assert a.device.type == b.device.type == 'cpu'
        return original_equal(a, b)
    with patch('torch.equal', side_effect=cpu_equal) as compare:
        reward.load_stack_payload(saved)
        assert compare.call_count == 2
        saved['condition_scale'][0] += 1
        with pytest.raises(ValueError, match='condition_scale'):
            reward.load_stack_payload(saved)


def test_candidate_coordinates_identical_to_original_helpers():
    from scripts.diagnose_ztautau_cij import tau_from_deltas
    from scripts.tau_relative_inputs import relative_angles
    from scripts.train_conditional_spin_ratio import tau_features
    v=np.array([[10.,3.,1.,7.],[12.,-4.,2.,-8.]])
    d=np.ones((2,2,2))*.01
    stats={'mean': [.1]*6,'scale':[2.]*6}
    a,b=tau_from_deltas(v,d[:,0]),tau_from_deltas(v[::-1],d[:,1])
    expected=np.concatenate(((relative_angles(a,b,v,v[::-1])-.1)/2,tau_features(a,b)),axis=1).astype('float32')
    np.testing.assert_array_equal(candidate_coordinates(v,v[::-1],d,stats),expected)


def test_runtime_has_one_tau_reward_no_old_h4_controls_and_pinned_setup(tmp_path):
    overlay=read_overlay_yaml(ROOT/'config/dgpo_tau_ratio_1110.yaml')
    validate_settings(overlay)
    source=dict(backbone_manifest={'checkpoint':'/tmp/raw1110.ckpt'},normalization_file='/tmp/norm.pt')
    cfg=assemble_runtime(read_yaml(ROOT/'config/train_diffusion_nersc.yaml'),overlay,source,tmp_path,tmp_path/'bundle.pt')
    d=cfg['dgpo']
    assert cfg['platform']['number_of_workers']==16 and cfg['platform']['batch_size']==512
    assert d['tau_ratio']['fit']['batch_size']==1024
    assert d['tau_ratio']['refit_every_epochs']==d['tau_ratio']['validation_every_epochs']==10
    assert d['tau_ratio']['policy_conditioning_contract']==POLICY_CONDITIONING_CONTRACT
    assert '1110_label0' in overlay['dgpo']['tau_ratio']['output_root']
    assert d['adaptive_omnifold']=={'enabled':False} and 'beta_kl' not in d
    assert cfg['reward_config']['type']=='conditional_tau' and 'omnifold' not in cfg['reward_config']
    assert d['checkpoint_load_mode']=='weights_only' and d['auto_resume_from_last']
    assert not d['lr_schedule']['resume_use_config']
    assert d['K']==8 and d['advantage_estimator']=='leave_one_out_unscaled'
    assert d['reference_trust']['coefficient']==1 and d['reference_trust']['objective']=='velocity_mse'
    assert cfg['options']['Dataset']['normalization_file']=='/tmp/norm.pt'
    assert cfg['options']['Training']['pretrain_model_load_path'] is None
    assert not cfg['options']['Training']['EMA']['enable']
    assert cfg['options']['Training']['learning_rate']==5e-5
    assert d['conditioning_learning_rates']['visible_conditioning']==1e-4
    assert d['lr_schedule']['total_epochs']==1500
    assert Path(cfg['options']['default']).is_absolute()
    w=cfg['logger']['wandb']
    assert w['fresh_run'] and w['id'] is None and w['resume']=='never'
    assert 'context256' in w['run_name']
    # Resolve against actual EveNet defaults (does not load remote checkpoint).
    import yaml
    from generate_event_info_yaml import (build_event_info_payload, parse_feature_config,
        parse_evenet_config, ordered_class_labels)
    analysis=read_yaml(ROOT/'config/analysis.yaml'); schema=read_yaml(ROOT/'config/evenet_schema.yaml')
    features=parse_feature_config(analysis)
    event_info=build_event_info_payload(ordered_class_labels(analysis,None),features,parse_evenet_config(schema,analysis,features))
    cfg['event_info']=event_info  # test-only schema; never regenerate NERSC's schema
    from evenet.control.global_config import Config
    file=tmp_path/'runtime.yaml'; file.write_text(yaml.safe_dump(cfg))
    c=Config(); c.load_yaml(file)
    assert c.dgpo.tau_ratio.workers==16
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import resolve_adaptive_config
    assert not resolve_adaptive_config(c.dgpo).enabled


def test_provenance_and_auto_resume(tmp_path):
    source=tmp_path/'raw1110.ckpt'; source.touch()
    meta={'schema_version':1,'sources':[dict(name='conditional_tau',weight=1.,metadata=
        dict(kind=KIND, source_checkpoint=str(source),reward_round_id=0))]}
    validate_omnifold_reward_startup(checkpoint=None,current_metadata=meta,policy_checkpoint=source)
    with pytest.raises(ValueError,match='denominator'):
        validate_omnifold_reward_startup(checkpoint=None,current_metadata=meta,policy_checkpoint='/tmp/wrong.ckpt')
    saved=dict(dgpo_checkpoint_version=1,dgpo_omnifold_reward_metadata=meta)
    validate_omnifold_reward_startup(checkpoint=saved,current_metadata=meta,policy_checkpoint='/tmp/resume.ckpt')
    assert select_dgpo_training_state(saved,load_mode='weights_only') is None
    assert select_dgpo_training_state(saved,load_mode='resume') is saved
    root=tmp_path/'checkpoints'; root.mkdir()
    assert resolve_dgpo_auto_resume_checkpoint(root,enabled=True) is None
    (root/'last.ckpt').touch()
    assert resolve_dgpo_auto_resume_checkpoint(root,enabled=True)==root/'last.ckpt'


def fit_fixture(n=64):
    rng=np.random.default_rng(42); cfg=example_bundle()['head']; cfg.pop('state_dict')
    cfg.update(seed=42,lr=2e-4,min_lr=1e-5,weight_decay=.001,batch_size=8,
               epochs=3,patience=3,min_delta=.01,min_steps=0)
    a=dict(condition=rng.normal(size=(n,cfg['condition_dim'])).astype('float32'),
           candidate_truth=rng.normal(size=(n,25)).astype('float32'),
           candidate_generated=rng.normal(size=(n,25)).astype('float32'),
           event_weight=np.linspace(.1,2,n),split=np.r_[np.zeros(n//2,int),np.ones(n-n//2,int)])
    return cfg,a


def test_refit_restores_absolute_best_bce_not_last(tmp_path):
    cfg,a=fit_fixture()
    from scripts.train_conditional_spin_ratio import pair_metrics
    reports=[]; count=[]
    def controlled(p,q,w):
        m=pair_metrics(p,q,w); m['bce']=[.5,.4999,.6][len(count)]; count.append(1); return m
    with patch('scripts.train_conditional_spin_ratio.pair_metrics',side_effect=controlled):
        model,best,status=fit_head(cfg,a,tmp_path/'best.pt',torch.device('cpu'),0,1,reports.append)
    assert best['epoch']==2 and status['best_val_bce']==.4999 and len(reports)==3
    assert status['total_steps']==12 and best['optimizer_steps']==8
    for k,v in best['state_dict'].items(): torch.testing.assert_close(model.state_dict()[k],v)
    assert all(np.isfinite(r['grad_norm']) for r in reports)


def test_physics_is_unweighted_and_cadence_resume_is_idempotent():
    a=np.zeros((5,9)); b=np.ones((5,9)); w=np.ones(5)
    report=summarize_cij(a,b,w)
    assert report['error']==3 and report['diagonal_error']==pytest.approx(3**.5)
    cycle=object.__new__(ConditionalTauCycle)
    cycle.cfg={'validation_every_epochs':10,'refit_every_epochs':10}
    calls=[]; cycle.evaluate=lambda epoch,step:calls.append(('eval',epoch,step))
    cycle.refit=lambda epoch,step:calls.append(('refit',epoch,step))
    assert not cycle.epoch_end(8,90)
    assert cycle.epoch_end(9,100)
    assert calls==[('eval',9,100),('refit',9,100)]
    calls.clear(); assert not cycle.epoch_end(19,200,final=True)
    assert calls==[('eval',19,200)]


def test_precision_scope_restored_on_failure():
    previous=torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('medium')
        with pytest.raises(RuntimeError):
            with feature_precision():
                assert torch.get_float32_matmul_precision()=='highest'
                raise RuntimeError('fixture')
        assert torch.get_float32_matmul_precision()=='medium'
    finally:
        torch.set_float32_matmul_precision(previous)


def test_refit_recenters_reference_without_changing_actor_optimizer(tmp_path):
    r=ConditionalTauReward(example_bundle(),torch.nn.Linear(1,1),'cpu')
    actor=torch.nn.Linear(1,1); reference=copy.deepcopy(actor)
    optimizer=torch.optim.AdamW(actor.parameters(),lr=5e-5)
    actor(torch.ones(2,1)).sum().backward(); optimizer.step()
    before=copy.deepcopy(optimizer.state_dict())
    cycle=object.__new__(ConditionalTauCycle)
    cycle.reward,cycle.actor,cycle.reference=r,actor,reference
    cycle.rank,cycle.world=0,1
    cycle.train={}; cycle.cfg={'refit_seed':42}; cycle.output=tmp_path
    cycle.generate=lambda *args,**kwargs: {}
    head=copy.deepcopy(r.bundle['head']); head['state_dict']['readout.bias']+=1
    cycle.fit=lambda *args,**kwargs:(None,head,{},None)
    cycle.emit=lambda *args:None
    cycle.refit(9,100)
    assert r.round_id==1 and r.denominator_step==100
    for k,v in actor.state_dict().items(): torch.testing.assert_close(reference.state_dict()[k],v)
    assert not any(p.requires_grad for p in reference.parameters())
    assert optimizer.state_dict()['param_groups']==before['param_groups']
    for k,entry in before['state'].items():
        for key,value in entry.items(): torch.testing.assert_close(optimizer.state_dict()['state'][k][key],value)
    # A restored epoch marker must not generate samples/refit a second time.
    cycle.generate=lambda *args,**kwargs:pytest.fail('Repeated completed cycle')
    cycle.refit(9,100)
    r.last_evaluation_epoch=9
    cycle.evaluate(9,100)
    assert r.round_id==1


def _two_rank_tau_fit(rank,directory):
    import torch.distributed as dist
    torch.set_num_threads(1)
    path=Path(directory)
    dist.init_process_group('gloo',init_method='file://'+str(path/'rendezvous'),rank=rank,world_size=2)
    try:
        cfg,a=fit_fixture(62); cfg['epochs']=2; cfg['batch_size']=7
        model,best,status=fit_head(cfg,a,path/'ddp.pt',torch.device('cpu'),rank,2,lambda _:None)
        for k,v in best['state_dict'].items(): torch.testing.assert_close(model.state_dict()[k],v)
        (path/f'status-{rank}.json').write_text(json.dumps(status))
    finally:
        dist.destroy_process_group()


def test_two_rank_fit_matches_global_batch_and_restores_best(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_two_rank_tau_fit,args=(str(tmp_path),),nprocs=2,join=True)
    left=json.loads((tmp_path/'status-0.json').read_text())
    assert left==json.loads((tmp_path/'status-1.json').read_text())
    assert left['total_steps']==6  # 31 events, global batch14; last rank has fewer real events.
    cfg,a=fit_fixture(62); cfg['epochs']=2; cfg['batch_size']=14
    _,serial,_=fit_head(cfg,a,tmp_path/'serial.pt',torch.device('cpu'),0,1,lambda _:None)
    parallel=torch.load(tmp_path/'ddp.pt',map_location='cpu',weights_only=True)
    assert serial['epoch']==parallel['epoch']
    assert serial['val_bce']==pytest.approx(parallel['val_bce'],abs=1e-6)
    for k,v in serial['state_dict'].items():
        torch.testing.assert_close(parallel['state_dict'][k],v,atol=2e-6,rtol=1e-4)


def test_canonical_process_label_preserves_observables_and_input():
    b=live_batch(); b['classification']=torch.arange(5)
    result=canonical_policy_batch(b)
    assert result['classification'].dtype==torch.long
    assert result['classification'].tolist()==[0]*5
    assert b['classification'].tolist()==list(range(5))
    for key in b:
        if key!='classification': assert result[key] is b[key]


def test_native_train_step_canonicalizes_before_any_policy_forward():
    from RL.DGPO_neutrino import dgpo_trainer as trainer
    class ReachedPolicy(Exception): pass
    seen=[]
    def intercept(_):
        import inspect
        seen.append(inspect.currentframe().f_back.f_locals['batch'])
        raise ReachedPolicy()
    b=live_batch(); b['classification']=torch.ones(5,dtype=torch.long)
    for tau in (object(),None):
        with patch.object(trainer,'_unwrap_core_evenet',new=intercept),pytest.raises(ReachedPolicy):
            trainer.train_step(None,None,None,None,b,None,None,
                SimpleNamespace(conditional_tau_source=tau),beta=1,K=8,num_ddim_steps=20,
                global_step=0,epoch=0,device=torch.device('cpu'),dtype=torch.float32)
    assert seen[0]['classification'].tolist()==[0]*5
    assert seen[1] is b  # Old H4 / other reward behavior untouched.


def test_live_panel_probe_is_paired_read_only_and_measures_label_effect():
    cycle=object.__new__(ConditionalTauCycle)
    cycle._training_path_verified=False
    cycle.device=torch.device('cpu'); cycle.rank=0; cycle.world=1
    cycle.actor=torch.nn.Linear(1,1).train(); cycle.sampler=None
    cycle.reward=SimpleNamespace(spec=EventPackingSpec.from_dict(example_bundle()['head']['packing_spec']),
        compute=lambda d,b:d.mean((-1,-2)))
    records=[]; cycle.emit=lambda row,step:records.append(row)
    b=live_batch(); b['classification']=torch.full((5,),2,dtype=torch.long)
    b['x_invisible']=torch.zeros(5,2,2); b['x_invisible_mask']=torch.ones(5,2,dtype=torch.bool)
    calls=[]
    def draw(policy,batch,sampler,**kw):
        calls.append(batch['classification'].clone())
        return torch.randn(1,5,2,2)+batch['classification'].reshape(1,5,1,1)
    rng=torch.random.get_rng_state().clone()
    with patch('RL.DGPO_neutrino.sampling.generate_neutrino_candidates',side_effect=draw):
        cycle.verify_training_batch(b,0)
        cycle.verify_training_batch(b,1)
    assert len(calls)==3 and calls[0].tolist()==[2]*5
    assert calls[1].tolist()==calls[2].tolist()==[0]*5
    assert cycle.actor.training and torch.equal(rng,torch.random.get_rng_state())
    row=records[0]
    assert row['tau/startup/incoming_label_nonzero_fraction']==1
    assert row['tau/startup/live_panel_max_sample_error']==0
    assert row['tau/startup/paired_original_label_reward']-row['tau/startup/paired_fixed0_reward']==pytest.approx(2)
    assert row['tau/startup/paired_fixed0_reward']==row['tau/startup/paired_panel_reward']
