from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
import yaml

import compare_fourier_checkpoints as exp


def settings():
    return yaml.safe_load((exp.ROOT/'config/compare_fourier_checkpoints.yaml').read_text())


def test_protocol_pins_cold_h4_and_rejects_undertrained_setup():
    spec = settings()
    exp.validate_spec(spec)
    assert spec['expected_epoch'] is None
    assert spec['checkpoint_selection'] == 'configured'
    assert spec['workers'] == 16
    assert spec['classifier_backbone'] not in [x['checkpoint'] for x in spec['checkpoints'].values()]
    spec['classifier_fit']['min_steps'] = 999
    with pytest.raises(ValueError, match='1000'):
        exp.validate_spec(spec)
    spec = settings(); spec['classifier_fit']['steps'] = 500
    with pytest.raises(ValueError, match='min_steps'):
        exp.validate_spec(spec)


def test_checkpoint_epoch_guard_and_strict_loading(tmp_path):
    path = tmp_path/'last.ckpt'
    model = nn.Linear(3, 2)
    torch.save({'epoch':37, 'state_dict':{'model.'+k:v for k,v in model.state_dict().items()}}, path)
    with pytest.raises(ValueError, match='expected epoch 49'):
        exp.checkpoint_provenance(path, 49)
    checkpoint, meta = exp.checkpoint_provenance(path, 37)
    other = nn.Linear(3, 2)
    exp.load_raw(other, checkpoint)
    torch.testing.assert_close(other.weight, model.weight)
    del checkpoint['state_dict']['model.weight']
    with pytest.raises(ValueError, match='missing'):
        exp.load_raw(other, checkpoint)


def checkpoint_fixture(tmp_path, epochs=((35, 37), (37, 49))):
    spec = settings()
    spec['checkpoint_selection'] = 'latest_common'
    for arm, available in zip(exp.ARMS, epochs):
        directory = tmp_path/arm
        directory.mkdir()
        for epoch in available:
            torch.save({'epoch':epoch, 'global_step':(epoch+1)*65}, directory/f'epoch={epoch}.ckpt')
        (directory/'last.ckpt').symlink_to(directory/f'epoch={available[-1]}.ckpt')
        spec['checkpoints'][arm]['checkpoint'] = str(directory/'last.ckpt')
    return spec


def test_resolver_matches_actual_saved_epoch_not_last_or_planned49(tmp_path):
    spec = checkpoint_fixture(tmp_path)
    selection = exp.resolve_checkpoints(spec)
    assert selection['matched_training_budget']
    assert len(selection['available']['baseline']) == 2  # Deduplicate last link.
    for arm in exp.ARMS:
        assert selection['checkpoints'][arm]['path'].endswith('epoch=37.ckpt')
    spec['expected_epoch'] = 49
    with pytest.raises(ValueError, match='No common saved checkpoint epoch'):
        exp.resolve_checkpoints(spec)


def test_configured_endpoints_record_different_epochs_but_strict_mode_rejects_budget_mismatch(tmp_path):
    spec = checkpoint_fixture(tmp_path)
    spec['checkpoint_selection'] = 'configured'
    selection = exp.resolve_checkpoints(spec)
    assert [selection['checkpoints'][a]['epoch'] for a in exp.ARMS] == [37,49]
    assert not selection['matched_training_budget']
    assert 'does not isolate' in selection['limitation']
    spec['checkpoint_selection'] = 'latest_common'
    torch.save({'epoch':37,'global_step':123}, tmp_path/'candidate/epoch=37.ckpt')
    with pytest.raises(ValueError, match='update budgets differ'):
        exp.resolve_checkpoints(spec)


def test_checkpoint_inspection_never_creates_outputs_or_starts_compute(tmp_path, monkeypatch):
    spec = checkpoint_fixture(tmp_path)
    spec['output_dir'] = str(tmp_path/'output')
    config = tmp_path/'config.yaml'; config.write_text(yaml.safe_dump(spec, sort_keys=False))
    def forbidden(*args, **kwargs):
        pytest.fail('Inspection must not prepare data, start compute or initialize W&B')
    monkeypatch.setattr(exp, 'prepare', forbidden)
    monkeypatch.setattr(exp, 'execute', forbidden)
    exp.main(['--config',str(config),'--inspect-checkpoints'])
    assert not Path(spec['output_dir']).exists()


@pytest.mark.parametrize('change', [dict(batch_size=511),dict(drop_last_batch=False)])
def test_distributed_global_batch_contract(change):
    spec = settings(); spec['classifier_fit'].update(change)
    with pytest.raises(ValueError, match='global batch_size'):
        exp.validate_spec(spec)


def test_sixteen_shards_replay_exact_event_noise_and_merge_in_order():
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import pack_event_inputs
    n = 35  # Deliberately uneven over 16 ranks.
    batch = dict(x=torch.randn(n,3,2), x_mask=torch.ones(n,3,dtype=torch.bool),
                 conditions=torch.randn(n,2), conditions_mask=torch.ones(n,2,dtype=torch.bool))
    c, packing = pack_event_inputs(batch)
    noise = torch.randn(n,2,2,2)
    spec = dict(generation_batch_size=3,candidates_per_event=2,ddim_steps=4,sampler='stable_v')
    def sample(c, noise):
        return exp.sample_policy(FixedVelocity(),c,packing,noise,spec,torch.device('cpu'),lambda row:None)
    expected = sample(c,noise)
    parts=[]
    for rank in range(16):
        ids = exp.event_positions(n,rank,16)
        parts.append(dict(positions=ids, generated=sample(c[ids],noise[ids])))
    actual = exp.merge_generation_shards(parts[::-1],n,2)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    with pytest.raises(ValueError,match='Missing generation events'):
        exp.merge_generation_shards(parts[1:],n,2)
    with pytest.raises(ValueError,match='Duplicate generation events'):
        exp.merge_generation_shards(parts+[parts[0]],n,2)


def tiny_distributed_fit():
    from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import ConditionalRatioMLP, RatioFitConfig, fit_density_ratio
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import _score_population
    rank, _ = exp.distributed_context()
    torch.set_num_threads(1)
    torch.manual_seed(27+rank)  # Production broadcast must restore common initialization.
    model = ConditionalRatioMLP(2,2,hidden_dim=4,hidden_layers=1,architecture='concat_mlp')
    rng = torch.Generator().manual_seed(18)
    c = torch.randn(19,2,generator=rng)
    t = torch.randn(19,2,generator=rng)+.3
    g = torch.randn(19,2,generator=rng)
    weight = torch.ones(len(c))
    cfg = RatioFitConfig(steps=4,batch_size=8,drop_last_batch=True,learning_rate=.001,
        sampling='independent_epoch_shuffle',validation_interval_steps=2,
        validation_patience_evaluations=100,validation_batch_size=3)
    diagnostics = fit_density_ratio(model,c,t,weight,c,g,weight,cfg,seed=5,
                                    validation=(c,t,weight,c,g,weight))
    return dict(state=model.state_dict(),scores=_score_population(model,c,t,3),
                steps=diagnostics.steps_completed,best_step=diagnostics.best_step)


def gloo_fit_worker(rank, rendezvous, directory):
    torch.distributed.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2,
                                        timeout=timedelta(seconds=60))
    try:
        torch.save(tiny_distributed_fit(),Path(directory)/f'rank{rank}.pt')
    finally:
        torch.distributed.destroy_process_group()


def test_two_process_production_fit_and_uneven_scoring_match_single_process(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOO_SOCKET_IFNAME','lo0' if sys.platform == 'darwin' else 'lo')
    expected = tiny_distributed_fit()
    torch.multiprocessing.spawn(gloo_fit_worker,
        args=((tmp_path/'rendezvous').as_uri(),str(tmp_path)),nprocs=2,join=True)
    for rank in range(2):
        actual = torch.load(tmp_path/f'rank{rank}.pt',weights_only=True)
        assert actual['steps'] == expected['steps'] == 4
        assert actual['best_step'] == expected['best_step']
        torch.testing.assert_close(actual['scores'],expected['scores'],atol=1e-6,rtol=1e-5)
        for k in actual['state']:
            torch.testing.assert_close(actual['state'][k],expected['state'][k],atol=1e-6,rtol=1e-5)


def test_runtime_contract_detects_different_validation_and_normalization():
    raw = dict(platform={'data_parquet_dir':'train','data_parquet_val_dir':'val'},
               options={'Dataset':{'normalization_file':'norm'},
                        'Training':{'seed':42,'epochs':50,'pretrain_model_load_path':'source'}})
    exp.runtime_contract(raw, deepcopy(raw))
    other = deepcopy(raw); other['platform']['data_parquet_val_dir'] = 'different'
    with pytest.raises(ValueError, match='data mismatch'): exp.runtime_contract(raw, other)
    other = deepcopy(raw); other['options']['Dataset']['normalization_file'] = 'different'
    with pytest.raises(ValueError, match='preprocessing'): exp.runtime_contract(raw, other)


def test_identity_splits_keep_duplicates_together_and_stay_fixed():
    condition = torch.randn(120, 7, generator=torch.Generator().manual_seed(42))
    condition = torch.cat([condition, condition[:8]])
    splits = exp.identity_splits(condition, 23)
    assert sorted(torch.cat(list(splits.values())).tolist()) == list(range(len(condition)))
    for split in splits.values():
        selected = set(split.tolist())
        for i in range(8): assert (i in selected) == (120+i in selected)
    for k,v in exp.identity_splits(condition.clone(), 23).items():
        torch.testing.assert_close(v, splits[k])


def test_paired_auc_zero_difference_and_label_orientation():
    rng = np.random.default_rng(17)
    t, g = rng.normal(.4, 1, 100), rng.normal(0,1,100)
    scores = [dict(baseline=(t,g), candidate=(-t,-g)) for _ in range(3)]
    result = exp.paired_auc_comparison(scores, np.arange(100)[:,None], 100, 2, .005, True)
    assert result['candidate_minus_baseline_mean_gap'] == pytest.approx(0., abs=1e-15)
    assert result['decision'] == 'no_material_gap_difference_on_this_panel'


def test_winner_is_blocked_when_any_audit_is_not_ready():
    scores = [dict(baseline=(np.ones(80),-np.ones(80)),
                   candidate=(np.zeros(80),np.zeros(80))) for _ in range(3)]
    args = (scores, np.repeat(np.arange(40),2)[:,None], 100, 42, .005)
    good = exp.paired_auc_comparison(*args, True)
    assert good['candidate_minus_baseline_mean_gap'] == -.5
    assert good['decision'] == 'candidate_improves_h4_gap'
    assert exp.paired_auc_comparison(*args, False)['decision'].startswith('inconclusive')


class FakeNormalizer:
    def denormalize(self, x, mask, remove_padding=False):
        return x


class FixedVelocity(nn.Module):
    invisible_input_dim = 2
    invisible_normalizer = FakeNormalizer()
    def predict_diffusion_vector(self, *, noise_x, time, mode, cond_x, noise_mask):
        assert mode == 'neutrino'
        assert cond_x['x_invisible'].count_nonzero() == 0
        return .15 * noise_x + cond_x['conditions'][:, :1].reshape(-1,1,1)*.01


def test_real_sampler_replays_same_noise_and_never_receives_truth():
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import pack_event_inputs
    n=6
    batch = dict(x=torch.randn(n,3,2), x_mask=torch.ones(n,3,dtype=torch.bool),
        conditions=torch.randn(n,2), conditions_mask=torch.ones(n,2,dtype=torch.bool),
        x_invisible=torch.full((n,2,2),12345.))
    c, packing = pack_event_inputs(batch)
    noise = torch.randn(n,3,2,2)
    before = noise.clone()
    spec = dict(generation_batch_size=2, candidates_per_event=3, ddim_steps=4, sampler='stable_v')
    a = exp.sample_policy(FixedVelocity(),c,packing,noise,spec,torch.device('cpu'),lambda row: None)
    torch.manual_seed(87654)
    b = exp.sample_policy(FixedVelocity(),c,packing,noise,spec,torch.device('cpu'),lambda row: None)
    assert a.shape == (6,3,4)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    torch.testing.assert_close(noise,before,atol=0,rtol=0)
    assert not torch.allclose(a[:,0],a[:,1])


def test_physics_exact_truth_has_zero_discrepancy():
    from test_h4_topology_resolution import fixture
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec
    b=fixture(); truth=b['test_truth']
    result = exp.physics_metrics(b['test_condition'],truth,truth[:,None].expand(-1,4,-1),
                                EventPackingSpec.from_dict(b['packing_spec']))
    assert result['conditional_joint_moment_rmse'] == 0.
    assert result['angular_w1_radians'] == dict(acoplanarity=pytest.approx(0,abs=1e-15), acollinearity=pytest.approx(0,abs=1e-15))


def test_parquet_panel_selection_is_global_deterministic_and_records_rows(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import evenet.dataset.preprocess as pre
    train, val = tmp_path/'train', tmp_path/'val'
    train.mkdir(); val.mkdir(); (train/'shape_metadata.json').write_text('{}')
    pq.write_table(pa.table({'index':np.arange(20)}),val/'a.parquet')
    pq.write_table(pa.table({'index':np.arange(20,40)}),val/'b.parquet')
    def unpack(flat, metadata, drop_column_prefix):
        ids=flat['index']; n=len(ids)
        batch=dict(x=np.repeat(ids[:,None,None],6,axis=1).reshape(n,3,2).astype('f4'),
            x_mask=np.ones((n,3),bool), conditions=ids[:,None].astype('f4'),
            conditions_mask=np.ones((n,1),bool), x_invisible=np.zeros((n,2,2),'f4'),
            x_invisible_mask=np.ones((n,2),bool))
        for leg in ('a','b'):
            for axis in ('px','py','pz'): batch[f'lead_{leg}_visible_{axis}']=np.ones(n,'f4')
        return batch
    monkeypatch.setattr(pre,'unflatten_dict',unpack)
    raw={'platform':{'data_parquet_dir':str(train),'data_parquet_val_dir':str(val)}}
    spec={'panel_seed':17,'panel_events':16,'evaluation_data_dir':None}
    a=exp.select_panel(raw,spec); b=exp.select_panel(raw,spec)
    torch.testing.assert_close(a[0],b[0],atol=0,rtol=0)
    assert len(a[0]) == 16
    assert len(a[3]) == 2
    assert sum(len(part['rows']) for part in a[3]) == 16
    expected=np.sort(np.random.default_rng(17).choice(40,16,replace=False))
    np.testing.assert_array_equal(a[0][:,0].numpy(),expected)
