import json
import math
import sys
import subprocess
import os
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_h4_topology_resolution import fixture
from diagnose_h4_ratio_tail import unpack
from diagnose_h4_calibration_impact import directions
from diagnose_h4_classifier_calibration import (
    projected_deltas,tail_report,metrics,verify_logits,score,paired_bce_change,
    original_shard,prepare_output,resolve_classifier_settings)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_projection_roundtrip(dtype):
    b=fixture(); fields=unpack(b)
    z=b['test_generated'].to(dtype)
    projected=projected_deltas(fields,z)
    assert projected.shape==z.shape and projected.dtype==dtype
    u=directions(fields,projected.numpy())
    assert np.max(np.abs(u[:,:,0]+u[:,:,1]))<2e-6
    torch.testing.assert_close(projected_deltas(fields,projected),projected,atol=2e-6,rtol=0)


def test_phi_seam():
    b=fixture(); z=b['test_generated'].clone();z[:,:,1]=3.13;z[:,:,3]=-3.12
    projected=projected_deltas(unpack(b),z)
    assert projected[:,:,1::2].abs().max()<=math.pi


def test_tail_mass_and_finite_json():
    b=fixture(); b['gen_logits']=torch.linspace(-4,4,100)
    r=tail_report(b)
    assert r['groups']['top20']['events']==20
    assert r['groups']['top1pct']['weight_mass']==pytest.approx(torch.softmax(b['gen_logits'].double(),0)[-1].item())
    assert r['spearman_logit']['visible_a_pt'] is None
    assert all(0<=x['raw_weight_mass']<=1 for x in r['regions'])
    json.dumps(r,allow_nan=False)


def test_noncanonical_theta_is_flagged_without_calibration_or_removal():
    b=fixture(); b['test_generated'][0,0,0]=4.
    with pytest.raises(ValueError,match='Theta outside physical range'):
        tail_report(b)
    r=tail_report(b,include_calibration=False)
    assert r['invalid_theta']['events']==1
    assert r['invalid_theta']['raw_weight_mass']==pytest.approx(.01)
    assert r['invalid_theta']['test_rows']==[0]
    assert 'calibration_shift' not in r['spearman_logit']
    assert r['groups']['top20']['events']==20


def test_balanced_bce_and_label_direction():
    r=metrics(torch.ones(8),-torch.ones(8))
    assert r['auc']==1
    assert r['bce']==pytest.approx(math.log1p(math.exp(-1)))
    assert metrics(torch.zeros(8),torch.zeros(8))['bce']==pytest.approx(math.log(2))


def test_replay_guard():
    assert verify_logits(torch.ones(3),torch.ones(3,dtype=torch.float64))==0
    with pytest.raises(ValueError,match='no calibrated inference'):
        verify_logits(torch.ones(3),torch.zeros(3))
    with pytest.raises(ValueError):verify_logits(torch.ones(3),torch.ones(4))
    with pytest.raises(ValueError):verify_logits(torch.tensor([float('nan')]),torch.zeros(1))


def test_original_shards_are_contiguous_and_complete():
    shards=[original_shard(23927,rank,16) for rank in range(16)]
    assert shards[0][0]==0 and shards[-1][1]==23927
    assert all(left[1]==right[0] for left,right in zip(shards,shards[1:]))
    assert max(stop-start for start,stop in shards)-min(stop-start for start,stop in shards)<=1
    with pytest.raises(ValueError):original_shard(5,5,4)


def test_score_uses_candidates_without_gradients():
    class Model(torch.nn.Module):
        def forward(self,c,z):
            assert not torch.is_grad_enabled()
            return c[:,0]+z[:,0]
    c=torch.ones(7,2); z=torch.zeros(7,4)
    assert torch.equal(score(Model(),c,z,'cpu',3),torch.ones(7))
    z[:,0]=2
    assert torch.equal(score(Model(),c,z,'cpu',3),torch.full((7,),3.))


def test_paired_bce_identity():
    s={key:torch.zeros(12) for key in ('truth_before','gen_before','truth_after','gen_after')}
    assert paired_bce_change(s)==dict(after_minus_before=0.,paired_event_se=0.)


def test_tail_only_cli(tmp_path):
    source=tmp_path/'artifact';source.mkdir()
    (source/'COMPLETE').write_text('h4-ratio-health-v1\n')
    torch.save(fixture(),source/'best_classifier_and_test.pt')
    output=tmp_path/'result'
    script=Path(__file__).with_name('diagnose_h4_classifier_calibration.py')
    command=[sys.executable,str(script),str(source),'--output',str(output)]
    subprocess.run(command,check=True,capture_output=True,text=True)
    report=json.loads((output/'tail_attribution.json').read_text())
    assert report['groups']['top20']['events']==20
    assert not (output/'COMPLETE').exists()
    assert subprocess.run(command,capture_output=True).returncode==0


def test_prepare_output_replaces_only_owned_files(tmp_path):
    output=tmp_path/'out';output.mkdir()
    (output/'manifest.json').write_text('old')
    (output/'rank-007.pt').write_text('old')
    (output/'keep-me.txt').write_text('user')
    (output/'ray_results').mkdir();(output/'ray_results'/'stale').write_text('x')
    prepare_output(output)
    assert not (output/'manifest.json').exists()
    assert not (output/'rank-007.pt').exists()
    assert not (output/'ray_results').exists()
    assert (output/'keep-me.txt').read_text()=='user'


def test_resolve_classifier_settings_matches_production_layers(tmp_path):
    checkpoint=tmp_path/'backbone.ckpt';checkpoint.touch()
    raw={'reward_config':{'omnifold':{'backbone_checkpoint':str(checkpoint)}},
         'dgpo':{'adaptive_omnifold':{'recalibration':{
             'adapter_bottleneck':32,'asymmetric_attention':True,
             'topology_context_residual_scale':.7},'audit_fit':{
             'asymmetric_attention':False,'decoder_hidden_dim':128,
             'learning_rate':1e-3}}}}
    base,overrides,path=resolve_classifier_settings(raw)
    assert base['adapter_bottleneck']==32
    assert base['asymmetric_attention'] is True
    assert base['topology_context_residual_scale']==.7
    assert overrides=={'asymmetric_attention':False,'decoder_hidden_dim':128}
    assert path==checkpoint.resolve()


def test_score_requires_ray_before_creating_output(tmp_path):
    runtime=tmp_path/'runtime.yaml';runtime.write_text('{}\n')
    output=tmp_path/'never-created'
    script=Path(__file__).with_name('diagnose_h4_classifier_calibration.py')
    env={key:value for key,value in os.environ.items() if key!='RAY_ADDRESS'}
    result=subprocess.run([sys.executable,str(script),str(tmp_path/'missing-artifact'),
        '--output',str(output),'--score','--runtime',str(runtime)],
        env=env,capture_output=True,text=True)
    assert result.returncode!=0
    assert 'requires an active Ray cluster' in result.stderr
    assert not output.exists()
