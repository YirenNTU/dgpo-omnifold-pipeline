import copy
import json
import subprocess
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose_h4_ratio_tail import analyze


def bundle():
    n=40
    g=torch.zeros(n); g[0]=10
    z=torch.linspace(-.8,.8,n)[:,None].repeat(1,3)
    condition=torch.ones(n,3)
    condition[:,1]=float('nan')  # padded feature must not flag valid inputs
    condition[:,2]=0
    return dict(schema='h4-ratio-health-v1',seed=42,fit_diagnostics={'best_step':7},
        gen_logits=g,truth_logits=torch.zeros(n),truth_topology=z,gen_topology=z,
        test_condition=condition,packing_spec={'shapes':{'x':[1,2],'x_mask':[1,1]}},
        model_state={},test_truth=torch.zeros(n,4),test_generated=torch.zeros(n,1,4),
        split_indices={'test':torch.arange(n),'fit':torch.tensor([100]),'early_stop':torch.tensor([101])})


def test_tail_and_no_mutation():
    source=bundle(); original=source['gen_logits'].clone()
    result=analyze(source)
    assert result['top20'][0]['test_row']==0
    assert result['top20'][0]['weight_mass']>.99
    row=next(r for r in result['sensitivity'] if r['arm']=='raw' and r['removed']==1)
    assert row['ess']==pytest.approx(39.)
    assert row['mean_ratio']==pytest.approx(1.)
    assert row['removed_gen_bce_share']>0
    assert torch.equal(source['gen_logits'],original)
    assert result['top20'][0]['input_ranges']['x']['nonfinite']==0
    json.dumps(result,allow_nan=False)


def test_overlap_and_nonfinite_rejected():
    source=bundle(); source['split_indices']['fit']=torch.tensor([0])
    with pytest.raises(ValueError,match='overlaps'): analyze(source)
    source=bundle(); source['gen_logits'][0]=float('inf')
    with pytest.raises(ValueError,match='Nonfinite'): analyze(source)


def test_affine_normalizer():
    source=bundle()
    source['model_state']={'backbone.invisible_normalizer.mean':torch.tensor([1.,2.]),
                           'backbone.invisible_normalizer.std':torch.tensor([2.,4.])}
    result=analyze(source)
    assert result['top20'][0]['generated_affine_pre_icdf']==[[-.5,-.5],[-.5,-.5]]


def test_cli_and_overwrite_protection(tmp_path):
    directory=tmp_path/'artifact'; directory.mkdir()
    torch.save(bundle(),directory/'best_classifier_and_test.pt')
    (directory/'COMPLETE').write_text('h4-ratio-health-v1\n')
    output=tmp_path/'tail.json'
    command=[sys.executable,str(Path(__file__).with_name('diagnose_h4_ratio_tail.py')),
             str(directory),'--output',str(output)]
    subprocess.run(command,check=True,capture_output=True)
    assert len(json.loads(output.read_text())['top20'])==20
    again=subprocess.run(command,capture_output=True,text=True)
    assert again.returncode!=0 and 'Output exists' in again.stderr
