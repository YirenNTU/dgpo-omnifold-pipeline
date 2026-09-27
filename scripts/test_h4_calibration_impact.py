import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose_h4_calibration_impact import analyze,calibrate,separation,check_panel,load_arm
from test_h4_topology_resolution import fixture


def panel():
    b=fixture()
    return dict(condition=b['test_condition'],truth=b['test_truth'],packing_spec=b['packing_spec'],
                test_rows=torch.arange(100),pool_rows=torch.arange(100))


def test_production_projection_is_back_to_back():
    u=np.array([[[[1.,0.,0.],[-.8,.6,0.]]]])
    v=calibrate(u)
    assert np.allclose(v[:,:,0],-v[:,:,1])
    expected=(u[:,:,0]-u[:,:,1])
    expected/=np.linalg.norm(expected,axis=-1,keepdims=True)
    assert np.allclose(v[:,:,0],expected)
    assert separation(u[:,:,0],v[:,:,0]).item()>0


def test_perfect_opposite_pair_unchanged():
    u=np.array([[[[1.,0.,0.],[-1.,0.,0.]]]])
    assert np.allclose(calibrate(u),u)


def test_truth_not_silently_projected():
    p=panel()
    candidates=np.repeat(p['truth'].numpy()[:,None,:],128,axis=1)
    r,errors=analyze(p,{'exact':candidates})
    assert r['arms']['exact']['before']['mean_leg_error_to_saved_target_radians']['mean']==pytest.approx(0)
    assert r['arms']['exact']['after']['mean_leg_error_to_saved_target_radians']['mean']>0
    assert r['arms']['exact']['after']['opening_deficit_radians']['mean']==pytest.approx(0,abs=1e-14)
    assert errors['exact/after'].shape==(100,)
    json.dumps(r,allow_nan=False)


def test_panel_identity_guard():
    a=panel(); b=panel()
    check_panel(a,b)
    b['pool_rows']=torch.arange(100)+1
    with pytest.raises(ValueError,match='pool_rows'):
        check_panel(a,b)


def test_incomplete_guard(tmp_path):
    with pytest.raises(ValueError,match='Incomplete'):
        load_arm(tmp_path)


def test_cli(tmp_path):
    import subprocess
    p=panel()
    command=[sys.executable,str(Path(__file__).with_name('diagnose_h4_calibration_impact.py'))]
    for name in ('pretrained10pct','step1110','pretrainedfull'):
        root=tmp_path/name
        root.mkdir()
        (root/'COMPLETE').write_text('h4-spike-coverage-v1')
        (root/'manifest.json').write_text(json.dumps(dict(arm=name,policy_updates=0,seed=42,workers=16,batch_size=16,ddim_steps=20,K=128,events=100)))
        torch.save(p,root/'panel.pt')
        torch.save(dict(generated=p['truth'][:,None,:].repeat(1,128,1)),root/'candidates.pt')
        command.extend(['--'+name,str(root)])
    output=tmp_path/'result'
    command.extend(['--output',str(output)])
    subprocess.run(command,check=True,capture_output=True)
    report=json.loads((output/'report.json').read_text())
    assert report['physics_closure_complete'] is False
    assert subprocess.run(command,capture_output=True).returncode!=0
