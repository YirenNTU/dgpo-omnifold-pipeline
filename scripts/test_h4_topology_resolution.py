import sys
from pathlib import Path
import json
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose_h4_topology_resolution import analyze,reconstruct,w1


def fixture():
    n=100
    fields={f'lead_{leg}_visible_{axis}':torch.full((n,),value,dtype=torch.float64)
            for leg,xyz in [('a',[1.,0.,0.]),('b',[-1.,0.,0.])]
            for axis,value in zip(['px','py','pz'],xyz)}
    t=torch.zeros(n,4,dtype=torch.float64); t[:,1]=torch.linspace(.001,.01,n)
    g=t.clone(); g[:,1]+=.03
    return dict(schema='h4-ratio-health-v1',test_truth=t,test_generated=g[:,None,:],
        test_condition=torch.stack(list(fields.values()),1),packing_spec={'shapes':{k:[] for k in fields}},
        gen_logits=torch.zeros(n),
        truth_topology=torch.tensor(reconstruct(fields,t)['topology']),
        gen_topology=torch.tensor(reconstruct(fields,g)['topology']),
        split_indices={'test':torch.arange(n),'fit':torch.tensor([100]),'early_stop':torch.tensor([101])})


def test_resolution_and_reconstruction():
    result=analyze(fixture())
    assert result['checks']['truth']['matches_saved']
    assert result['checks']['generated']['matches_saved']
    assert result['histograms'][0]['generated_dominant_cell_fraction']==1.
    assert result['histograms'][0]['unweighted_jsd']==pytest.approx(0.)
    assert result['histograms'][-1]['unweighted_jsd']>.1
    assert result['wasserstein1_radians']['opening']['unweighted']==pytest.approx(.03,abs=1e-8)
    json.dumps(result,allow_nan=False)


def test_known_back_to_back():
    b=fixture(); fields={k:b['test_condition'][:,i] for i,k in enumerate(b['packing_spec']['shapes'])}
    r=reconstruct(fields,torch.zeros(100,4))
    assert np.allclose(r['opening'],np.pi)
    assert np.allclose(r['acoplanarity'],0)
    assert np.allclose(r['topology'][:,2],-1)


def test_saved_mismatch_is_reported_not_hidden():
    b=fixture(); b['gen_topology'][:,1]=0
    result=analyze(b)
    assert not result['checks']['generated']['matches_saved']


def test_overlap_rejected():
    b=fixture(); b['split_indices']['fit']=torch.tensor([0])
    with pytest.raises(ValueError,match='overlapping'): analyze(b)


def test_wasserstein_weights():
    assert w1(np.array([0.,1.]),np.array([0.,1.]),np.array([.5,.5]))==0
    assert w1(np.array([0.,1.]),np.array([0.,1.]),np.array([1.,0.]))==pytest.approx(.5)
