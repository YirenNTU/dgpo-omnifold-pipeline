import sys
from pathlib import Path
import copy
import torch
import pytest
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evenet_dgpo'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evenet.network.body.relation_conditioning import pair_summary, RelationConditioning, load_relation_weights
from evenet.network.body.visible_conditioning import VisibleConditioning


def branch(mode=None):
    torch.manual_seed(4)
    return VisibleConditioning(feature_names=['Part_eta','Part_phi','Part_energy','Part_pt'],token_dim=8,hidden_dim=8,num_layers=3,n_branches=2,
        relation_adapter=None if mode is None else dict(enabled=True,mode=mode))


def test_pair_geometry_mask_permutation_periodicity():
    theta=torch.tensor([[1.,2.,float('nan')],[1.,2.,3.]])
    phi=torch.tensor([[.2,.8,float('nan')],[.3,.2,.7]])
    e=torch.tensor([[2.,3.,float('nan')],[1.,2.,3.]])
    valid=torch.tensor([[1,1,0],[0,0,0]],dtype=torch.bool)
    a,p=pair_summary(theta,phi,e,e,valid)
    b,_=pair_summary(theta.flip(1),phi.flip(1)+2*torch.pi,e.flip(1),e.flip(1),valid.flip(1))
    assert torch.allclose(a,b,atol=1e-6)
    assert torch.isfinite(a).all() and a[1].count_nonzero()==0
    assert p[:,0].tolist()==[True,False]
    assert a[0,2].item()==pytest.approx(.04)


def test_identity_shared_weights_rng_and_trainability():
    base=branch(); rng=torch.random.get_rng_state()
    for mode in ['context','relations']:
        model=branch(mode)
        assert torch.equal(torch.random.get_rng_state(),rng)
        for k,v in base.state_dict().items(): assert torch.equal(v,model.state_dict()[k])
        raw=torch.randn(2,3,4);raw[...,2:]=raw[...,2:].abs()
        tokens=torch.randn(2,3,8);mask=torch.ones(2,3,1,dtype=torch.bool)
        # Also check identity for already-trained nonzero global modulation.
        for h in base.modulations:
            nn.init.normal_(h.weight,std=.1)
        model.load_state_dict(base.state_dict(),strict=False)
        old=base(raw,tokens,mask);new=model(raw,tokens,mask)
        assert all(torch.equal(a,b) for xs,ys in zip(old,new) for a,b in zip(xs,ys))
        sum(v.sum() for layer in new for v in layer).backward()
        assert all(h.weight.grad.abs().sum()>0 for h in model.relation_adapter.outputs)
        base=branch();rng=torch.random.get_rng_state()


def test_parameter_match():
    a,b=branch('context'),branch('relations')
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    assert all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items())


def wrap(b):
    m=nn.Module();m.TruthGeneration=nn.Module();m.TruthGeneration.visible_conditioning=b
    return m


def test_strict_migration_rejects_missing_shared_tensor():
    source=wrap(branch());target=wrap(branch('relations'))
    state=source.state_dict()
    load_relation_weights(target,dict(state_dict=state))
    for k,v in state.items():assert torch.equal(target.state_dict()[k],v)
    bad=dict(state);bad.pop(next(iter(bad)))
    with pytest.raises(ValueError):load_relation_weights(target,dict(state_dict=bad))


def test_matched_configs():
    from train_neutrino_backend import deep_update,read_yaml,read_overlay_yaml
    root=Path(__file__).resolve().parents[1]
    configs=[deep_update(read_yaml(root/'config/train_diffusion_nersc.yaml'),read_overlay_yaml(root/f'config/train_diffusion_relation_{m}.yaml')) for m in ['context','relations']]
    a,b=configs
    assert a['platform']==b['platform'] and a['platform']['number_of_workers']==16
    for c in configs:
        t=c['options']['Training']
        assert t['epochs']==t['total_epochs']==50 and t['strict_relation_source']
        assert t['JointCoverage']['enable'] and t['JointCoverage']['every_n_epochs']==50
        assert not t['EMA']['enable'] and not c['rl']['enabled']
        assert len(c['logger']['wandb']['run_name'])<96
    x,y=copy.deepcopy(a['options']['Training']),copy.deepcopy(b['options']['Training'])
    x.pop('model_checkpoint_save_path');y.pop('model_checkpoint_save_path');assert x==y


def test_supervised_famo_auxiliary_state_is_not_a_model_mismatch():
    source=wrap(branch());target=wrap(branch('relations'))
    state={f'model.{k}':v.clone() for k,v in source.state_dict().items()}
    state['model.famo.w.neutrino'] = torch.tensor(.7)
    load_relation_weights(target,dict(state_dict=state))
    for k,v in source.state_dict().items():
        assert torch.equal(target.state_dict()[k],v)
    assert all(torch.count_nonzero(p)==0 for head in target.TruthGeneration.visible_conditioning.relation_adapter.outputs for p in head.parameters())


@pytest.mark.parametrize('extra',['famo.unexpected','TruthGeneration.unexpected','Classification.weight'])
def test_other_unexpected_checkpoint_keys_still_rejected(extra):
    state=dict(wrap(branch()).state_dict());state[extra]=torch.ones(1)
    with pytest.raises(ValueError,match='unexpected=') as error:
        load_relation_weights(wrap(branch('relations')),dict(state_dict=state))
    assert extra in str(error.value)
