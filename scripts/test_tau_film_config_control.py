"""Old checkpoint compatibility and single-application runtime FiLM gains."""
from types import SimpleNamespace
import pytest
import torch
from evenet.network.body.visible_conditioning import VisibleConditioning,visible_conditioning_spec,diffusion_film_gains,FILM_GAIN_KEYS
from evenet.network.heads.generation.generation_head import EventGenerationHead


def head(gains=None):
    spec=dict(feature_names=['Part_eta','Part_phi'],token_dim=8,hidden_dim=8,num_layers=1,n_branches=2,width=8)
    if gains is not None:spec['film_gains']=gains
    return EventGenerationHead(8,8,4,3,2,1,2,0.,True,.3,0.,0.,visible_conditioning=spec).eval()


@pytest.mark.parametrize('gains',[
    dict(ffn_gamma=0.,ffn_beta=0.),dict(ffn_gamma=.6,ffn_beta=.6),
    dict(ffn_gamma=0.),dict(ffn_beta=0.),dict(attention_gamma=0.,attention_beta=0.),
    {k:0. for k in FILM_GAIN_KEYS}])
def test_original_generation_head_equals_independent_coefficient_hook_without_weight_change(gains):
    torch.manual_seed(21);old=head();new=head(gains)
    with torch.no_grad():
        old.visible_conditioning.modulations[0].weight.normal_(std=.2)
        old.visible_conditioning.modulations[0].bias.normal_(std=.2)
    new.load_state_dict(old.state_dict(),strict=True)
    assert old.state_dict().keys()==new.state_dict().keys()
    args=dict(x=torch.randn(2,5,8),global_cond=torch.randn(2,1,4),global_cond_mask=torch.ones(2,1,1),
        num_x=torch.full((2,),5.),x_mask=torch.ones(2,5,1),time=torch.full((2,),.6),label=torch.tensor([[0],[2]]),
        visible_raw=torch.randn(2,3,2),visible_tokens=torch.randn(2,3,8),visible_mask=torch.ones(2,3,1),
        time_masking=torch.cat([torch.zeros(2,3,1),torch.ones(2,2,1)],dim=1))
    values=diffusion_film_gains(gains)
    def hook(module,args,kwargs):
        parts=kwargs['modulation']
        return args,{**kwargs,'modulation':tuple(p if values[k]==1 else p*values[k] for p,k in zip(parts,FILM_GAIN_KEYS))}
    with torch.no_grad():
        handle=old.gen_transformer_blocks[0].register_forward_pre_hook(hook,with_kwargs=True)
        try:expected=old(**args)
        finally:handle.remove()
        actual=new(**args)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    # Gain .6 must be applied once, rather than .6^2 at two call sites.
    for k,v in old.state_dict().items():torch.testing.assert_close(new.state_dict()[k],v,rtol=0,atol=0)


def test_identity_defaults_preserve_old_spec_rng_and_tensor_ownership():
    torch.manual_seed(8);old=head()
    torch.manual_seed(8);new=head({k:1. for k in FILM_GAIN_KEYS})
    assert old.visible_conditioning.spec==new.visible_conditioning.spec
    assert old.state_dict().keys()==new.state_dict().keys()
    for k,v in old.state_dict().items():torch.testing.assert_close(new.state_dict()[k],v,rtol=0,atol=0)


def test_diffusion_only_schema_and_legacy_zero_flags_take_precedence():
    cfg=SimpleNamespace(VisibleConditioning=dict(diffusion_enabled=True,classifier_enabled=True,
        diffusion_film_gains={'ffn_gamma':.6}))
    args=dict(feature_names=['Part_eta','Part_phi'],token_dim=8,hidden_dim=8,num_layers=1,n_branches=2)
    assert visible_conditioning_spec(cfg,target='diffusion',**args)['film_gains']['ffn_gamma']==.6
    assert 'film_gains' not in visible_conditioning_spec(cfg,target='classifier',**args)
    obj=VisibleConditioning(**args,ffn_film_enabled=False,film_gains={'ffn_gamma':2.})
    parts=tuple(torch.ones(2,8) for _ in range(4))
    got=obj.apply_runtime_gains(obj.apply_scale_policy(parts))
    assert got[2].count_nonzero()==0 and got[3].count_nonzero()==0
    obj=VisibleConditioning(**args,film_scale_enabled=False,film_gains={'attention_gamma':2.,'ffn_gamma':2.})
    got=obj.apply_runtime_gains(obj.apply_scale_policy(parts))
    assert got[0].count_nonzero()==0 and got[2].count_nonzero()==0


@pytest.mark.parametrize('bad',[{'typo':0.},{'ffn_gamma':True},{'ffn_beta':-1.},{'attention_gamma':float('nan')},{'ffn_gamma':'0'},[0,1],{'ffn_gamma':float('inf')}])
def test_invalid_controls_fail_closed(bad):
    with pytest.raises(ValueError):diffusion_film_gains(bad)
