"""Process-label preservation, learned FiLM embedding and fresh-reward guards."""
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from scripts.train_dgpo_tau_process_labels import (
    assemble_runtime, validate_settings, snapshot_source_runtime, deterministic_policy_network,
)
from scripts.train_neutrino_backend import read_overlay_yaml, read_yaml, deep_update
from scripts.train_conditional_spin_ratio import build_classifier
from scripts.test_dgpo_tau_ratio import live_batch, example_bundle, fake_hidden
from scripts.conditional_tau_preprocessing import fit_masked_feature, layout
from RL.DGPO_neutrino.conditional_tau_reward import (
    ConditionalTauReward, canonical_policy_batch, condition_inputs, validate_head,
    SUPERVISED_POLICY_CONDITIONING_CONTRACT as CONTRACT,
)
from RL.DGPO_neutrino.conditional_tau_cycle import panel_batch, ConditionalTauCycle
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec, pack_event_inputs

ROOT=Path(__file__).resolve().parents[1]


def bundle(fitted=True):
    b=example_bundle()
    spec=EventPackingSpec({**b['head']['packing_spec']['shapes'],'conditions_mask':(1,), 'classification':()})
    h=b['head']
    h.update(packing_spec=spec.to_dict(),condition_dim=spec.width+35,num_process_classes=19,
        policy_conditioning_contract=CONTRACT,process_label_conditioning=True,
        fitting_current_policy=fitted)
    h['state_dict']=build_classifier(h).state_dict()
    b.update(policy_conditioning_contract=CONTRACT,source_run='xgp6ls8b',source_global_step=18655,
        source_checkpoint='/tmp/source-epoch287.ckpt',condition_mean=torch.zeros(spec.width+35),
        condition_scale=torch.ones(spec.width+35))
    return b


def batch():
    return dict(live_batch(),classification=torch.tensor([0,5,8,13,15]),conditions_mask=torch.ones(5,1,dtype=torch.bool))


@pytest.mark.parametrize('bad',[None,[-1,0,1,2,3],[0,1,2,3,19],[0,1,2,3,4.5]])
def test_missing_or_invalid_process_labels_fail_without_fallback(bad):
    b=batch()
    if bad is None: b.pop('classification')
    else: b['classification']=torch.tensor(bad)
    with pytest.raises(ValueError,match='labels|classification'):
        canonical_policy_batch(b)


def test_panel_and_live_inputs_keep_exact_labels_and_learned_embedding():
    b=batch(); r=ConditionalTauReward(bundle(),torch.nn.Linear(1,1),'cpu',microbatch=2)
    packed,_=pack_event_inputs(b,r.spec)
    panel=dict(raw_condition=packed.numpy(),category=b['event_category'].numpy())
    for leg in ('a','b'):
        panel['visible_'+leg]=torch.stack([b[f'lead_{leg}_visible_{k}'] for k in ('E','px','py','pz')],-1).numpy()
    reconstructed=panel_batch(panel,np.arange(5),r.spec,torch.device('cpu'))
    torch.testing.assert_close(reconstructed['classification'],b['classification'])
    observed=[]
    def hidden(policy,data,delta):
        observed.extend(data['classification'].tolist())
        return fake_hidden(policy,data,delta)
    with patch('scripts.tau_backbone_alignment.candidate_hidden',side_effect=hidden):
        c,f=r.features(b,torch.zeros(5,2,2))
    assert observed==[0,5,8,13,15]
    np.testing.assert_array_equal(c[:,-19:],np.eye(19)[observed])
    # Input answer fields cannot change the conditioning; process labels can.
    leaked=dict(b,x_invisible=torch.full((5,2,2),999.),truth_cij=torch.ones(5,9))
    np.testing.assert_array_equal(r.prepare_inputs(leaked)['condition'],c)
    assert r.head.process_label_dense.weight.shape==(256,19)
    assert r.head.condition_encoder[0].in_features==c.shape[1]-20


def test_process_embedding_is_added_to_film_context_and_receives_gradients():
    cfg=bundle()['head']; model=build_classifier(cfg)
    b=batch(); spec=EventPackingSpec.from_dict(cfg['packing_spec'])
    packed,_=pack_event_inputs(b,spec)
    c=torch.tensor(condition_inputs(packed.numpy(),b['event_category'].numpy(),
        np.zeros(cfg['condition_dim']),np.ones(cfg['condition_dim']),spec.to_dict()))
    candidate=torch.randn(5,25)
    with torch.no_grad():
        for block in model.blocks: block.context.weight.normal_(0,.05)
    expected=(model.condition_encoder(c[:,model.continuous_condition_columns])+
        model.process_label_dense(torch.nn.functional.one_hot(b['classification'],19).float()))
    contexts=[]
    handle=model.blocks[0].register_forward_pre_hook(lambda module,args:contexts.append(args[1]))
    try: scores=model(c,candidate)
    finally: handle.remove()
    torch.testing.assert_close(contexts[0],expected)
    scores.square().mean().backward()
    grad=model.process_label_dense.weight.grad
    assert torch.isfinite(grad).all() and grad[:,b['classification']].norm()>0
    changed=c.clone(); index=model.process_label_index
    changed[:,index]=(changed[:,index]+1)%19
    assert not torch.allclose(model(c,candidate),model(changed,candidate))


def test_categorical_ids_are_not_standardized_and_head_contract_cannot_mix():
    saved=bundle(); spec=saved['head']['packing_spec']; b=batch()
    packed,_=pack_event_inputs(b,EventPackingSpec.from_dict(spec))
    raw=condition_inputs(packed.numpy(),b['event_category'].numpy(),saved['condition_mean'],saved['condition_scale'],spec)
    mean,scale=fit_masked_feature(raw,np.ones(5,bool),spec)
    span=layout(spec)[0]['classification'][0]
    np.testing.assert_array_equal(mean[span],0)
    np.testing.assert_array_equal(scale[span],1)
    bad=copy.deepcopy(saved['head']); bad['process_label_conditioning']=False
    with pytest.raises(ValueError,match='Process-conditioned'): validate_head(bad)
    reward=ConditionalTauReward(saved,torch.nn.Linear(1,1),'cpu')
    with pytest.raises(ValueError,match='different process-label'):
        reward.install(example_bundle()['head'],policy_step=0,epoch=-1)
    old=copy.deepcopy(reward.stack_payload()); old['policy_conditioning_contract']='saved_tau_denominator_label0_v1'
    with pytest.raises(ValueError,match='classification contract'): reward.load_stack_payload(old)


def test_unfitted_head_cannot_reward_actor_and_resume_keeps_fitted_embedding():
    pending=ConditionalTauReward(bundle(False),torch.nn.Linear(1,1),'cpu')
    with pytest.raises(ValueError,match='Fit the process-conditioned'):
        pending.compute(torch.zeros(8,5,2,2),batch())
    fitted=bundle()['head']; pending.install(fitted,policy_step=0,epoch=-1)
    clone=ConditionalTauReward(bundle(False),torch.nn.Linear(1,1),'cpu')
    clone.load_stack_payload(pending.stack_payload())
    torch.testing.assert_close(clone.head.process_label_dense.weight,pending.head.process_label_dense.weight)
    assert clone.installed_head['fitting_current_policy'] and clone.round_id==1


def test_runtime_restores_source_architecture_over_inherited_depth3(tmp_path):
    overlay=read_overlay_yaml(ROOT/'config/dgpo_tau_process_labels_film.yaml')
    validate_settings(overlay)
    source=deep_update(read_yaml(ROOT/'config/train_diffusion_nersc.yaml'),
                       read_overlay_yaml(ROOT/'config/train_diffusion_film_10pct_foundation.yaml'))
    cfg=assemble_runtime(source,overlay,tmp_path)
    assert cfg['network']==deterministic_policy_network(source['network'])
    assert cfg['dgpo']['require_deterministic_policy'] is True
    assert cfg['options']['Dataset']==source['options']['Dataset']
    assert cfg['network']['TruthGeneration']['num_layers']==1
    assert cfg['platform']['number_of_workers']==16
    assert cfg['dgpo']['tau_ratio']['policy_conditioning_contract']==CONTRACT
    assert cfg['dgpo']['tau_ratio']['minimum_selected_steps']==1000
    assert cfg['dgpo']['K']==8 and cfg['dgpo']['reference_trust']['coefficient']==1
    assert cfg['dgpo']['checkpoint_load_mode']=='weights_only'
    assert cfg['options']['Training']['model_checkpoint_load_path']==str(tmp_path/'source.ckpt')
    assert '1110' not in cfg['options']['Dataset']['normalization_file']


def test_bootstrap_precedes_updates_and_fitted_resume_does_not_refit():
    cycle=object.__new__(ConditionalTauCycle)
    cycle.reward=ConditionalTauReward(bundle(False),torch.nn.Linear(1,1),'cpu')
    cycle.cfg=dict(policy_conditioning_contract=CONTRACT,train_panel='training',fit={'min_steps':1000})
    cycle.validation={'source_ids':np.arange(5)}
    calls=[]
    cycle.read_panel=lambda _: {'source_ids':np.arange(5)}
    cycle._prepare_truth_features=lambda p,label:calls.append(label)
    cycle.verify_initial_policy=lambda:calls.append('source-parity')
    def fit(epoch,step,force):
        calls.append('fit-before-update')
        h=bundle()['head']; h['optimizer_steps']=1000
        cycle.reward.install(h,policy_step=step,epoch=epoch)
    cycle.refit=fit
    assert cycle.bootstrap_supervised_reward(-1,0)
    assert calls==['train','validation','source-parity','fit-before-update']
    calls.clear()
    assert not cycle.bootstrap_supervised_reward(9,100)
    assert calls==['train','validation']


def test_source_snapshot_pins_default_class_mapping_and_normalizer(tmp_path):
    import yaml
    defaults=tmp_path/'network.yaml'; defaults.write_text('TruthGeneration: {hidden_dim: 256, num_layers: 3}\n')
    schema=tmp_path/'labels.yaml'; schema.write_text('labels: [a,b,c]\n')
    normalizer=tmp_path/'original.pt'; normalizer.write_bytes(b'original pinned normalizer')
    runtime=tmp_path/'runtime.yaml'
    runtime.write_text(yaml.safe_dump(dict(network={'default':str(defaults),'TruthGeneration':{'num_layers':1}},
        event_info={'default':str(schema)},options={'Dataset':{'normalization_file':str(normalizer)}})))
    output=tmp_path/'output'; output.mkdir()
    pinned=snapshot_source_runtime(runtime,output)
    defaults.write_text('TruthGeneration: {num_layers: 7}\n'); schema.write_text('labels: [c,b,a]\n')
    actual=read_yaml(output/'source_runtime.yaml')
    assert actual['network']['TruthGeneration']=={'hidden_dim':256,'num_layers':1}
    assert actual['event_info']['labels']==['a','b','c']
    assert (output/'normalization.pt').read_bytes()==normalizer.read_bytes()
    assert actual==pinned
