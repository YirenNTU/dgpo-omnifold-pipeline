"""Prepare and audit the opt-in MET FiLM experiment without starting training."""
from pathlib import Path
import json
import numpy as np
import torch

from evenet.network.body.met_conditioning import FIELDS, met_spec, load_met_weights


def global_names(runtime):
    return [name for group in runtime['event_info']['INPUTS']['GLOBAL'].values() for name in group]


def classifier_met_contract(head, runtime):
    from scripts.conditional_tau_preprocessing import layout
    from scripts.train_conditional_spin_ratio import build_classifier
    names = global_names(runtime)
    met_spec(dict(enabled=True), names)
    spans, _ = layout(head['packing_spec'])
    span, shape = spans['conditions']
    if shape != (len(names),):raise ValueError('Classifier globals differ from policy schema')
    indices = [span.start+names.index(k) for k in FIELDS]
    with torch.random.fork_rng(devices=[]):
        model = build_classifier(head)
    base = model.base
    continuous = base.continuous_condition_columns.tolist()
    if not set(indices).issubset(continuous):raise ValueError('MET fields excluded from classifier conditioning')
    if any(branch.condition[0].in_features != len(continuous) for branch in (model.spin, model.tau)):
        raise ValueError('Spin/Tau conditioning omits global inputs')
    return dict(fields=list(FIELDS), packed_condition_columns=indices,
                branches=['General', 'Spin', 'Tau'], existing_inputs=True,
                transform='Classifier fit-only mean/scale and mask, shared between positive and negative samples')


def prepare_policy_source(runtime_path, source_path):
    from evenet.control.global_config import Config
    from RL.DGPO_neutrino.model_utils import build_evenet_on_device, load_normalization_dict
    cfg = Config();cfg.load_yaml(runtime_path)
    destination = Path(cfg.options.Training.model_checkpoint_load_path)
    if destination.exists():raise ValueError('MET policy source already exists; never overwrite')
    source = torch.load(source_path, map_location='cpu', weights_only=False, mmap=True)
    if (source.get('epoch'),source.get('global_step')) != (286,18655) or source.get('dgpo_checkpoint_version'):
        raise ValueError('MET ablation starts from raw supervised epoch287')
    model = build_evenet_on_device(cfg, load_normalization_dict(cfg), torch.device('cpu'))
    report = load_met_weights(model, source, allow_new=True)
    # This is a weights-only source, never an optimizer or training resume.
    payload = dict(epoch=286, global_step=18655, state_dict=model.state_dict(),
                   met_source_checkpoint=str(source_path),
                   met_conditioning=model.TruthGeneration.visible_conditioning.met_conditioning.spec)
    pending = destination.with_suffix('.pending.ckpt')
    torch.save(payload, pending);pending.replace(destination)
    return report


def preflight_met_inputs(model, reward, panel_path, runtime, *, initial):
    """Real cached events: source parity, classifier MET gradients, actor gradients."""
    from RL.DGPO_neutrino.conditional_tau_cycle import panel_batch
    contract = classifier_met_contract(reward.installed_head, runtime)
    with np.load(panel_path, allow_pickle=False) as z:
        panel = {k:z[k][:8] for k in ('raw_condition','category','visible_a','visible_b','truth_deltas','condition','candidate_truth')}
    batch = panel_batch(panel, np.arange(len(panel['category'])), reward.spec, torch.device('cpu'))
    observed = reward.prepare_inputs(batch)['condition']
    np.testing.assert_allclose(observed, panel['condition'], rtol=1e-6, atol=1e-6)
    # Verify every branch receives the observed MET coordinates through FiLM.
    condition = torch.tensor(observed, requires_grad=True)
    candidate = torch.tensor(panel['candidate_truth'])
    head = reward.head
    labels = torch.nn.functional.one_hot(condition[:,head.base.process_label_index].long(), head.classes).to(condition)
    context = condition[:,head.base.continuous_condition_columns]
    scores = dict(General=head.base(condition,candidate[:,:head.base_dim]),
                  Spin=head.spin(candidate[:,head.base_dim:head.base_dim+15],context,labels),
                  Tau=head.tau(candidate[:,head.base_dim+15:],context,labels))
    derivatives = {}
    for name, value in scores.items():
        grad, = torch.autograd.grad(value.sum(), condition, retain_graph=True)
        selected = grad[:,contract['packed_condition_columns']]
        if not torch.isfinite(selected).all():raise ValueError('Nonfinite classifier MET derivative')
        derivatives[name] = selected.abs().mean(0).tolist()
    model.eval();noise=torch.zeros(len(condition),2,model.invisible_input_dim)
    mask=torch.ones(len(condition),2,1);error=0.
    if initial:
        with torch.no_grad():
            for time in (.1,.5,.9):
                args=dict(noise_x=noise,cond_x=batch,time=torch.full((len(condition),),time),mode='neutrino',noise_mask=mask)
                a=model.predict_diffusion_vector(**args);b=reward.backbone.predict_diffusion_vector(**args)
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                error=max(error,float((a-b).abs().max()))
    # Read-only gradient check; never takes an optimizer step or changes weights.
    branch=model.TruthGeneration.visible_conditioning.met_conditioning
    output_parameters=[p for h in branch.modulations for p in h.parameters()]
    previous=[p.requires_grad for p in output_parameters]
    try:
        for p in output_parameters:p.requires_grad_(True)
        value=model.predict_diffusion_vector(noise,batch,torch.full((len(condition),),.5),'neutrino',noise_mask=mask)
        gradient=torch.autograd.grad(value.square().mean(),output_parameters)
        norm=float(torch.stack([g.square().sum() for g in gradient]).sum().sqrt())
        if not np.isfinite(norm) or norm<=0:raise ValueError('MET FiLM cannot receive a gradient')
    finally:
        for p,flag in zip(output_parameters,previous):p.requires_grad_(flag)
    return dict(classifier=contract,classifier_mean_abs_input_gradient=derivatives,
                source_velocity_parity_max_error=error if initial else None,met_output_gradient_norm=norm,
                initial_policy_preserved=initial,optimizer_steps=0)
